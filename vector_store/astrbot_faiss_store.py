import os
import asyncio
import json
import gc
import time
import psutil  # 用于内存监控
from typing import List, Dict, Tuple, Set, Optional

# 引入 cachetools 和 Lock, Set, Optional
try:
    from cachetools import LRUCache
except ImportError:
    raise ImportError("Please install cachetools: pip install cachetools")

from .base import (
    VectorDBBase,
    Document,
    ProcessingBatch,
    DEFAULT_BATCH_SIZE,
)
from astrbot.api import logger
from astrbot.core.db.vec_db.faiss_impl import FaissVecDB
from astrbot.core.provider.provider import EmbeddingProvider
from ..utils.embedding import EmbeddingSolutionHelper
from .faiss_store import FaissStore as OldFaissStore

# 定义默认的缓存大小
DEFAULT_MAX_CACHE_SIZE = 3
# 定义内存管理相关常量
DEFAULT_MEMORY_BATCH_SIZE = 50  # 内存中同时处理的文档数量
MAX_DOCUMENTS_WARNING_THRESHOLD = 5000  # 大文件警告阈值
# 高效处理相关常量
OPTIMAL_CONCURRENT_TASKS = 3  # 最优并发任务数(基准值)
EMBEDDING_BATCH_SIZE = 15  # embedding批量处理大小
MEMORY_CHECK_INTERVAL = 50  # 每处理多少文档检查一次内存
# 动态内存管理常量
DEFAULT_MAX_MEMORY_MB = 2048  # 默认最大内存限制(MB)
MIN_CONCURRENT_TASKS = 2  # 最小并发任务数
MAX_CONCURRENT_TASKS = 12  # 最大并发任务数
MEMORY_SAFETY_FACTOR = 0.7  # 可用内存的安全使用比例
# 流式并发处理常量
DEFAULT_RPM_LIMIT = 2000  # 默认每分钟请求限制
EMBEDDING_RETRY_DELAY = 1.0  # 限制重试延迟(秒)
EMBEDDING_MAX_RETRIES = 3  # 最大重试次数
# 更精确的API限制错误检测
RATE_LIMIT_KEYWORDS = [
    # 通用限制错误
    "rate limit", "rate_limit", "too many requests", "quota", "429", 
    "rate_limit_exceeded", "throttle", "throttling", "exceeded",
    # OpenAI相关
    "rate limit exceeded", "requests per minute", "rpm", "tpm",
    "rate limited", "quota exceeded", "usage quota",
    # 其他API提供商
    "concurrency limit", "request limit", "api limit",
    "frequency limit", "call limit", "requests exceeded",
    # HTTP状态码和响应
    "http 429", "status 429", "429 too many",
    # 中文错误信息
    "请求过于频繁", "访问频率限制", "超出配额", "请求限制"
]
# API限制错误的详细分类
CRITICAL_RATE_LIMIT_KEYWORDS = [
    "429", "rate limit exceeded", "quota exceeded", "rpm", "tpm"
]


def _is_rate_limit_error(error: Exception) -> tuple[bool, bool]:
    """
    智能检测是否为API限制错误
    返回: (is_rate_limit, is_critical)
    - is_rate_limit: 是否为限制错误
    - is_critical: 是否为严重限制错误（需要立即大幅减少并发）
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # 检查错误消息
    is_rate_limit = any(keyword in error_str for keyword in RATE_LIMIT_KEYWORDS)
    is_critical = any(keyword in error_str for keyword in CRITICAL_RATE_LIMIT_KEYWORDS)
    
    # 检查异常类型
    if not is_rate_limit:
        rate_limit_types = [
            "ratelimiterror", "quotaerror", "throttleerror", 
            "httperror", "apierror", "requestsexception"
        ]
        is_rate_limit = any(rate_type in error_type for rate_type in rate_limit_types)
    
    # 检查HTTP状态码（如果有）
    if hasattr(error, 'status_code'):
        if error.status_code == 429:
            is_rate_limit = True
            is_critical = True
    
    return is_rate_limit, is_critical


def _check_pickle_file(file_path: str) -> bool:
    """检查文件是否为 Pickle 格式"""
    try:
        if not os.path.exists(file_path):
            return False
        with open(file_path, "rb") as f:
            magic = f.read(2)
            # 兼容python3.8之前的pickle协议版本
            return magic in [b"\x80\x04", b"\x80\x03", b"\x80\x02"]
    except Exception:
        return False


def _get_memory_usage_mb() -> float:
    """获取当前进程的内存使用量(MB)"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # 转换为MB
    except Exception:
        return 0.0


def _get_system_memory_info() -> Tuple[float, float, float]:
    """
    获取系统内存信息
    返回: (总内存MB, 可用内存MB, 内存使用率%)
    """
    try:
        memory = psutil.virtual_memory()
        total_mb = memory.total / 1024 / 1024
        available_mb = memory.available / 1024 / 1024
        usage_percent = memory.percent
        return total_mb, available_mb, usage_percent
    except Exception:
        return 0.0, 0.0, 0.0


def _calculate_optimal_concurrency(max_memory_limit_mb: float = DEFAULT_MAX_MEMORY_MB) -> Tuple[int, float, str]:
    """
    基于系统内存动态计算最优并发任务数
    
    Args:
        max_memory_limit_mb: 用户设置的最大内存限制(MB)
    
    Returns:
        (optimal_tasks, memory_limit_mb, decision_reason)
    """
    try:
        total_mb, available_mb, usage_percent = _get_system_memory_info()
        current_process_mb = _get_memory_usage_mb()
        
        # 计算可用内存的70%
        safe_available_mb = available_mb * MEMORY_SAFETY_FACTOR
        
        # 在用户限制和安全可用内存之间选择较小者
        effective_limit_mb = min(max_memory_limit_mb, safe_available_mb)
        
        # 估算每个并发任务的内存开销(基于经验值)
        estimated_task_memory_mb = 50  # 每个任务大约50MB内存开销
        
        # 计算可支持的并发任务数
        max_tasks_by_memory = max(1, int(effective_limit_mb / estimated_task_memory_mb))
        
        # 应用并发任务数限制
        optimal_tasks = min(max_tasks_by_memory, MAX_CONCURRENT_TASKS)
        optimal_tasks = max(optimal_tasks, MIN_CONCURRENT_TASKS)
        
        # 生成决策原因
        if effective_limit_mb == max_memory_limit_mb:
            reason = f"用户限制({max_memory_limit_mb:.0f}MB)"
        else:
            reason = f"系统可用({safe_available_mb:.0f}MB)"
        
        logger.debug(f"[内存分析] 系统内存: {total_mb:.0f}MB, 可用: {available_mb:.0f}MB ({100-usage_percent:.1f}%空闲), "
                    f"当前进程: {current_process_mb:.0f}MB, 有效限制: {effective_limit_mb:.0f}MB({reason}), "
                    f"计算并发数: {optimal_tasks}")
        
        return optimal_tasks, effective_limit_mb, reason
        
    except Exception as e:
        logger.warning(f"[内存分析] 内存分析失败，使用默认并发数: {e}")
        return OPTIMAL_CONCURRENT_TASKS, max_memory_limit_mb, "内存分析失败"


def _should_trigger_gc(processed_count: int, memory_threshold_mb: float = 1000) -> bool:
    """判断是否应该触发垃圾回收"""
    return (processed_count % MEMORY_CHECK_INTERVAL == 0 and 
            _get_memory_usage_mb() > memory_threshold_mb)


class StreamingConcurrencyController:
    """流式并发控制器 - 快速响应的动态调节策略"""
    
    def __init__(self, rpm_limit: int = DEFAULT_RPM_LIMIT):
        self.rpm_limit = rpm_limit
        self.safe_rps = (rpm_limit * 0.8) / 60  # 安全请求率 (80%裕量)
        self.current_concurrency = 3  # 初始并发数
        self.success_count = 0
        self.failure_count = 0
        self.rate_limit_failures = 0  # 专门记录限制错误
        self.last_adjust_time = time.time()
        self.adjust_interval = 3.0  # 3秒快速调整间隔
        self.recent_requests = []  # 记录最近请求的时间戳，用于RPS计算
        self.window_size = 30.0  # 30秒滑动窗口
        
        logger.info(f"[流式并发控制] 快速调节模式: RPM限制={rpm_limit}, 安全RPS={self.safe_rps:.1f}, 初始并发={self.current_concurrency}")
    
    def record_success(self):
        """记录成功请求"""
        now = time.time()
        self.success_count += 1
        self.recent_requests.append(now)
        self._cleanup_old_requests(now)
        
    def record_failure(self, is_rate_limit: bool = False, is_critical: bool = False):
        """记录失败请求"""
        now = time.time()
        self.failure_count += 1
        if is_rate_limit:
            self.rate_limit_failures += 1
            # 如果遇到严重限制错误，立即触发调节
            if is_critical:
                self.last_adjust_time = now - self.adjust_interval
    
    def _cleanup_old_requests(self, now: float):
        """清理窗口外的请求记录"""
        cutoff = now - self.window_size
        self.recent_requests = [t for t in self.recent_requests if t > cutoff]
    
    def get_current_rps(self) -> float:
        """计算当前实际RPS"""
        if len(self.recent_requests) < 2:
            return 0.0
        return len(self.recent_requests) / min(self.window_size, time.time() - self.recent_requests[0])
    
    def should_adjust_concurrency(self) -> bool:
        """判断是否需要调整并发数 - 更频繁的检查"""
        now = time.time()
        
        # 快速检查条件
        if (now - self.last_adjust_time) >= self.adjust_interval:
            return True
        
        # 如果遇到限制错误，立即调节
        if self.rate_limit_failures > 0:
            return True
            
        return False
    
    def calculate_new_concurrency(self) -> int:
        """基于启发式规则计算新的并发数 - 更激进的调节策略"""
        total_requests = self.success_count + self.failure_count
        
        # 对于少量样本也进行调节
        if total_requests < 3:
            return self.current_concurrency
            
        success_rate = self.success_count / total_requests
        rate_limit_rate = self.rate_limit_failures / total_requests
        current_rps = self.get_current_rps()
        
        new_concurrency = self.current_concurrency
        
        # 规则1: 遇到任何限制错误都立即减少并发
        if self.rate_limit_failures > 0:
            reduction = max(1, self.rate_limit_failures)  # 按失败次数减少
            new_concurrency = max(1, self.current_concurrency - reduction)
            logger.warning(f"[流式并发控制] API限制错误{self.rate_limit_failures}次，立即减少并发: {self.current_concurrency} → {new_concurrency}")
        
        # 规则2: 成功率很高且RPS未达上限，激进增加并发
        elif success_rate > 0.9 and current_rps < self.safe_rps * 0.8 and self.current_concurrency < 8:
            # 根据成功率决定增加幅度
            if success_rate > 0.98:
                increase = 2  # 非常高成功率，快速增长
            else:
                increase = 1
            new_concurrency = min(8, self.current_concurrency + increase)
            logger.info(f"[流式并发控制] 高成功率({success_rate:.1%})且RPS未满({current_rps:.1f}/{self.safe_rps:.1f})，增加并发: {self.current_concurrency} → {new_concurrency}")
        
        # 规则3: 成功率偏低，减少并发
        elif success_rate < 0.85:
            reduction = 2 if success_rate < 0.7 else 1  # 成功率很低时更大幅度减少
            new_concurrency = max(1, self.current_concurrency - reduction)
            logger.warning(f"[流式并发控制] 成功率低({success_rate:.1%})，减少并发: {self.current_concurrency} → {new_concurrency}")
        
        # 规则4: RPS接近上限，预防性减少并发
        elif current_rps > self.safe_rps * 0.9:
            new_concurrency = max(1, self.current_concurrency - 1)
            logger.info(f"[流式并发控制] RPS接近上限({current_rps:.1f}/{self.safe_rps:.1f})，预防性减少并发: {self.current_concurrency} → {new_concurrency}")
        
        return new_concurrency
    
    def adjust_concurrency(self) -> int:
        """调整并发数并返回新值 - 快速重置统计"""
        if not self.should_adjust_concurrency():
            return self.current_concurrency
            
        old_concurrency = self.current_concurrency
        self.current_concurrency = self.calculate_new_concurrency()
        
        if self.current_concurrency != old_concurrency:
            # 部分重置统计计数器，保留一些历史信息用于RPS计算
            self.success_count = max(0, self.success_count // 2)  # 保留一半历史
            self.failure_count = max(0, self.failure_count // 2)
            self.rate_limit_failures = 0  # 限制错误立即清零
            
        self.last_adjust_time = time.time()
        return self.current_concurrency


class StreamingDocumentProcessor:
    """内存安全的流式文档处理器 - 始终保持固定数量的请求在执行，严格控制内存使用"""
    
    def __init__(self, vecdb: FaissVecDB, collection_name: str, rpm_limit: int = DEFAULT_RPM_LIMIT, max_memory_mb: float = DEFAULT_MAX_MEMORY_MB):
        self.vecdb = vecdb
        self.collection_name = collection_name
        self.controller = StreamingConcurrencyController(rpm_limit)
        self.document_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.active_consumers = set()
        self.total_documents = 0
        self.processed_documents = 0
        
        # 内存安全配置
        self.max_memory_mb = max_memory_mb
        self.initial_memory_mb = _get_memory_usage_mb()
        self.last_memory_check = time.time()
        self.memory_check_interval = 5.0  # 5秒检查一次内存
        self.high_memory_threshold = max_memory_mb * 0.8  # 80%内存使用率告警
        self.critical_memory_threshold = max_memory_mb * 0.95  # 95%使用率紧急处理
        
        # 资源管理
        self.processed_docs_cleanup_count = 0
        self.cleanup_interval = 20  # 每处理20个文档进行一次资源清理
        
        logger.info(f"[内存安全流式处理] 初始化: RPM限制={rpm_limit}, 内存限制={max_memory_mb}MB, "
                   f"初始内存={self.initial_memory_mb:.1f}MB")
        
    def _check_memory_status(self) -> tuple[bool, float, str]:
        """
        检查内存状态
        返回: (需要采取行动, 当前内存使用MB, 状态描述)
        """
        current_memory = _get_memory_usage_mb()
        memory_increase = current_memory - self.initial_memory_mb
        
        if memory_increase > self.critical_memory_threshold:
            return True, current_memory, f"紧急: 内存增长{memory_increase:.1f}MB，超过临界阈值"
        elif memory_increase > self.high_memory_threshold:
            return True, current_memory, f"告警: 内存增长{memory_increase:.1f}MB，接近限制"
        else:
            return False, current_memory, f"正常: 内存增长{memory_increase:.1f}MB"
    
    async def _memory_monitor(self):
        """内存监控协程 - 持续监控内存使用情况"""
        while True:
            try:
                await asyncio.sleep(self.memory_check_interval)
                
                needs_action, current_memory, status = self._check_memory_status()
                
                if needs_action:
                    logger.warning(f"[内存安全流式处理] 内存监控: {status}")
                    
                    # 如果内存使用过高，触发垃圾回收
                    if current_memory - self.initial_memory_mb > self.critical_memory_threshold:
                        logger.info(f"[内存安全流式处理] 触发紧急垃圾回收: 当前{current_memory:.1f}MB")
                        gc.collect()
                        after_gc_memory = _get_memory_usage_mb()
                        logger.info(f"[内存安全流式处理] GC完成: {current_memory:.1f}MB → {after_gc_memory:.1f}MB "
                                  f"(释放{current_memory-after_gc_memory:.1f}MB)")
                        
                        # 如果GC后内存仍然很高，减少并发数
                        if after_gc_memory - self.initial_memory_mb > self.high_memory_threshold:
                            old_concurrency = self.controller.current_concurrency
                            if old_concurrency > 1:
                                new_concurrency = max(1, old_concurrency - 1)
                                self.controller.current_concurrency = new_concurrency
                                await self._adjust_consumer_count(new_concurrency)
                                logger.warning(f"[内存安全流式处理] 内存压力过大，降低并发: {old_concurrency} → {new_concurrency}")
                else:
                    # 定期报告内存状态
                    if time.time() - self.last_memory_check > 30:  # 30秒报告一次
                        logger.debug(f"[内存安全流式处理] 内存状态: {status}")
                        self.last_memory_check = time.time()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[内存安全流式处理] 内存监控异常: {e}")
                await asyncio.sleep(self.memory_check_interval)
    
    async def _progress_monitor(self):
        """进度监控协程 - 检测处理停滞和死锁"""
        last_processed = 0
        stalled_count = 0
        
        while True:
            try:
                await asyncio.sleep(30)  # 30秒检查一次
                
                current_processed = self.processed_documents
                queue_size = self.document_queue.qsize()
                active_workers = len(self.active_consumers)
                
                # 检查是否有进度
                if current_processed == last_processed and queue_size > 0:
                    stalled_count += 1
                    logger.warning(f"[内存安全流式处理] ⚠️ 进度停滞检测: 连续{stalled_count}次无进展, "
                                 f"已处理{current_processed}/{self.total_documents}, 队列剩余{queue_size}, "
                                 f"活跃worker{active_workers}")
                    
                    # 连续3次无进展，可能死锁
                    if stalled_count >= 3:
                        logger.error(f"[内存安全流式处理] 💀 疑似死锁: 队列{queue_size}个文档，但无worker处理")
                        
                        # 尝试重启一个worker
                        if active_workers > 0 and queue_size > 0:
                            logger.info(f"[内存安全流式处理] 🔄 尝试重启worker解决死锁")
                            new_worker = asyncio.create_task(self._document_consumer(f"recovery-worker"))
                            self.active_consumers.add(new_worker)
                else:
                    # 有进展，重置计数器
                    if stalled_count > 0:
                        logger.info(f"[内存安全流式处理] ✅ 进度恢复: {current_processed}/{self.total_documents}")
                    stalled_count = 0
                
                last_processed = current_processed
                
                # 详细状态报告
                progress = (current_processed / self.total_documents * 100) if self.total_documents > 0 else 0
                logger.debug(f"[内存安全流式处理] 状态检查: 进度{progress:.1f}%, 队列{queue_size}, worker{active_workers}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[内存安全流式处理] 进度监控异常: {e}")
                await asyncio.sleep(30)
        
    async def process_documents_streaming(self, documents: List[Document]) -> Tuple[List[str], int]:
        """内存安全的流式处理所有文档 - 带超时和死锁防护"""
        if not documents:
            return [], 0
            
        self.total_documents = len(documents)
        start_time = time.time()
        
        logger.info(f"[内存安全流式处理] 🚀 开始处理 {self.total_documents} 个文档，初始并发数: {self.controller.current_concurrency}, "
                   f"内存限制: {self.max_memory_mb}MB")
        
        # 1. 将所有文档放入队列
        for doc in documents:
            await self.document_queue.put(doc)
        
        # 2. 启动初始消费者
        await self._start_consumers(self.controller.current_concurrency)
        
        # 3. 启动监控和调节器
        adjuster_task = asyncio.create_task(self._concurrency_adjuster())
        memory_monitor_task = asyncio.create_task(self._memory_monitor())
        progress_monitor_task = asyncio.create_task(self._progress_monitor())
        
        try:
            # 4. 等待所有文档处理完成，带超时保护
            timeout_seconds = max(300, self.total_documents * 2)  # 至少5分钟，每文档最多2秒
            logger.info(f"[内存安全流式处理] 设置超时: {timeout_seconds}秒")
            
            await asyncio.wait_for(self.document_queue.join(), timeout=timeout_seconds)
            logger.info(f"[内存安全流式处理] ✅ 所有文档处理完成")
            
        except asyncio.TimeoutError:
            remaining = self.document_queue.qsize()
            processed = self.total_documents - remaining
            logger.error(f"[内存安全流式处理] ⏰ 处理超时: 已处理{processed}/{self.total_documents}, 剩余{remaining}个文档")
            
            # 清空队列，防止join()继续阻塞
            while not self.document_queue.empty():
                try:
                    self.document_queue.get_nowait()
                    self.document_queue.task_done()
                except:
                    break
        
        except Exception as e:
            logger.error(f"[内存安全流式处理] 💥 处理异常: {type(e).__name__}: {e}")
        
        # 5. 清理任务
        adjuster_task.cancel()
        memory_monitor_task.cancel() 
        progress_monitor_task.cancel()
        await self._stop_all_consumers()
        
        # 6. 收集结果
        results = await self._collect_results()
        
        # 7. 最终内存清理
        documents.clear()
        gc.collect()
        final_memory = _get_memory_usage_mb()
        memory_delta = final_memory - self.initial_memory_mb
        
        # 8. 统计和日志
        end_time = time.time()
        processing_time = end_time - start_time
        success_count = len([r for r in results if r[0] == "success"])
        failed_count = len([r for r in results if r[0] == "failed"])
        
        logger.info(f"[内存安全流式处理] ✅ 处理完成: 成功{success_count}, 失败{failed_count}, "
                   f"用时{processing_time:.1f}秒, 平均{self.total_documents/processing_time:.1f}文档/秒, "
                   f"内存变化{memory_delta:+.1f}MB ({self.initial_memory_mb:.1f}→{final_memory:.1f}MB)")
        
        successful_doc_ids = [r[1] for r in results if r[0] == "success" and r[1]]
        return successful_doc_ids, failed_count
    
    async def _start_consumers(self, count: int):
        """启动指定数量的消费者"""
        for i in range(count):
            consumer = asyncio.create_task(self._document_consumer(f"worker-{i}"))
            self.active_consumers.add(consumer)
    
    async def _stop_all_consumers(self):
        """停止所有消费者"""
        for consumer in self.active_consumers:
            consumer.cancel()
        
        # 等待所有消费者停止
        if self.active_consumers:
            await asyncio.gather(*self.active_consumers, return_exceptions=True)
        self.active_consumers.clear()
    
    async def _concurrency_adjuster(self):
        """动态调整并发数的协程 - 快速响应模式"""
        while True:
            try:
                await asyncio.sleep(1)  # 1秒高频检查
                
                old_concurrency = self.controller.current_concurrency
                new_concurrency = self.controller.adjust_concurrency()
                
                if new_concurrency != old_concurrency:
                    await self._adjust_consumer_count(new_concurrency)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[流式文档处理] 并发调节器异常: {e}")
                await asyncio.sleep(1)  # 异常后短暂等待
    
    async def _adjust_consumer_count(self, target_count: int):
        """调整消费者数量"""
        current_count = len(self.active_consumers)
        
        if target_count > current_count:
            # 增加消费者
            for i in range(target_count - current_count):
                consumer = asyncio.create_task(self._document_consumer(f"worker-{current_count + i}"))
                self.active_consumers.add(consumer)
            logger.info(f"[流式文档处理] 增加消费者: {current_count} → {target_count}")
            
        elif target_count < current_count:
            # 减少消费者 (让一些消费者自然结束)
            consumers_to_stop = list(self.active_consumers)[target_count:]
            for consumer in consumers_to_stop:
                consumer.cancel()
                self.active_consumers.discard(consumer)
            logger.info(f"[流式文档处理] 减少消费者: {current_count} → {target_count}")
    
    async def _document_consumer(self, worker_id: str):
        """内存安全的文档处理消费者 - 及时释放资源，确保task_done调用"""
        processed_by_worker = 0
        last_cleanup = time.time()
        
        while True:
            doc = None
            try:
                # 直接从队列获取文档，无超时等待
                doc = await self.document_queue.get()
                
                # 预处理：记录文档大小以便内存监控
                doc_size_estimate = len(doc.text_content) if doc.text_content else 0
                
                try:
                    # 处理文档
                    result = await self._process_single_document_with_retry(doc, worker_id)
                    await self.result_queue.put(result)
                    
                    # 立即清理文档引用，确保内存释放
                    doc.text_content = ""
                    doc.metadata.clear()
                    del doc  # 显式删除文档对象
                    doc = None  # 防止finally中重复处理
                    
                    # 更新统计
                    processed_by_worker += 1
                    self.processed_documents += 1
                    self.processed_docs_cleanup_count += 1
                    
                    # 定期进行资源清理
                    if self.processed_docs_cleanup_count >= self.cleanup_interval:
                        current_time = time.time()
                        if current_time - last_cleanup > 10:  # 至少间隔10秒
                            gc.collect()  # 触发垃圾回收
                            self.processed_docs_cleanup_count = 0
                            last_cleanup = current_time
                            
                            memory_status = self._check_memory_status()
                            logger.debug(f"[内存安全流式处理] {worker_id} 定期清理: 已处理{processed_by_worker}个文档, {memory_status[2]}")
                    
                    # 更频繁的进度报告
                    if self.processed_documents % 15 == 0:  # 每15个文档报告一次
                        progress = (self.processed_documents / self.total_documents) * 100
                        remaining = self.total_documents - self.processed_documents
                        current_memory = _get_memory_usage_mb()
                        memory_delta = current_memory - self.initial_memory_mb
                        
                        logger.info(f"[内存安全流式处理] 进度: {self.processed_documents}/{self.total_documents} ({progress:.1f}%) "
                                  f"剩余{remaining}, 内存增长{memory_delta:+.1f}MB")
                
                except Exception as doc_error:
                    # 文档处理异常，记录失败结果
                    logger.error(f"[内存安全流式处理] {worker_id} 文档处理异常: {doc_error}")
                    error_result = ("failed", "", f"{type(doc_error).__name__}: {str(doc_error)}")
                    await self.result_queue.put(error_result)
                    
                    # 清理异常文档
                    if doc:
                        doc.text_content = ""
                        doc.metadata.clear()
                        del doc
                        doc = None
                
                finally:
                    # 无论成功失败，都必须调用task_done
                    try:
                        self.document_queue.task_done()
                    except ValueError as e:
                        logger.warning(f"[内存安全流式处理] {worker_id} task_done异常: {e}")
                
            except asyncio.CancelledError:
                logger.debug(f"[内存安全流式处理] {worker_id} 收到取消信号，已处理 {processed_by_worker} 个文档")
                # 如果还有文档在处理，需要标记完成
                if doc is not None:
                    try:
                        self.document_queue.task_done()
                    except ValueError:
                        pass
                break
            except Exception as e:
                logger.error(f"[内存安全流式处理] {worker_id} 消费者异常: {e}")
                # 确保异常情况下也调用task_done
                if doc is not None:
                    try:
                        self.document_queue.task_done()
                    except ValueError:
                        pass
                # 短暂等待后继续处理
                await asyncio.sleep(0.1)
    
    async def _process_single_document_with_retry(self, doc: Document, worker_id: str) -> Tuple[str, str, str]:
        """带智能重试的单文档处理"""
        for attempt in range(EMBEDDING_MAX_RETRIES + 1):
            try:
                doc_id = await self.vecdb.insert(
                    content=doc.text_content,
                    metadata=doc.metadata,
                )
                
                # 及时清理文档引用
                doc.text_content = ""
                doc.metadata.clear()
                
                # 记录成功
                self.controller.record_success()
                return ("success", doc_id, "")
                
            except Exception as e:
                # 智能错误检测
                is_rate_limit, is_critical = _is_rate_limit_error(e)
                
                if is_rate_limit and attempt < EMBEDDING_MAX_RETRIES:
                    # API限制错误，智能退避重试
                    if is_critical:
                        # 严重限制错误，更长的退避时间
                        delay = EMBEDDING_RETRY_DELAY * (3 ** attempt)
                        logger.warning(f"[流式文档处理] {worker_id} 严重API限制错误，{delay:.1f}s后重试 (第{attempt+1}次)")
                    else:
                        # 普通限制错误，指数退避
                        delay = EMBEDDING_RETRY_DELAY * (2 ** attempt)
                        logger.warning(f"[流式文档处理] {worker_id} API限制错误，{delay:.1f}s后重试 (第{attempt+1}次)")
                    
                    await asyncio.sleep(delay)
                    self.controller.record_failure(is_rate_limit=True, is_critical=is_critical)
                    continue
                else:
                    # 最终失败
                    self.controller.record_failure(is_rate_limit=is_rate_limit, is_critical=is_critical)
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    
                    if is_rate_limit:
                        logger.error(f"[流式文档处理] {worker_id} API限制错误重试耗尽: {error_msg}")
                    else:
                        logger.error(f"[流式文档处理] {worker_id} 文档处理失败: {error_msg}")
                    
                    return ("failed", "", error_msg)
    
    async def _collect_results(self) -> List[Tuple[str, str, str]]:
        """收集所有处理结果"""
        results = []
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break
        return results


class AstrBotEmbeddingProviderWrapper(EmbeddingProvider):
    """AstrBot Embedding Provider 包装类"""

    def __init__(
        self,
        embedding_util: EmbeddingSolutionHelper,
        collection_name: str,
    ):
        self.embedding_util = embedding_util
        self.collection_name = collection_name

    async def get_embedding(self, text: str) -> List[float]:
        vec = await self.embedding_util.get_embedding_async(text, self.collection_name)
        if not vec:
            raise ValueError(
                "获取向量失败，返回的向量为空或无效。请检查输入文本和配置。"
            )
        return vec

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的嵌入"""
        vecs = await self.embedding_util.get_embeddings_async(
            texts, self.collection_name
        )
        if not vecs:
            raise ValueError(
                "获取向量失败，返回的向量为空或无效。请检查输入文本和配置。"
            )
        return vecs

    def get_dim(self) -> int:
        return self.embedding_util.get_dimensions(self.collection_name)


class FaissStore(VectorDBBase):
    """
    对 AstrBot FaissVecDB 的包装类，以适应 KB 的接口规范
    使用 LRU Cache 按需加载和管理知识库集合
    """

    def __init__(
        self,
        embedding_util: EmbeddingSolutionHelper,
        data_path: str,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
        max_memory_limit_mb: float = DEFAULT_MAX_MEMORY_MB,
    ):
        super().__init__(embedding_util, data_path)
        # self.vecdbs: Dict[str, FaissVecDB] = {} # 被 cache 替代

        # ---- LRU Cache 相关 ----
        # LRU 缓存，存储 collection_name -> FaissVecDB 实例
        self.cache: LRUCache[str, FaissVecDB] = LRUCache(maxsize=max_cache_size)
        # 记录磁盘上所有已知的新格式集合名称（无论是否加载）
        self._all_known_collections: Set[str] = set()
        # 加载锁，防止同一集合并发加载
        self._locks: Dict[str, asyncio.Lock] = {}
        self.max_cache_size = max_cache_size
        # 内存管理配置
        self.max_memory_limit_mb = max_memory_limit_mb
        self.memory_batch_size = DEFAULT_MEMORY_BATCH_SIZE
        logger.info(
            f"[知识库-缓存] FaissStore LRU缓存初始化完成: 最大缓存大小={max_cache_size}, 内存限制={max_memory_limit_mb}MB"
        )
        # ------------------------

        self._old_faiss_store: Optional[OldFaissStore] = None
        self._old_collections: Dict[str, str] = {}  # 记录所有旧格式的集合
        self.embedding_utils: Dict[str, AstrBotEmbeddingProviderWrapper] = {}
        os.makedirs(self.data_path, exist_ok=True)

    async def initialize(self):
        """初始化：仅扫描磁盘，不加载任何集合到内存"""
        logger.info(f"[知识库-初始化] 开始扫描Faiss存储路径: {self.data_path}")
        # 初始化时只扫描，不加载
        await self._scan_collections_on_disk()
        logger.info(
            f"[知识库-初始化] 扫描完成 - 新格式集合: {len(self._all_known_collections)}个 {list(self._all_known_collections)}, "
            f"旧格式集合: {len(self._old_collections)}个 {list(self._old_collections.keys())}"
        )

    def _get_collection_meta(self, collection_name: str) -> Tuple[str, str, str, str]:
        """工具函数：根据集合名获取真实名称, file_id 和路径"""
        true_coll_name = (
            self.embedding_util.user_prefs_handler.get_collection_name_by_file_id(
                collection_name
            )
        )
        # 检查元数据
        collection_md = (
            self.embedding_util.user_prefs_handler.user_collection_preferences.get(
                "collection_metadata", {}
            ).get(collection_name, {})
        )

        if true_coll_name:
            # collection_name is actually a file_id
            file_id = collection_name
            final_collection_name = true_coll_name
        elif collection_md:
            # collection_name is a true name, get file_id from metadata
            file_id = collection_md.get("file_id", collection_name)
            final_collection_name = collection_name
        else:
            # fallback
            file_id = collection_name
            final_collection_name = collection_name

        index_path = os.path.join(self.data_path, f"{file_id}.index")
        storage_path = os.path.join(self.data_path, f"{file_id}.db")
        _old_storage_path = os.path.join(self.data_path, f"{file_id}.docs")
        return (
            final_collection_name,
            file_id,
            index_path,
            storage_path,
            _old_storage_path,
        )

    async def _scan_collections_on_disk(self):
        """扫描磁盘目录，识别新旧集合，填充 _all_known_collections 和 _old_collections"""
        self._all_known_collections.clear()
        self._old_collections.clear()
        if not os.path.exists(self.data_path):
            return

        scanned_file_ids = set()
        # 优先处理 .index 和 .db 文件
        all_files = os.listdir(self.data_path)
        relevant_extensions = (".index", ".db", ".docs")

        for filename in all_files:
            if not filename.endswith(relevant_extensions):
                continue

            base, ext = os.path.splitext(filename)
            if base in scanned_file_ids:
                continue

            file_id = base
            collection_name, _, index_path, storage_path, _old_storage_path = (
                self._get_collection_meta(file_id)
            )

            is_old = False
            # 检查是否为旧格式
            if _check_pickle_file(storage_path) or os.path.exists(_old_storage_path):
                is_old = True
            # 如果 .index 和 .db 都存在，认为是新格式 (除非 .db 是pickle 或存在 .docs)
            elif os.path.exists(index_path) and os.path.exists(storage_path):
                is_old = False
            # 如果只有 .docs，认为是旧格式
            elif ext == ".docs" and not os.path.exists(index_path):
                is_old = True
            else:
                # 其他情况，例如只有 .index 或只有 .db (非pickle)，暂时跳过或认为是新格式不完整
                # 为简单起见，如果存在 index 和 db 之一且非旧格式，就认为是新格式
                if ext in (".index", ".db"):
                    is_old = False
                else:
                    continue  # 忽略不明确的文件

            scanned_file_ids.add(file_id)
            if is_old:
                self._old_collections[collection_name] = collection_name
                logger.debug(f"[知识库-扫描] 发现旧格式集合: {collection_name} (file_id: {file_id})")
            else:
                self._all_known_collections.add(collection_name)
                logger.debug(f"[知识库-扫描] 发现新格式集合: {collection_name} (file_id: {file_id})")

        # 如果发现了旧集合，初始化旧存储实例
        if self._old_collections and not self._old_faiss_store:
            logger.info(f"[知识库-扫描] 检测到旧格式集合，初始化OldFaissStore处理器...")
            self._old_faiss_store = OldFaissStore(self.embedding_util, self.data_path)
            await self._old_faiss_store.initialize()

    async def _perform_load(
        self, collection_name: str, index_path: str, storage_path: str
    ) -> FaissVecDB:
        """执行实际的加载/创建 FaissVecDB 逻辑，不涉及缓存和锁"""
        logger.info(f"[知识库-加载] 开始加载/创建Faiss集合实例: '{collection_name}'")
        self.embedding_utils[collection_name] = AstrBotEmbeddingProviderWrapper(
            embedding_util=self.embedding_util,
            collection_name=collection_name,
        )
        params = {
            "doc_store_path": storage_path,
            "index_store_path": index_path,
            "embedding_provider": self.embedding_utils[collection_name],
        }
        rerank_prov = self.embedding_util.get_rerank_provider(collection_name)
        if rerank_prov:
            params["rerank_provider"] = rerank_prov
        vecdb = FaissVecDB(**params)
        await vecdb.initialize()
        logger.info(f"[知识库-加载] Faiss集合实例 '{collection_name}' 加载/创建完成")
        return vecdb

    async def _evict_lru_if_needed(self):
        """如果缓存已满，则移除并关闭最少使用的集合"""
        evicted_count = 0
        while len(self.cache) >= self.max_cache_size and self.max_cache_size > 0:
            try:
                lru_key, lru_vecdb = self.cache.popitem()
                logger.info(
                    f"[知识库-缓存] 缓存已满(max={self.max_cache_size})，移出最少使用的集合: '{lru_key}'"
                )
                self.embedding_utils.pop(lru_key, None)
                self._locks.pop(lru_key, None)  # 清理锁
                try:
                    await lru_vecdb.close()
                    logger.info(f"[知识库-缓存] 成功关闭被移出的集合: '{lru_key}'")
                    evicted_count += 1
                except Exception as close_e:
                    logger.error(f"[知识库-缓存] 关闭被移出的集合 '{lru_key}' 时发生错误: {close_e}")
            except KeyError:
                # 缓存为空
                break
            except Exception as e:
                logger.error(f"[知识库-缓存] 缓存移出过程发生未知错误: {e}")
                break

        # 如果有移出操作，触发垃圾回收
        if evicted_count > 0:
            gc.collect()
            logger.debug(f"已移出 {evicted_count} 个集合，触发垃圾回收")

    async def _unload_collection(self, collection_name: str):
        """从缓存中卸载并关闭一个指定的集合"""
        vecdb_to_close = self.cache.pop(collection_name, None)
        self.embedding_utils.pop(collection_name, None)
        self._locks.pop(collection_name, None)  # 清理锁
        if vecdb_to_close:
            logger.info(f"从缓存中卸载并关闭集合: '{collection_name}'")
            try:
                await vecdb_to_close.close()
            except Exception as e:
                logger.error(f"关闭集合 '{collection_name}' 时出错: {e}")

    async def _get_or_load_vecdb(
        self, collection_name: str, for_create: bool = False
    ) -> Optional[FaissVecDB]:
        """
        核心函数：从缓存获取或按需加载集合
        1. 检查缓存
        2. 缓存未命中则加锁
        3. 锁内再次检查缓存（Double-Check Locking）
        4. 检查是否需要移出 LRU
        5. 加载集合
        6. 放入缓存
        """
        # 1. 旧集合或已在缓存中，直接返回
        if collection_name in self._old_collections:
            return None
        if collection_name in self.cache:
            # 访问即更新其在 LRU 中的位置
            return self.cache[collection_name]

        # 2. 获取或创建针对此集合的锁
        lock = self._locks.setdefault(collection_name, asyncio.Lock())

        async with lock:
            # 3. 锁内再次检查，防止在等待锁期间其他协程已加载
            if collection_name in self.cache:
                return self.cache[collection_name]

            logger.info(f"[知识库-加载] 缓存未命中，准备加载集合: '{collection_name}'")

            _, _, index_path, storage_path, _ = self._get_collection_meta(
                collection_name
            )

            # 如果不是创建操作，且文件不存在，则不加载
            if not for_create and not (
                os.path.exists(index_path) and os.path.exists(storage_path)
            ):
                logger.warning(f"[知识库-加载] 警告: 集合 '{collection_name}' 的文件不存在，无法加载。索引文件: {index_path}, 存储文件: {storage_path}")
                # self._locks.pop(collection_name, None) # 加载失败，清理锁
                return None

            # 4. 加载前检查并执行移出操作
            await self._evict_lru_if_needed()

            # 5. 执行加载
            try:
                vecdb = await self._perform_load(
                    collection_name, index_path, storage_path
                )
                # 6. 放入缓存
                self.cache[collection_name] = vecdb
                self._all_known_collections.add(collection_name)  # 确保已记录
                logger.info(
                    f"[知识库-加载] 集合 '{collection_name}' 已加载并放入缓存。当前缓存大小: {len(self.cache)}/{self.max_cache_size}"
                )
                return vecdb
            except Exception as e:
                logger.error(f"[知识库-加载] 加载知识库集合(FAISS) '{collection_name}' 时出错: {type(e).__name__} - {str(e)}")
                # 清理可能残留的状态
                self.cache.pop(collection_name, None)
                self.embedding_utils.pop(collection_name, None)
                # self._locks.pop(collection_name, None) # 加载失败，清理锁
                return None
        # 锁自动释放

    # async def _load_collection(self, collection_name: str): # 废弃
    # async def _load_all_collections(self): # 废弃，由 _scan_collections_on_disk 替代扫描功能

    async def create_collection(self, collection_name: str):
        """创建并加载一个新集合到缓存"""
        if await self.collection_exists(collection_name):
            # 如果已存在（在磁盘或旧存储中），尝试加载到缓存（如果还不在）
            logger.info(f"[知识库-创建] Faiss集合 '{collection_name}' 已存在，尝试加载到缓存")
            await self._get_or_load_vecdb(collection_name)
            return

        logger.info(f"[知识库-创建] 开始创建新Faiss集合 '{collection_name}'")
        # 保存偏好设置
        await self.embedding_util.user_prefs_handler.save_user_preferences()

        # 使用 _get_or_load_vecdb 进行创建，它会处理锁、缓存移出和加载
        # 设置 for_create=True 使得即使文件不存在也会继续 _perform_load
        vecdb = await self._get_or_load_vecdb(collection_name, for_create=True)

        if vecdb:
            # 新创建的集合需要显式保存一下索引文件
            await vecdb.embedding_storage.save_index()
            # _get_or_load_vecdb 已经将其加入 _all_known_collections
            logger.info(f"[知识库-创建] Faiss集合 '{collection_name}' 创建成功并已加载到缓存")
        else:
            logger.error(f"[知识库-创建] Faiss集合 '{collection_name}' 创建或加载失败")

    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在于磁盘（新格式）或旧存储中"""
        # 检查已知的（扫描到的或创建的）新格式集合，以及旧格式集合
        return (
            collection_name in self._all_known_collections
            or collection_name in self._old_collections
        )

    async def _process_documents_batch(
        self,
        documents: List[Document],
        collection_name: str,
        vecdb: FaissVecDB,
    ) -> Tuple[List[str], int]:
        """
        顺序处理一批文档，优化内存使用（保留作为后备方案）
        返回: (成功添加的文档ID列表, 失败数量)
        """
        if not vecdb:
            logger.error(f"[知识库-批次处理] 致命错误: 集合 '{collection_name}' 的 vecdb 实例为空，无法处理文档")
            return [], len(documents)

        doc_ids = []
        failed_count = 0
        batch_size = len(documents)
        
        logger.debug(f"[知识库-批次处理] 开始逐个处理 {batch_size} 个文档，集合: '{collection_name}'")
        
        for i, doc in enumerate(documents):
            try:
                # 获取文档预览用于日志
                doc_preview = doc.text_content[:50].replace("\n", " ") if doc.text_content else "[空文档]"
                
                doc_id = await vecdb.insert(
                    content=doc.text_content,
                    metadata=doc.metadata,
                )
                doc_ids.append(doc_id)
                
                # 详细的进度日志（每10个文档记录一次）
                if (i + 1) % 10 == 0 or (i + 1) == batch_size:
                    logger.debug(f"[知识库-批次处理] 进度: {i+1}/{batch_size} 个文档已处理，最新: '{doc_preview}...'")
                
                # 及时清理文档引用以释放内存
                doc.text_content = ""
                doc.metadata.clear()
                
            except Exception as e:
                failed_count += 1
                excerpt = doc.text_content[:50].replace("\n", " ") if doc.text_content else "[空文档]"
                logger.error(
                    f"[知识库-批次处理] 文档处理失败: 第{i+1}/{batch_size}个文档 '{excerpt}...' 添加到集合 '{collection_name}' 失败，"
                    f"错误类型: {type(e).__name__}，错误详情: {str(e)}"
                )
            
            # 每处理一定数量的文档后触发垃圾回收
            if (i + 1) % 20 == 0:
                gc.collect()
                logger.debug(f"[知识库-批次处理] 已处理 {i+1} 个文档，执行内存垃圾回收")
                
        # 清空整个批次的文档列表
        documents.clear()
        
        if failed_count == 0:
            logger.debug(f"[知识库-批次处理] ✅ 批次处理完成: 集合 '{collection_name}' 成功处理 {len(doc_ids)} 个文档")
        else:
            logger.warning(f"[知识库-批次处理] ⚠️ 批次处理完成: 集合 '{collection_name}' 成功 {len(doc_ids)} 个，失败 {failed_count} 个文档")
            
        return doc_ids, failed_count

    async def _process_documents_efficiently(
        self,
        documents: List[Document],
        collection_name: str,
        vecdb: FaissVecDB,
        use_parallel: bool = True
    ) -> Tuple[List[str], int]:
        """
        高效的文档批次处理 - 基于系统内存动态调节并发数
        策略：智能并发调节 + 实时内存监控 + 动态垃圾回收
        """
        if not vecdb:
            logger.error(f"[知识库-高效处理] 致命错误: 集合 '{collection_name}' 的 vecdb 实例为空")
            return [], len(documents)

        total_docs = len(documents)
        if total_docs == 0:
            return [], 0

        # 动态计算最优并发数
        optimal_concurrent, effective_memory_limit, decision_reason = _calculate_optimal_concurrency(self.max_memory_limit_mb)
        
        # 获取初始内存状态
        initial_memory = _get_memory_usage_mb()
        total_memory, available_memory, usage_percent = _get_system_memory_info()
        
        logger.info(f"[知识库-高效处理] 智能并发分析: 文档数={total_docs}, 并发任务数={optimal_concurrent}({decision_reason}), "
                   f"内存限制={effective_memory_limit:.0f}MB, 系统内存={available_memory:.0f}MB可用")

        all_doc_ids = []
        total_failed = 0

        # 根据分析结果决定处理策略
        if not use_parallel or total_docs < 30 or optimal_concurrent <= 2:
            logger.info(f"[知识库-高效处理] 使用顺序处理: 文档数={total_docs}, 并发数={optimal_concurrent}")
            return await self._process_documents_batch(documents, collection_name, vecdb)

        # 智能分块计算
        concurrent_tasks = optimal_concurrent
        chunk_size = max(3, total_docs // (concurrent_tasks * 3))  # 动态调整分块大小
        max_parallel_chunks = concurrent_tasks * 2  # 控制同时处理的块数
        
        logger.info(f"[知识库-高效处理] 启动智能并发处理: {concurrent_tasks}个并发任务, 每块{chunk_size}个文档, "
                   f"最大并行块数={max_parallel_chunks}")

        # 创建动态信号量
        semaphore = asyncio.Semaphore(concurrent_tasks)
        processed_chunks = 0
        total_chunks = (total_docs + chunk_size - 1) // chunk_size

        async def process_chunk_with_monitoring(chunk_docs: List[Document], chunk_idx: int) -> Tuple[List[str], int]:
            """带内存监控的文档块处理 - 使用主上下文日志"""
            async with semaphore:
                chunk_start_memory = _get_memory_usage_mb()
                chunk_size_actual = len(chunk_docs)
                
                # 使用主协程的logger上下文记录日志
                logger.debug(f"[知识库-高效处理] 块{chunk_idx}开始: {chunk_size_actual}个文档, 内存{chunk_start_memory:.1f}MB")
                
                chunk_doc_ids = []
                chunk_failed = 0
                
                # 收集处理过程中的关键信息，避免在循环中频繁记录日志
                failed_docs = []
                
                for i, doc in enumerate(chunk_docs):
                    try:
                        doc_id = await vecdb.insert(
                            content=doc.text_content,
                            metadata=doc.metadata,
                        )
                        chunk_doc_ids.append(doc_id)
                        
                        # 及时清理文档引用
                        doc.text_content = ""
                        doc.metadata.clear()
                        
                    except Exception as e:
                        chunk_failed += 1
                        excerpt = doc.text_content[:30].replace("\n", " ") if doc.text_content else "[空]"
                        # 收集失败信息，稍后统一记录
                        failed_docs.append((i+1, excerpt, type(e).__name__, str(e)))
                
                # 清理chunk文档列表
                chunk_docs.clear()
                
                chunk_end_memory = _get_memory_usage_mb()
                memory_delta = chunk_end_memory - chunk_start_memory
                
                # 统一记录块处理结果（在主协程上下文中）
                if chunk_failed == 0:
                    logger.debug(f"[知识库-高效处理] 块{chunk_idx}完成: 成功{len(chunk_doc_ids)}, "
                               f"内存变化{memory_delta:+.1f}MB ({chunk_start_memory:.1f}→{chunk_end_memory:.1f}MB)")
                else:
                    logger.warning(f"[知识库-高效处理] 块{chunk_idx}完成: 成功{len(chunk_doc_ids)}, 失败{chunk_failed}, "
                                 f"内存变化{memory_delta:+.1f}MB ({chunk_start_memory:.1f}→{chunk_end_memory:.1f}MB)")
                    # 记录前几个失败的文档详情
                    for i, (doc_idx, excerpt, error_type, error_msg) in enumerate(failed_docs[:3]):
                        logger.error(f"[知识库-高效处理] 块{chunk_idx}文档{doc_idx}失败: '{excerpt}' - {error_type}: {error_msg}")
                    if len(failed_docs) > 3:
                        logger.warning(f"[知识库-高效处理] 块{chunk_idx}还有{len(failed_docs)-3}个文档失败（已省略详情）")
                
                return chunk_doc_ids, chunk_failed

        # 分批处理任务，避免创建过多异步任务
        tasks = []
        for i in range(0, total_docs, chunk_size):
            chunk_end = min(i + chunk_size, total_docs)
            chunk_docs = documents[i:chunk_end]
            chunk_idx = i // chunk_size + 1
            
            task = asyncio.create_task(process_chunk_with_monitoring(chunk_docs, chunk_idx))
            tasks.append(task)
            
            # 分批执行，避免内存峰值过高
            if len(tasks) >= max_parallel_chunks or chunk_idx == total_chunks:
                logger.info(f"[知识库-高效处理] 执行批次: {len(tasks)}个并发任务 (块{chunk_idx-len(tasks)+1}-{chunk_idx})")
                
                # 执行当前批次的任务
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 在主协程上下文中处理所有结果和日志
                successful_chunks = 0
                failed_chunks = 0
                
                for j, result in enumerate(results):
                    chunk_idx_actual = (i // chunk_size) - len(tasks) + j + 2
                    
                    if isinstance(result, Exception):
                        failed_chunks += 1
                        error_type = type(result).__name__
                        error_msg = str(result)
                        logger.error(f"[知识库-高效处理] 块{chunk_idx_actual}异步任务失败: {error_type} - {error_msg}")
                        total_failed += chunk_size  # 估算失败数量
                    else:
                        successful_chunks += 1
                        doc_ids, failed = result
                        all_doc_ids.extend(doc_ids)
                        total_failed += failed
                        
                        # 如果该块有失败文档，在主协程中记录汇总信息
                        if failed > 0:
                            success_in_chunk = len(doc_ids)
                            logger.info(f"[知识库-高效处理] 块{chunk_idx_actual}汇总: 成功{success_in_chunk}, 失败{failed}")
                
                # 记录批次执行汇总
                if failed_chunks > 0:
                    logger.warning(f"[知识库-高效处理] 批次执行完成: 成功块{successful_chunks}, 异常块{failed_chunks}")
                else:
                    logger.debug(f"[知识库-高效处理] 批次执行完成: {successful_chunks}个块全部成功")
                
                processed_chunks += len(tasks)
                tasks.clear()
                
                # 批次完成后的内存管理
                current_memory = _get_memory_usage_mb()
                memory_increase = current_memory - initial_memory
                
                if memory_increase > effective_memory_limit * 0.3:  # 超过限制的30%
                    logger.info(f"[知识库-高效处理] 内存增长{memory_increase:.1f}MB，触发垃圾回收 (当前{current_memory:.1f}MB)")
                    gc.collect()
                    after_gc_memory = _get_memory_usage_mb()
                    logger.debug(f"[知识库-高效处理] GC后内存: {after_gc_memory:.1f}MB (释放{current_memory-after_gc_memory:.1f}MB)")
                
                # 进度报告
                progress_percent = (processed_chunks / total_chunks) * 100
                logger.info(f"[知识库-高效处理] 进度: {processed_chunks}/{total_chunks}块 ({progress_percent:.1f}%), "
                           f"已处理{len(all_doc_ids)}个文档, 当前内存{current_memory:.1f}MB")

        # 清理原始文档列表
        documents.clear()
        
        # 最终统计和清理
        gc.collect()
        final_memory = _get_memory_usage_mb()
        memory_delta_total = final_memory - initial_memory
        
        success_count = len(all_doc_ids)
        success_rate = (success_count / total_docs * 100) if total_docs > 0 else 0
        
        logger.info(f"[知识库-高效处理] ✅ 智能并发处理完成: 成功{success_count}/{total_docs} ({success_rate:.1f}%), "
                   f"失败{total_failed}, 并发数{concurrent_tasks}, 内存变化{memory_delta_total:+.1f}MB "
                   f"({initial_memory:.1f}→{final_memory:.1f}MB)")
        
    async def _process_documents_streaming_v2(
        self, 
        documents: List[Document], 
        collection_name: str, 
        vecdb: FaissVecDB,
        rpm_limit: int = DEFAULT_RPM_LIMIT,
        max_memory_mb: float = DEFAULT_MAX_MEMORY_MB
    ) -> Tuple[List[str], int]:
        """
        新版内存安全流式文档处理 - 真正的流式处理策略
        特性：始终保持N个请求并行执行，动态调节，快速响应，严格内存控制
        """
        if not vecdb:
            logger.error(f"[知识库-流式处理v2] 致命错误: 集合 '{collection_name}' 的 vecdb 实例为空")
            return [], len(documents)
        
        total_docs = len(documents)
        if total_docs == 0:
            return [], 0
            
        # 创建内存安全的流式处理器
        processor = StreamingDocumentProcessor(vecdb, collection_name, rpm_limit, max_memory_mb)
        
        logger.info(f"[知识库-流式处理v2] 🚀 启动内存安全流式处理: {total_docs}个文档, RPM限制={rpm_limit}, 内存限制={max_memory_mb}MB")
        
        try:
            # 执行流式处理
            successful_doc_ids, failed_count = await processor.process_documents_streaming(documents)
            
            success_count = len(successful_doc_ids)
            success_rate = (success_count / total_docs * 100) if total_docs > 0 else 0
            
            logger.info(f"[知识库-流式处理v2] ✅ 流式处理完成: 成功{success_count}, 失败{failed_count}, "
                       f"成功率{success_rate:.1f}%")
            
            return successful_doc_ids, failed_count
            
        except Exception as e:
            logger.error(f"[知识库-流式处理v2] 💥 流式处理异常: {type(e).__name__}: {e}")
            return [], total_docs

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        """
        向指定集合中添加文档，使用顺序处理优化内存使用
        """
        # 处理旧集合
        if collection_name in self._old_collections:
            if self._old_faiss_store:
                logger.info(f"[知识库-添加文档] 检测到旧格式集合 '{collection_name}'，使用旧存储引擎处理")
                return await self._old_faiss_store.add_documents(
                    collection_name, documents
                )
            else:
                logger.error(f"[知识库-添加文档] 致命错误: 旧集合 '{collection_name}' 存在但 OldFaissStore 未初始化，请检查插件配置")
                return []

        # 检查或创建集合
        if not await self.collection_exists(collection_name):
            logger.info(f"[知识库-添加文档] 目标集合 '{collection_name}' 不存在，开始自动创建新集合")
            await self.create_collection(collection_name)
        else:
            logger.info(f"[知识库-添加文档] 目标集合 '{collection_name}' 已存在，准备添加文档")

        # 获取集合实例
        vecdb = await self._get_or_load_vecdb(collection_name)
        if not vecdb:
            logger.error(f"[知识库-添加文档] 严重错误: 无法获取或加载集合 '{collection_name}' 的向量数据库实例，文档添加操作失败")
            return []

        total_documents = len(documents)
        if total_documents == 0:
            logger.warning(f"[知识库-添加文档] 警告: 传入的文档列表为空，集合 '{collection_name}' 无需处理")
            return []

        # 内存使用警告和系统状态检查
        if total_documents > MAX_DOCUMENTS_WARNING_THRESHOLD:
            logger.warning(
                f"[知识库-添加文档] 内存警告: 准备一次性处理 {total_documents} 个文档 (超过阈值 {MAX_DOCUMENTS_WARNING_THRESHOLD})，"
                f"建议分批上传以避免内存溢出。当前内存批次大小: {self.memory_batch_size}"
            )

        logger.info(f"[知识库-添加文档] 开始处理: 集合='{collection_name}', 总文档数={total_documents}")
        
        all_doc_ids = []
        total_failed = 0

        # 统一使用内存安全的流式处理器
        logger.info(f"[知识库-添加文档] 使用内存安全流式处理器: {total_documents} 个文档")
        batch_doc_ids, batch_failed = await self._process_documents_streaming_v2(
            documents, collection_name, vecdb, 
            rpm_limit=DEFAULT_RPM_LIMIT,
            max_memory_mb=self.max_memory_limit_mb
        )
        all_doc_ids.extend(batch_doc_ids)
        total_failed += batch_failed

        # 保存索引
        try:
            logger.info(f"[知识库-添加文档] 开始保存集合 '{collection_name}' 的索引文件...")
            await vecdb.embedding_storage.save_index()
            logger.info(f"[知识库-添加文档] 集合 '{collection_name}' 索引文件保存成功")
        except Exception as e:
            logger.error(f"[知识库-添加文档] 索引保存失败: 集合 '{collection_name}' 索引文件保存时发生错误，"
                        f"可能影响后续搜索功能，错误详情: {str(e)}")

        # 最终清理和统计
        documents.clear()
        gc.collect()
        
        success_count = len(all_doc_ids)
        success_rate = (success_count / total_documents * 100) if total_documents > 0 else 0
        
        if total_failed == 0:
            logger.info(f"[知识库-添加文档] ✅ 任务完成: 集合 '{collection_name}' 成功添加 {success_count}/{total_documents} 个文档 (100%)")
        else:
            logger.warning(f"[知识库-添加文档] ⚠️ 任务完成(有失败): 集合 '{collection_name}' 成功 {success_count}/{total_documents} 个文档 "
                          f"({success_rate:.1f}%), 失败 {total_failed} 个，请检查上方错误日志")
        
        return all_doc_ids

    async def search(
        self, collection_name: str, query_text: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        logger.info(f"[知识库-搜索] 开始搜索: 集合='{collection_name}', 查询文本预览='{query_text[:30]}...', top_k={top_k}")
        
        if not await self.collection_exists(collection_name):
            logger.warning(f"[知识库-搜索] 警告: Faiss集合 '{collection_name}' 不存在，搜索结果为空")
            return []

        # 首先处理旧集合
        if collection_name in self._old_collections:
            if self._old_faiss_store:
                logger.info(f"[知识库-搜索] 使用旧存储引擎处理集合 '{collection_name}' 的搜索")
                return await self._old_faiss_store.search(
                    collection_name, query_text, top_k
                )
            else:
                logger.error(
                    f"[知识库-搜索] 错误: 旧集合 '{collection_name}' 存在但 OldFaissStore 未初始化"
                )
                return []

        # 获取或加载集合实例
        logger.debug(f"[知识库-搜索] 获取集合 '{collection_name}' 的向量数据库实例")
        vecdb = await self._get_or_load_vecdb(collection_name)
        if not vecdb:
            logger.error(f"[知识库-搜索] 错误: 无法获取或加载集合 '{collection_name}' 的向量数据库实例，搜索失败")
            return []

        try:
            # 安全地检查是否使用重排序，记录调试信息
            has_rerank_attr = hasattr(vecdb, 'rerank_provider')
            if has_rerank_attr:
                rerank_provider_value = getattr(vecdb, 'rerank_provider', None)
                use_rerank = rerank_provider_value is not None
                logger.debug(f"[知识库-搜索] FaissVecDB有rerank_provider属性，值: {rerank_provider_value is not None}")
            else:
                use_rerank = False
                logger.debug(f"[知识库-搜索] FaissVecDB没有rerank_provider属性，使用普通搜索模式")
                
            if use_rerank:
                # 对于有重排序的情况，先检索更多结果再重排序
                logger.debug(f"[知识库-搜索] 使用重排序模式搜索，初始检索数量: {max(20, top_k)}")
                results = await vecdb.retrieve(
                    query=query_text,
                    k=max(20, top_k)
                )
                # 手动重排序（如果有重排序提供商的话）
                if hasattr(vecdb, 'rerank_provider') and vecdb.rerank_provider:
                    try:
                        # 提取文档文本用于重排序
                        documents = [result.data.get("text", "") for result in results if result]
                        reranked_results = await vecdb.rerank_provider.rerank(query_text, documents)
                        # 重新排序结果
                        if reranked_results:
                            reranked_indices = [item.index for item in reranked_results[:top_k]]
                            results = [results[i] for i in reranked_indices if i < len(results)]
                        else:
                            results = results[:top_k]
                        logger.debug(f"[知识库-搜索] 重排序完成，最终结果数量: {len(results)}")
                    except Exception as rerank_e:
                        logger.warning(f"[知识库-搜索] 重排序失败，使用原始结果: {str(rerank_e)}")
                        results = results[:top_k]
                else:
                    results = results[:top_k]
            else:
                logger.debug(f"[知识库-搜索] 使用普通模式搜索，检索数量: {top_k}")
                results = await vecdb.retrieve(query=query_text, k=top_k)
                
        except Exception as e:
            logger.error(f"[知识库-搜索] 搜索异常: 在集合 '{collection_name}' 中搜索时发生错误，"
                        f"错误类型: {type(e).__name__}，错误详情: {str(e)}")
            return []

        # 处理搜索结果
        ret = []
        failed_parse_count = 0
        for i, result in enumerate(results):
            if result is not None:
                try:
                    metadata = json.loads(result.data.get("metadata", "{}"))
                except json.JSONDecodeError as json_e:
                    failed_parse_count += 1
                    metadata = {}
                    logger.warning(
                        f"[知识库-搜索] JSON解析失败: 集合 {collection_name} 文档 {result.data.get('doc_id')} 元数据解析失败，错误: {str(json_e)}"
                    )
                doc = Document(
                    id=result.data.get("doc_id"),
                    embedding=[],  # 原始代码这里就是空
                    text_content=result.data.get("text", ""),
                    metadata=metadata,
                )
                ret.append((doc, result.similarity))
                
        # 详细的搜索结果日志
        if len(ret) == 0:
            logger.warning(f"[知识库-搜索] 搜索结果为空: 集合 '{collection_name}' 中未找到与查询 '{query_text[:30]}...' 相关的内容")
        else:
            avg_similarity = sum(score for _, score in ret) / len(ret)
            logger.info(
                f"[知识库-搜索] ✓ 搜索完成: 集合='{collection_name}', 查询='{query_text[:30]}...', "
                f"返回结果数={len(ret)}, 平均相似度={avg_similarity:.3f}"
                + (f", JSON解析失败={failed_parse_count}个" if failed_parse_count > 0 else "")
            )
        return ret

    async def delete_collection(self, collection_name: str) -> bool:
        if not await self.collection_exists(collection_name):
            logger.info(f"Faiss 集合 '{collection_name}' 不存在，无需删除。")
            return False

        # 首先处理旧集合
        if collection_name in self._old_collections:
            self._old_collections.pop(collection_name, None)
            if self._old_faiss_store:
                return await self._old_faiss_store.delete_collection(collection_name)
            return False

        # 如果集合在缓存中，先卸载并关闭它
        await self._unload_collection(collection_name)
        # 从已知集合列表中移除
        self._all_known_collections.discard(collection_name)

        # 保持文件删除在线程中执行
        def _delete_sync():
            # self.vecdbs.pop(collection_name, None) # 改为 _unload_collection
            _, file_id, index_path, storage_path, _ = self._get_collection_meta(
                collection_name
            )

            try:
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(storage_path):
                    os.remove(storage_path)
                logger.info(
                    f"Faiss 集合文件 '{collection_name}' (file_id: {file_id}) 已删除。"
                )
                return True
            except Exception as e:
                logger.error(f"删除 Faiss 集合 '{collection_name}' 文件时出错: {e}")
                return False

        return await asyncio.to_thread(_delete_sync)

    async def list_collections(self) -> List[str]:
        """列出所有已知的集合（包括缓存中的、磁盘上未加载的、旧格式的）"""
        # 重新扫描可能更准确，但为了效率，依赖初始化扫描和创建/删除时的维护
        # await self._scan_collections_on_disk()
        return list(self._all_known_collections) + list(self._old_collections.keys())

    async def count_documents(self, collection_name: str) -> int:
        if not await self.collection_exists(collection_name):
            return 0
        # 首先处理旧集合
        if collection_name in self._old_collections:
            if self._old_faiss_store:
                return await self._old_faiss_store.count_documents(collection_name)
            else:
                return 0

        # 获取或加载集合实例
        vecdb = await self._get_or_load_vecdb(collection_name)
        if not vecdb:
            logger.warning(f"无法获取或加载集合 '{collection_name}' 来计数。")
            return 0
        try:
            cnt = await vecdb.count_documents()
            return cnt
        except Exception as e:
            logger.error(f"获取集合 '{collection_name}' 文档数量时出错: {e}")
            return 0

    async def close(self):
        """关闭所有缓存中的集合和旧存储"""
        logger.info(f"正在关闭所有已加载的 Faiss 集合 (缓存大小: {len(self.cache)})...")
        # 复制 key 列表，因为 _unload_collection 会修改 self.cache
        try:
            collections_to_unload = list(self.cache.keys())
            for collection_name in collections_to_unload:
                await self._unload_collection(collection_name)

            self.cache.clear()
            self.embedding_utils.clear()
            self._locks.clear()
            self._all_known_collections.clear()
            logger.info("所有缓存中的 Faiss 集合已关闭和清理。")

            if self._old_faiss_store:
                logger.info("正在关闭 OldFaissStore...")
                await self._old_faiss_store.close()
                self._old_faiss_store = None
                self._old_collections.clear()
                logger.info("OldFaissStore 已关闭。")

            # 强制垃圾回收
            gc.collect()
            logger.debug("已触发垃圾回收释放内存")

        except Exception as e:
            logger.error(f"关闭 Faiss 集合时发生错误: {e}")
        logger.info("FaissStore 关闭完成。")
