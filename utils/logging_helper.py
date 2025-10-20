# astrbot_plugin_knowledge_base/utils/logging_helper.py
"""日志辅助工具 - 统一日志格式和级别"""
from typing import Optional, Any, Dict
from astrbot.api import logger


class LogHelper:
    """日志辅助类 - 提供统一的日志格式"""

    @staticmethod
    def format_operation(
        operation: str,
        target: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        格式化操作日志

        Args:
            operation: 操作名称 (如 "搜索知识库", "添加文档")
            target: 操作目标 (如知识库名称)
            details: 额外的详细信息

        Returns:
            str: 格式化后的日志消息

        Example:
            >>> LogHelper.format_operation("搜索知识库", "general", {"query": "test", "top_k": 5})
            "[搜索知识库] 目标: general | query='test', top_k=5"
        """
        msg_parts = [f"[{operation}]"]

        if target:
            msg_parts.append(f"目标: {target}")

        if details:
            detail_str = ", ".join(f"{k}={repr(v)}" for k, v in details.items())
            msg_parts.append(detail_str)

        return " | ".join(msg_parts)

    @staticmethod
    def log_operation_start(
        operation: str,
        target: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录操作开始

        Args:
            operation: 操作名称
            target: 操作目标
            details: 额外的详细信息
        """
        msg = LogHelper.format_operation(operation, target, details)
        logger.info(f"开始 {msg}")

    @staticmethod
    def log_operation_success(
        operation: str,
        target: Optional[str] = None,
        result: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录操作成功

        Args:
            operation: 操作名称
            target: 操作目标
            result: 操作结果描述
            details: 额外的详细信息
        """
        msg = LogHelper.format_operation(operation, target, details)
        if result:
            msg = f"{msg} => {result}"
        logger.info(f"✓ {msg}")

    @staticmethod
    def log_operation_error(
        operation: str,
        error: Exception,
        target: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
    ) -> None:
        """
        记录操作错误

        Args:
            operation: 操作名称
            error: 异常对象
            target: 操作目标
            details: 额外的详细信息
            exc_info: 是否记录异常堆栈
        """
        msg = LogHelper.format_operation(operation, target, details)
        logger.error(f"✗ {msg} | 错误: {str(error)}", exc_info=exc_info)

    @staticmethod
    def log_operation_warning(
        operation: str,
        message: str,
        target: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录操作警告

        Args:
            operation: 操作名称
            message: 警告消息
            target: 操作目标
            details: 额外的详细信息
        """
        msg = LogHelper.format_operation(operation, target, details)
        logger.warning(f"⚠ {msg} | {message}")

    @staticmethod
    def log_debug(
        context: str,
        details: Dict[str, Any],
    ) -> None:
        """
        记录调试信息

        Args:
            context: 上下文描述
            details: 详细信息字典
        """
        detail_str = ", ".join(f"{k}={repr(v)}" for k, v in details.items())
        logger.debug(f"[DEBUG] {context} | {detail_str}")


# 便捷函数别名
log_start = LogHelper.log_operation_start
log_success = LogHelper.log_operation_success
log_error = LogHelper.log_operation_error
log_warning = LogHelper.log_operation_warning
log_debug = LogHelper.log_debug
