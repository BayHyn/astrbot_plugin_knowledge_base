# 知识库存储底层接口使用文档

## 概述

知识库插件现在提供了一个底层存储接口，允许其他AstrBot插件或组件直接使用知识库的存储功能，实现增删改查、分块控制、关键词提取等高级功能。

## 主要特性

- ✅ **完整的CRUD操作**: 支持增加、查询、更新、删除文档
- ✅ **灵活的分块控制**: 可控制是否分块、分块大小、重叠度
- ✅ **关键词提取**: 支持自动提取文本关键词
- ✅ **元数据管理**: 支持自定义元数据和过滤查询
- ✅ **批量操作**: 支持批量添加和删除文档
- ✅ **统计信息**: 提供知识库统计和状态信息

## 快速开始

### 1. 获取存储API服务

```python
from astrbot.api.star import Context

async def get_knowledge_storage_api(context: Context):
    """获取知识库存储API服务"""
    from astrbot_plugin_knowledge_base.services.storage_api_service import KnowledgeBaseStorageAPI
    
    # 方法1: 通过统一接口获取
    storage_api = await KnowledgeBaseStorageAPI.get_storage_api(context)
    return storage_api
    
    # 方法2: 直接从插件实例获取
    plugin_instance = KnowledgeBaseStorageAPI.get_plugin_instance(context)
    if plugin_instance:
        return await plugin_instance.get_storage_api()
    return None
```

### 2. 基本使用示例

```python
async def example_usage(context: Context):
    # 获取存储API
    storage_api = await get_knowledge_storage_api(context)
    if not storage_api:
        print("无法获取知识库存储API")
        return
    
    # 添加文本到知识库
    from astrbot_plugin_knowledge_base.services.storage_api_service import StorageOptions
    
    options = StorageOptions(
        enable_chunking=True,      # 启用分块
        chunk_size=800,           # 分块大小
        chunk_overlap=100,        # 重叠大小
        extract_keywords=True,    # 提取关键词
        auto_create_collection=True,  # 自动创建知识库
        custom_metadata={"category": "技术文档", "priority": "high"}
    )
    
    result = await storage_api.add_text(
        collection_name="test_kb",
        text_content="这是一段测试文本，用于演示知识库存储功能。",
        options=options,
        source_info={"source": "api_demo", "user": "admin"}
    )
    
    print(f"添加结果: {result.message}")
    if result.success:
        print(f"文档IDs: {result.doc_ids}")
```

## API参考

### StorageOptions - 存储选项

```python
@dataclass
class StorageOptions:
    enable_chunking: bool = True          # 是否启用文本分块
    chunk_size: int = 1000               # 分块大小(字符数)
    chunk_overlap: int = 200             # 分块重叠(字符数)
    extract_keywords: bool = False        # 是否提取关键词
    auto_create_collection: bool = True   # 是否自动创建知识库
    custom_metadata: Optional[Dict[str, Any]] = None  # 自定义元数据
```

### QueryOptions - 查询选项

```python
@dataclass
class QueryOptions:
    top_k: int = 5                       # 返回结果数量
    similarity_threshold: float = 0.0    # 相似度阈值(0.0-1.0)
    enable_rerank: bool = False          # 是否启用重排序
    filters: Optional[Filter] = None     # 元数据过滤器
    include_metadata: bool = True        # 是否包含元数据
```

### 核心方法

#### 1. 添加文本

```python
async def add_text(
    collection_name: str,           # 知识库名称
    text_content: str,             # 文本内容
    options: Optional[StorageOptions] = None,  # 存储选项
    source_info: Optional[Dict[str, Any]] = None,  # 来源信息
) -> StorageResult
```

**示例:**
```python
result = await storage_api.add_text(
    collection_name="my_knowledge_base",
    text_content="人工智能是计算机科学的一个分支...",
    options=StorageOptions(
        enable_chunking=True,
        chunk_size=500,
        extract_keywords=True
    ),
    source_info={"user": "admin", "document": "AI基础教程.pdf"}
)
```

#### 2. 搜索知识

```python
async def search_knowledge(
    collection_name: str,           # 知识库名称
    query_text: str,               # 查询文本
    options: Optional[QueryOptions] = None,  # 查询选项
) -> QueryResult
```

**示例:**
```python
# 基本搜索
result = await storage_api.search_knowledge(
    collection_name="my_knowledge_base",
    query_text="什么是机器学习？",
    options=QueryOptions(top_k=3, similarity_threshold=0.7)
)

# 带过滤器的搜索
from astrbot_plugin_knowledge_base.vector_store.base import Filter, FilterCondition

filter_condition = Filter(
    conditions=[
        FilterCondition(key="custom_fields.category", operator="=", value="技术文档")
    ]
)

result = await storage_api.search_knowledge(
    collection_name="my_knowledge_base",
    query_text="Python编程",
    options=QueryOptions(top_k=5, filters=filter_condition)
)
```

#### 3. 删除文档

```python
async def delete_documents(
    collection_name: str,           # 知识库名称
    doc_ids: List[str],            # 文档ID列表
) -> StorageResult
```

**示例:**
```python
result = await storage_api.delete_documents(
    collection_name="my_knowledge_base",
    doc_ids=["doc_id_1", "doc_id_2", "doc_id_3"]
)
```

#### 4. 更新文档

```python
async def update_document(
    collection_name: str,           # 知识库名称
    doc_id: str,                   # 文档ID
    new_content: Optional[str] = None,  # 新内容
    new_metadata: Optional[Dict[str, Any]] = None,  # 新元数据
    options: Optional[StorageOptions] = None,  # 存储选项
) -> StorageResult
```

**示例:**
```python
result = await storage_api.update_document(
    collection_name="my_knowledge_base",
    doc_id="doc_123",
    new_content="更新后的文档内容...",
    new_metadata={"status": "updated", "version": "2.0"}
)
```

#### 5. 知识库管理

```python
# 列出所有知识库
collections = await storage_api.list_collections()

# 创建知识库
result = await storage_api.create_collection("new_knowledge_base")

# 删除知识库
result = await storage_api.delete_collection("old_knowledge_base")

# 获取统计信息
stats = await storage_api.get_collection_stats("my_knowledge_base")
```

## 高级用法

### 1. 批量处理

```python
async def batch_add_documents(storage_api, collection_name, documents_data):
    """批量添加多个文档"""
    results = []
    
    for doc_data in documents_data:
        result = await storage_api.add_text(
            collection_name=collection_name,
            text_content=doc_data["content"],
            options=StorageOptions(
                enable_chunking=doc_data.get("enable_chunking", True),
                chunk_size=doc_data.get("chunk_size", 1000),
                extract_keywords=doc_data.get("extract_keywords", False)
            ),
            source_info=doc_data.get("source_info", {})
        )
        results.append(result)
    
    return results
```

### 2. 智能分块策略

```python
async def smart_chunking(storage_api, collection_name, content, content_type="general"):
    """根据内容类型使用不同的分块策略"""
    
    if content_type == "code":
        options = StorageOptions(
            enable_chunking=True,
            chunk_size=2000,  # 代码块可以更大
            chunk_overlap=100,
            extract_keywords=False
        )
    elif content_type == "dialogue":
        options = StorageOptions(
            enable_chunking=True,
            chunk_size=500,   # 对话需要更小的块
            chunk_overlap=50,
            extract_keywords=True
        )
    else:  # 通用文本
        options = StorageOptions(
            enable_chunking=True,
            chunk_size=1000,
            chunk_overlap=200,
            extract_keywords=True
        )
    
    return await storage_api.add_text(
        collection_name=collection_name,
        text_content=content,
        options=options,
        source_info={"content_type": content_type}
    )
```

### 3. 高级搜索和过滤

```python
async def advanced_search(storage_api, collection_name, query, filters=None):
    """高级搜索功能"""
    
    # 构建复合过滤器
    if filters:
        filter_conditions = []
        for filter_item in filters:
            condition = FilterCondition(
                key=filter_item["key"],
                operator=filter_item["operator"],
                value=filter_item["value"]
            )
            filter_conditions.append(condition)
        
        search_filter = Filter(conditions=filter_conditions, logic="and")
    else:
        search_filter = None
    
    # 执行搜索
    result = await storage_api.search_knowledge(
        collection_name=collection_name,
        query_text=query,
        options=QueryOptions(
            top_k=10,
            similarity_threshold=0.6,
            enable_rerank=True,
            filters=search_filter,
            include_metadata=True
        )
    )
    
    # 处理搜索结果
    if result.success and result.results:
        processed_results = []
        for search_result in result.results:
            processed_results.append({
                "content": search_result.document.text_content,
                "score": search_result.score,
                "rerank_score": search_result.rerank_score,
                "metadata": search_result.document.metadata.custom_fields,
                "source": search_result.document.metadata.source
            })
        return processed_results
    
    return []
```

## 错误处理

所有API方法都返回结构化的结果对象，包含成功状态和错误信息：

```python
result = await storage_api.add_text(...)

if result.success:
    print(f"操作成功: {result.message}")
    if result.doc_ids:
        print(f"文档IDs: {result.doc_ids}")
else:
    print(f"操作失败: {result.message}")
    if result.error:
        print(f"错误详情: {result.error}")
```

## 注意事项

1. **插件依赖**: 确保知识库插件已正确安装和激活
2. **异步操作**: 所有API方法都是异步的，需要使用`await`
3. **资源管理**: 大量操作时注意内存使用
4. **错误处理**: 始终检查返回结果的`success`字段
5. **并发限制**: 避免同时进行大量并发操作

## 完整示例

```python
async def complete_example(context: Context):
    """完整的使用示例"""
    
    # 获取存储API
    storage_api = await get_knowledge_storage_api(context)
    if not storage_api:
        return
    
    collection_name = "demo_knowledge_base"
    
    # 1. 创建知识库
    result = await storage_api.create_collection(collection_name)
    print(f"创建知识库: {result.message}")
    
    # 2. 添加文档
    documents = [
        "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个子集，它使用统计技术使计算机能够从数据中学习。",
        "深度学习是机器学习的一个子集，使用神经网络来模拟人脑的学习过程。"
    ]
    
    doc_ids = []
    for i, content in enumerate(documents):
        result = await storage_api.add_text(
            collection_name=collection_name,
            text_content=content,
            options=StorageOptions(
                enable_chunking=False,  # 短文本不需要分块
                extract_keywords=True
            ),
            source_info={"document_id": f"doc_{i+1}", "user": "demo"}
        )
        if result.success:
            doc_ids.extend(result.doc_ids)
    
    print(f"添加了 {len(doc_ids)} 个文档")
    
    # 3. 搜索测试
    query_result = await storage_api.search_knowledge(
        collection_name=collection_name,
        query_text="什么是深度学习？",
        options=QueryOptions(top_k=2, similarity_threshold=0.5)
    )
    
    if query_result.success:
        print(f"搜索结果 ({query_result.total_count} 个):")
        for result in query_result.results:
            print(f"- 相似度: {result.score:.3f}")
            print(f"  内容: {result.document.text_content[:100]}...")
    
    # 4. 获取统计信息
    stats = await storage_api.get_collection_stats(collection_name)
    print(f"知识库统计: {stats}")
    
    # 5. 清理 (可选)
    # await storage_api.delete_collection(collection_name)
```

这个底层接口为AstrBot生态系统提供了强大的知识库存储能力，可以轻松集成到其他插件或应用中。