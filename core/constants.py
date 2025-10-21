# astrbot_plugin_knowledge_base/constants.py

PLUGIN_REGISTER_NAME = "astrbot_plugin_knowledge_base"

# 定义知识库内容标记
KB_START_MARKER = "###KBDATA_START###"
KB_END_MARKER = "###KBDATA_END###"

# 用于 'prepend_prompt' 方式时，在用户原始问题前添加的标记
USER_PROMPT_DELIMITER_IN_HISTORY = "\n\n用户的原始问题是：\n"

# 文件下载相关常量
ALLOWED_FILE_EXTENSIONS = [
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".json",
    ".xml",
    ".csv",
    ".epub",
    ".jpg",
    ".jpeg",
    ".png",
]
TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".acc", ".aiff"}
MARKITDOWN_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".json",
    ".xml",
    ".csv",
    ".epub",
}
MAX_DOWNLOAD_FILE_SIZE_MB = 50
COMMON_ENCODINGS = [
    "utf-8",
    "gbk",
    "gb2312",
    "gb18030",
    "utf-16",
    "latin-1",
    "iso-8859-1",
]
READ_FILE_LIMIT = 4096  # 4KB

# 搜索和批处理相关常量
MAX_SEARCH_TOP_K = 30  # 搜索结果最大数量
MIN_SEARCH_TOP_K = 1   # 搜索结果最小数量
DEFAULT_SEARCH_TOP_K = 1  # 默认搜索结果数量
DEFAULT_EMBEDDING_BATCH_SIZE = 10  # Embedding 批处理大小

# 文本显示长度限制（用于日志和输出）
QUERY_PREVIEW_LENGTH = 30  # 查询文本预览长度
QUERY_LOG_LENGTH = 50  # 查询文本日志长度
CONTENT_PREVIEW_LENGTH = 100  # 内容预览长度（日志）
CONTENT_DISPLAY_LENGTH = 200  # 内容显示长度（用户界面）
PROMPT_PREVIEW_LENGTH = 200  # Prompt 预览长度

# 超时和重试设置
HTTP_REQUEST_TIMEOUT = 30.0  # HTTP 请求超时（秒）
PROCESSING_TIMEOUT = 10.0  # 处理队列超时（秒）
DEFAULT_CONNECT_TIMEOUT = 10.0  # 默认连接超时（秒）

# 批处理和GC设置
GC_INTERVAL_BATCHES = 10  # 每处理多少批次执行一次垃圾回收
MAX_CONCURRENT_BATCHES = 10  # 最大并发批次数

# Milvus 搜索参数
DEFAULT_MILVUS_NPROBE = 10  # Milvus 搜索时的 nprobe 参数

# 版本依赖
MIN_FAISS_VERSION = "1.10.0"  # Faiss 最低版本
MIN_MILVUS_LITE_VERSION = "2.4.10"  # Milvus Lite 最低版本
