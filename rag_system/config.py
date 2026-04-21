from dotenv import load_dotenv
import os
from pathlib import Path

# 定位项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载 .env 文件 (如果存在)
load_dotenv(BASE_DIR / ".env")

# --- API密钥配置 ---
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

# --- 模型 identifiers ---
# 本地模型
LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
LOCAL_LLM_MODEL = "qwen3:4b" 

# 在线API模型
ONLINE_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
ONLINE_LLM_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"

# --- API端点 ---
OLLAMA_BASE_URL = "http://localhost:11434"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# --- 文件与路径配置 ---
FAISS_INDEX_PATH_LOCAL = str(BASE_DIR / "data" / "faiss_index_local")
FAISS_INDEX_PATH_ONLINE = str(BASE_DIR / "data" / "faiss_index_online")
DOCUMENT_DIR = str(BASE_DIR / "data" / "source_documents")

# --- 文本分块配置 ---
CHUNK_SIZE = 600                    # RecursiveCharacterTextSplitter 的目标块大小
CHUNK_OVERLAP = 120                  # RecursiveCharacterTextSplitter 的重叠大小

# --- 预分块配置 (document_parser 阶段) ---
MIN_CHUNK_SIZE = 200               # 最小块大小，小于此值会合并
MAX_CHUNK_SIZE_BEFORE_SPLIT = 800 # 最大块大小，超过此值会进一步拆分
TITLE_MERGE_THRESHOLD = 100        # 标题块合并阈值

# --- 检索配置 ---
DEFAULT_RETRIEVAL_K = 10           # 默认检索数量
DEFAULT_RETRIEVAL_THRESHOLD = 0.4  # 默认相似度阈值
CHUNK_PREVIEW_LENGTH = 200         # 分块预览字符数
