import os
from pathlib import Path
from dotenv import load_dotenv

# 1. 自動載入 .env 環境變數 (用於存放 API Key 等敏感資訊)
load_dotenv()

# 2. 基礎路徑設定 (確保無論在哪執行，路徑都不會亂掉)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
VECTOR_DB_DIR = OUTPUT_DIR / "chroma_db"

# 確保目錄存在
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 3. AI 模型參數 (AI 核心大腦)
# 嵌入模型：用於將文字轉為向量
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

GEMINI_MODEL = "models/gemini-3.1-flash-image-preview"

# 確保這裡確實設定為 "gemini"
LLM_PROVIDER = "gemini" 

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "flashrank/ms-marco-MiniLM-L-12-v2"

# 重排序模型：用於精確篩選檢索結果
RERANK_MODEL_NAME = "flashrank/ms-marco-MiniLM-L-12-v2"


# 生成模型 (LLM)：用於回答問題
OLLAMA_MODEL = "llama3"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"

# 4. RAG 參數 (可視資料特性調整)
# TOP_K: 檢索時回傳的片段數量
TOP_K = 5
# CHUNK_SIZE: 每個片段的大小 (字元數)
CHUNK_SIZE = 512
# CHUNK_OVERLAP: 片段重疊大小，確保上下文銜接
CHUNK_OVERLAP = 64 

# 5. 其他擴充參數
# 設定預設使用的 LLM (例如: "ollama" 或 "anthropic")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")