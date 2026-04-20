# src/vector_store.py
from langchain_community.vectorstores import FAISS
from src.embeddings import get_embedding_model
import os

# 設定向量庫儲存路徑
VECTOR_DB_PATH = "faiss_index"

def save_to_vector_store(chunks):
    """
    接收來自 chunking.py 的 chunks，將其向量化並存入 FAISS
    """
    print("正在將 Chunks 轉換為向量...")
    embeddings = get_embedding_model()
    
    # 建立向量庫
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # 儲存到本地
    vector_db.save_local(VECTOR_DB_PATH)
    print(f"向量庫已成功儲存至: {VECTOR_DB_PATH}")

def load_vector_store():
    """
    讀取已儲存的向量庫
    """
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(f"找不到向量庫: {VECTOR_DB_PATH}")
        
    embeddings = get_embedding_model()
    return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)