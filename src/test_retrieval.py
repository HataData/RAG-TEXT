# src/test_retrieval.py
from src.vector_store import load_vector_store

def test_search(query):
    print(f"正在搜尋: '{query}'")
    
    # 1. 載入我們剛剛建好的資料庫
    vector_db = load_vector_store()
    
    # 2. 進行相似度搜尋 (Similarity Search)
    # k=2 代表回傳最相關的前 2 個 chunks
    results = vector_db.similarity_search(query, k=2)
    
    # 3. 顯示結果
    print("-" * 30)
    for i, doc in enumerate(results):
        print(f"結果 {i+1}:")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")
        print("-" * 30)

if __name__ == "__main__":
    # 測試一下你的履歷內容，例如搜尋與 Python 相關的關鍵字
    test_search("What is the user's experience with Python?")