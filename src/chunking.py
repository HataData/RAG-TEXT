import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.vector_store import save_to_vector_store
from src.document_loader import DocumentLoader

# --- 1. 先定義類別 (確保 Python 讀得到) ---
class ChunkingStrategy:
    def __init__(self):
        # 針對簡歷結構，保持語句完整性
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def create_contextual_chunks(self, docs: List[Document]) -> List[Document]:
        """
        將文件切分並強化 Metadata。
        """
        chunks = self.splitter.split_documents(docs)
        
        # 強化 Metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
            
        return chunks

# --- 2. 最後再執行程式入口 ---
if __name__ == "__main__":
    try:
        # 1. 載入文件
        print("正在載入文件...")
        raw_docs = DocumentLoader.load_resume_pdf("sample_resume.pdf")
        
        # 2. 執行切分
        print("正在進行切分...")
        chunker = ChunkingStrategy() # 現在 Python 認識它了
        final_chunks = chunker.create_contextual_chunks(raw_docs)
        
        print(f"原始文件頁數: {len(raw_docs)}")
        print(f"切分後 Chunk 數量: {len(final_chunks)}")
        
        # 3. 儲存向量
        if final_chunks:
            print("正在儲存至向量資料庫...")
            save_to_vector_store(final_chunks)
            print("所有 chunks 已成功寫入向量資料庫！")
            
    except Exception as e:
        print(f"執行失敗: {e}")