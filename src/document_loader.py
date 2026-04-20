import fitz  # PyMuPDF
from typing import List
from langchain_core.documents import Document
from src.config import DATA_DIR

class DocumentLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_resume_pdf(file_name: str) -> List[Document]:
        """讀取 PDF 簡歷，並保留頁面資訊作為 metadata"""
        file_path = DATA_DIR / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"找不到檔案: {file_path}")
        
        doc = fitz.open(file_path)
        documents = []
        
        for i, page in enumerate(doc):
            text = page.get_text()
            # 將頁碼與檔案名稱寫入 metadata，這是後面「拒絕幻覺」的關鍵
            metadata = {"source": str(file_name), "page": i + 1}
            documents.append(Document(page_content=text, metadata=metadata))
            
        return documents

# 簡單測試一下是否能讀取
if __name__ == "__main__":
    loader = DocumentLoader()
    #
    try:
        docs = loader.load_resume_pdf("sample_resume.pdf")
        print(f"成功載入簡歷，共有 {len(docs)} 頁，第一頁內容預覽: {docs[0].page_content[:50]}...")
    except Exception as e:
        print(e)