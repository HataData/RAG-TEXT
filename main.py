from fastapi import FastAPI, Request
import json
import os
from dotenv import load_dotenv
from main import start_rag_pipeline
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. FastAPI 設定 ---
app = FastAPI()

# 接收反思結果的 API
@app.post("/log_reflection")
async def log_reflection(request: Request):
    data = await request.json()
    # 將結果存入根目錄下的 log 檔
    with open("reflection_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
    return {"status": "success"}


# --- 2. RAG Pipeline 相關代碼 ---
# 1. load for API Key
load_dotenv()

def start_rag_pipeline(pdf_path):
    # 2. load PDF
    print(f"Reading Documents: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    # 3. (Chunking)
    # chunk_size ，overlap 是每一塊之間重複的部分，保證上下文連貫
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)
    
    print(f"切分完成！原本有 {len(docs)} 頁，現在切成了 {len(splits)} 個區塊。")
    return splits

if __name__ == "__main__":
    # 測試：請確保 data 文件夾裡真的有一份論文，並改好檔名
    sample_pdf = "data/sample_resume.pdf" 
    if os.path.exists(sample_pdf):
        chunks = start_rag_pipeline(sample_pdf)
        print(f"第一個區塊的內容摘要: {chunks[0].page_content[:100]}...")
    else:
        print(f"找不到文件：{sample_pdf}，請確認路徑是否正確。")