import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.llm_client import LLMClient

# 確保載入 .env 中的 GOOGLE_API_KEY
load_dotenv()

class RAGPipeline:
    def __init__(self, pdf_path):
        """初始化 RAG 管線：讀取 PDF -> 切分 -> 向量化 -> 建立資料庫"""
        # 1. 讀取 PDF
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        # 2. 切分文件
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = splitter.split_documents(docs)

        # 3. 建立向量庫 (使用診斷出的正確模型名稱)
        # 這裡會自動讀取環境變數 GOOGLE_API_KEY
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vector_db = FAISS.from_documents(splits, embeddings)
        
        # 4. 初始化 LLM Client
        self.llm_client = LLMClient()

    def run(self, query: str):
        """接收問題，檢索相關內容並生成回應"""
        # 1. 檢索最相關的 chunks
        docs = self.vector_db.similarity_search(query, k=3)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 2. 組合 Prompt
        prompt = f"""
        你是一位專業的履歷分析助手。請根據以下提供的履歷內容，回答使用者的問題。
        若內容中沒有相關資訊，請誠實告知。

        【相關履歷內容】：
        {context_text}

        【使用者問題】：
        {query}
        """
        
        # 3. 生成回應
        return self.llm_client.generate(prompt)