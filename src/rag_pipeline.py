# src/rag_pipeline.py
from src.vector_store import load_vector_store
from src.llm_client import LLMClient # 假設你有這個 Client
from src.prompt_templates import RAG_PROMPT # 假設你有定義 Prompt Template

class RAGPipeline:
    def __init__(self):
        self.vector_db = load_vector_store()
        self.llm = LLMClient() # 初始化你的 Claude/Ollama client

    def run(self, query: str):
        # 1. 檢索 (Retrieval)
        print(f"正在搜尋相關資訊: {query}")
        docs = self.vector_db.similarity_search(query, k=3)
        
        # 2. 組合 Context
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. 準備 Prompt
        # 這裡假設你有一個 prompt 模板，將 context 和 query 放入
        final_prompt = RAG_PROMPT.format(context=context_text, question=query)
        
        # 4. 生成回答 (Generation)
        print("正在生成回答...")
        response = self.llm.generate(final_prompt)
        
        return response, docs

if __name__ == "__main__":
    pipeline = RAGPipeline()
    query = "What is the user's experience with Python?"
    response, source_docs = pipeline.run(query)
    
    print("\n--- 回答 ---")
    print(response)
    print("\n--- 參考來源 ---")
    for doc in source_docs:
        print(f"- {doc.metadata.get('source', 'Unknown')}")