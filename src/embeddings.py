# src/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    初始化 Embedding 模型。
    目前使用 all-MiniLM-L6-v2，輕量且適合中文與英文混合任務。
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}  # 如果你有 GPU，可改為 'cuda'
    encode_kwargs = {'normalize_embeddings': False}
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )