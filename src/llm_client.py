import os
from dotenv import load_dotenv
from google import genai  # 使用最新的 SDK
from src.config import GEMINI_MODEL, LLM_PROVIDER

load_dotenv()

class LLMClient:
    def __init__(self):
        self.provider = LLM_PROVIDER.lower()
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # 初始化新的 Client
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        try:
            # 使用最新的 API 呼叫方式
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"生成失敗: {str(e)}"