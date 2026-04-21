import os
from dotenv import load_dotenv
from google import genai
from src.config import GEMINI_MODEL, LLM_PROVIDER
from tenacity import retry, wait_exponential, stop_after_attempt

# 載入環境變數
load_dotenv()

class LLMClient:
    def __init__(self):
        self.provider = LLM_PROVIDER.lower()
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # 初始化新的 Client
        # 請確保 Google API Key 已經存在於你的 .env 檔案中
        self.client = genai.Client(api_key=api_key)

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10), 
        stop=stop_after_attempt(3),
        reraise=True  # 如果重試 3 次都失敗，會拋出錯誤讓程式知道
    )
    def generate(self, prompt: str) -> str:
        """
        使用 Google 新版 SDK 生成內容，並內建自動重試機制
        """
        # 使用最新的 Client 呼叫方式
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text