import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print(f"--- 正在檢查 API Key: {api_key[:5]}**** ---")
# 列出所有模型
for m in genai.list_models():
    # 檢查是否支援 embedContent
    if 'embedContent' in m.supported_generation_methods:
        print(f"找到支援的模型名稱: {m.name}")