# src/check_models.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# 獲取 API Key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("錯誤：找不到 API Key，請檢查 .env 檔案")
else:
    try:
        genai.configure(api_key=api_key)
        print("連線成功，正在查詢模型列表...\n")
        # 列出所有可用的模型
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"可用模型名稱: {m.name}")
    except Exception as e:
        print(f"連線失敗，請檢查 API Key 是否有效: {e}")