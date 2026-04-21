import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
# 初始化
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("--- 正在向 Google 查詢可用的模型名稱 ---")
# 修正屬性名稱為 supported_actions
for model in client.models.list():
    if 'generateContent' in model.supported_actions:
        print(f"找到正確名稱: {model.name}")