import streamlit as st
import os
from src.rag_pipeline import RAGPipeline

# 1. 頁面設定
st.set_page_config(page_title="AI Resume RAG System", page_icon="📄")
st.title("📄 AI Resume RAG System")

# 2. 側邊欄：檔案處理
st.sidebar.header("設定")
uploaded_file = st.sidebar.file_uploader("上傳 Resume PDF", type=["pdf"])

if uploaded_file:
    # 確保資料夾存在
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", uploaded_file.name)
    
    # 儲存檔案並放入 session_state，解決 NameError
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.file_path = file_path
    st.sidebar.info(f"save files: {uploaded_file.name}")

# 3. 側邊欄：RAG 初始化
if st.sidebar.button("Load Data From PDF For RAG"):
    if "file_path" in st.session_state:
        with st.spinner("create DB..."):
            try:
                # 這裡假設你的 RAGPipeline __init__ 可以接收 pdf_path 來處理
                # 如果你的 RAGPipeline 需要先切分好 chunks，請改用之前討論的 staticmethod 方式
                st.session_state.pipeline = RAGPipeline(pdf_path=st.session_state.file_path)
                st.sidebar.success("System Ready！")
            except Exception as e:
                st.sidebar.error(f"初始化錯誤: {str(e)}")
    else:
        st.sidebar.warning("Please Upload PDF")

# 4. 聊天介面邏輯
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示聊天記錄
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 處理使用者輸入
if prompt := st.chat_input("Input about resume problem..."):
    # 顯示用戶訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回應
    with st.chat_message("assistant"):
        if "pipeline" in st.session_state:
            with st.spinner("Gemini Thinking..."):
                try:
                    response = st.session_state.pipeline.run(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"發生錯誤: {str(e)}")
        else:
            st.markdown("⚠️ Please load PDF and click「init RAG」。")