import time
import streamlit as st
from knowledgebase import KnowledgeBaseService
import io
import pypdf
import docx

st.title("论文润色AI")

# 1. 在侧边栏获取用户 ID
user_id = st.sidebar.text_input("请输入您的用户标识 (User ID)", value="user_001")

uploader_file = st.file_uploader(
    "请上传论文文件",
    type=['txt', 'pdf', 'docx'],  # 注意：word 应该是 docx，建议加上 txt 方便测试
    accept_multiple_files=False,
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    # 如果用户没填 ID，就不让上传
    if not user_id.strip():
        st.error("请先在左侧输入用户ID！")
        st.stop()

    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size / 1024  # KB

    st.subheader(f"文件名：{uploader_file.name}")
    st.write(f"格式：{uploader_file.type} | 大小 {file_size:.2f} KB")

    # 2. 根据文件类型提取文本
    text = ""
    try:
        if file_name.endswith('.txt'):
            text = uploader_file.getvalue().decode("utf-8")

        elif file_name.endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(io.BytesIO(uploader_file.getvalue()))
            for page in pdf_reader.pages:
                # 提取文本并处理 None 的情况
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        elif file_name.endswith('.docx'):
            doc = docx.Document(io.BytesIO(uploader_file.getvalue()))
            for para in doc.paragraphs:
                text += para.text + "\n"

        else:
            st.error("暂不支持该文件格式解析。")
            st.stop()

        if not text.strip():
            st.warning("未能从该文件中提取到有效文字，可能是扫描版 PDF。")
            st.stop()

    except Exception as e:
        st.error(f"文件解析失败：{str(e)}")
        st.stop()

    # 3. 存入知识库
    with st.spinner("存入向量库中。。。"):
        result = st.session_state["service"].upload_by_str(text, file_name, user_id)
        st.write(result)