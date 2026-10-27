# 已处理文档的 MD5 记录文件，用于去重
md5_path = "./md5.text"
# Chroma 集合名（类似数据库中的表名）
collection_name = "rag"
# 向量库的本地持久化目录（整个数据库的落盘位置）
persist_directory = "./chroma_db"

max_spliter_char_number = 1000            # 文本分割的阈值

embedding_model = "text-embedding-v4"
chat_model_name = "qwen3.8-max"

# ChatOpenAI 走 DashScope 的 OpenAI 兼容模式端点，qwen3.8 系列模型必须用这个端点
chat_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

similarity_threshold = 2

# spliter
chunk_size = 1000
chunk_overlap = 50
separators = [".", "。", "!", "！", "", " ", "?", "？"]


def build_session_config(user_id: str, session_id: str = None) -> dict:
    """
    根据当前用户构造 RunnableWithMessageHistory 需要的 config。
    - session_id 默认用 user_id，保证每个用户一条独立会话；
    - 若将来想支持“一个用户多个会话”，显式传 session_id 即可。
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id 不能为空")
    return {
        "configurable": {
            "session_id": session_id or user_id.strip(),
        }
    }