import streamlit as st
from Rag import RagService
import config as config


st.title("论文润色AI")
st.divider()

# 侧边栏：用户标识（与 file_upload.py 保持一致的字段名）
user_id = st.sidebar.text_input("请输入您的用户标识 (User ID)", value="user_001").strip()

if not user_id:
    st.warning("请先在左侧输入用户 ID")
    st.stop()

# 首次运行为空，在session_state里创建rag_user_id这个变量记录状态
if st.session_state.get("rag_user_id") != user_id:
    # 为新用户创建自己的 RagService类
    st.session_state["rag"] = RagService(user_id=user_id)
    # 记录当前的用户id
    st.session_state["rag_user_id"] = user_id

# 第一次运行时建一个空字典作为总仓库，之后重跑不再覆盖，从而保证所有用户的聊天记录能跨帧保留下来
if "messages_by_user" not in st.session_state:
    st.session_state["messages_by_user"] = {}
    # st.session_state = {
    #     "messages_by_user": {},        ← 空的，还没有任何用户
    # }

#
if user_id not in st.session_state["messages_by_user"]:
    st.session_state["messages_by_user"][user_id] = [
        {"role": "assistant", "content": "你好，有什么可以帮助你的？"}
    ]
    # st.session_state = {
    #     "messages_by_user": {                    ← 第一行建的外层字典
    #         "user_001": [                        ← 第二行加进去的 key
    #             {"role": "assistant", "content": "你好，有什么可以帮助你的？"}
    #         ]
    #     }
    # }

# 从「总仓库」里，把「当前用户那个抽屉」取出来，交给变量 messages。
messages = st.session_state["messages_by_user"][user_id]

for message in messages:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    messages.append({"role": "user", "content": prompt})

    ai_res_list = []
    with st.spinner("AI思考中......"):
        session_config = config.build_session_config(user_id)

        #  res_stream是一个 generator生成器
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, session_config)

        def capture(generator, cache_list):
            """
            generator: 数据源
            cache_list: 攒碎片的桶
            """
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))
        # "".join 拼成完整回答
        messages.append({"role": "assistant", "content": "".join(ai_res_list)})