import os
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
import config as cfg
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    return prompt


DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


class RagService(object):

    def __init__(self, user_id: str):

        if not user_id or not user_id.strip():
            raise ValueError("RagService 必须传入非空 user_id，用于数据隔离")

        self.user_id = user_id.strip()

        # 关键：把 user_id 传给检索层，Chroma 查询时强制过滤
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(
                model=cfg.embedding_model,
                dashscope_api_key=DASHSCOPE_API_KEY,
            ),
            user_id=self.user_id,
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我的参考资料为主，简洁和专业的回答用户问题，参考资料:{context}。"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}"),
            ]
        )

        self.chat_model = ChatOpenAI(
            model=cfg.chat_model_name,
            api_key=DASHSCOPE_API_KEY,
            base_url=cfg.chat_base_url,
        )

        self.chain = self.__get_chain()

    def __get_chain(self):
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
           # "".join 拼成完整回答
            return "".join(
                f"文档片段：{d.page_content}\n文档元数据：{d.metadata}\n\n" for d in docs
            )

        def format_for_retriever(value: dict) -> str:
            # 从复杂的输入结构中提取出检索用的 query
            return value["input"]

        def tmp2(value):
            return {
                "input": value["input"]["input"],
                "context": value["context"],
                "history": value["input"]["history"],
            }

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document,
            }
            | RunnableLambda(tmp2)
            | self.prompt_template
            | print_prompt
            | self.chat_model
            | StrOutputParser()
        )

        # session_id 由 config.build_session_config(user_id) 注入，
        # 这里把 user_id 当 session_id 用，于是历史文件落在 ./chat_history/{user_id}
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain