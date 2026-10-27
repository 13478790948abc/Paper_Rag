"""
向量存储与检索服务（支持多用户隔离）
这个类本身主要只负责“查询/检索向量”，不负责“存储/写入
"""

from langchain_chroma import Chroma
import config as cfg

class VectorStoreService(object):
    def __init__(self, embedding, user_id: str = None):
        """
        初始化向量库服务
        :param embedding: 嵌入模型（如 DashScopeEmbeddings）
        :param user_id: 当前用户的唯一标识（从鉴权系统或 JWT 中获取的可信ID）
        """
        self.embedding = embedding
        self.user_id = user_id

        # 连接 Chroma 持久化数据库
        self.vector_store = Chroma(
            collection_name=cfg.collection_name,
            embedding_function=self.embedding,
            persist_directory=cfg.persist_directory,
        )

    def get_retriever(self):
        """
        返回向量检索器，自动附带当前用户的过滤条件，防止越权访问。
        """
        # 检索返回的文档数量
        search_kwargs = {"k": cfg.similarity_threshold}

        # 核心安全防线：如果传入了 user_id，强制加上 metadata 过滤
        if self.user_id:
            # "filter" 这个键名是固定的，必须严格写成 "filter"
            # 换成其他名字（比如 "where"、"filters"）都无法触发 Chroma 数据库的元数据过滤功能
            search_kwargs["filter"] = {"user_id": self.user_id}

        print(search_kwargs)

        # 基于当前已经连接好的 Chroma 向量库，给我生成一个‘检索员’（Retriever）
        # 并且把检索的规矩（即 search_kwargs）提前交给他。”
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)