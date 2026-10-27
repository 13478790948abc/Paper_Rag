from datetime import datetime
import hashlib
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter  #字符分割器导入
from langchain_experimental.text_splitter import SemanticChunker
import config as cfg
import os

"""

"""

# 检测是否重复
def check_md5(md5_str):
    if not os.path.exists(cfg.md5_path):
        # 如果文件不存在就创建一个文件
        open(cfg.md5_path, 'w', encoding='utf-8').close()
        return False
    else:
        for line in open(cfg.md5_path, 'r', encoding='utf-8').readlines():
            line = line.strip()  # 处理字符串前后的空格和回车
            if line == md5_str:
                return True  # 已处理过

    return False

# 保存 MD5
def save_md5(md5_str):
    with open(cfg.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str+'\n')

# 将传入的字符串转换为 md5 字符串
def get_string_md5(input_str: str, encoding='utf-8'):
    # 将字符串转换为 bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)

    # 创建 MD5对象
    md5_obj = hashlib.md5()  # 得到 md5 对象
    md5_obj.update(str_bytes)  # 更新内容（传入即将要转换的字节数组）
    md5_hex = md5_obj.hexdigest()  # 得到 MD5 的十六进制字符串
    return md5_hex


class KnowledgeBaseService(object):

    def __init__(self):

        # 如果文件夹不存在，则创建，如果存在就跳过
        os.makedirs(cfg.persist_directory, exist_ok=True)

        # 创建数据库
        self.chroma = Chroma(
            collection_name=cfg.collection_name,
            embedding_function=DashScopeEmbeddings(model=cfg.embedding_model),
            persist_directory=cfg.persist_directory,
        )

        # 你原来的 RecursiveCharacterTextSplitter
        # 依赖 chunk_size、chunk_overlap、separators 这些字符级参数，语义分块器完全不需要它们。
        # self.spliter = RecursiveCharacterTextSplitter(
        #     chunk_size=cfg.chunk_size,  # 分割后的文本段最大长度
        #     chunk_overlap=cfg.chunk_overlap,  # 连续文本段之间的字符串重叠数量
        #     separators=cfg.separators,  # 自然段落划分的符号
        #     length_function=len,  # 使用 Python 自带的 len 函数做长度统计
        # )  # 文本分割器的对象

        self.spliter = SemanticChunker(
            embeddings=DashScopeEmbeddings(model=cfg.embedding_model),  # 复用已传入的嵌入模型
            breakpoint_threshold_type="percentile",  # 阈值策略，见下方说明
            breakpoint_threshold_amount=95,  # 百分位阈值
            min_chunk_size=100,  # 最小块字符数，防止碎片化
            sentence_split_regex=r"(?<=[。！？.!?])\s*"  #?????
        )

    def upload_by_str(self, data, filename, user_id):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        md5_hex = get_string_md5(f"{user_id}:{data}")

        if check_md5(md5_hex):
            return "【跳过】内容已经存在知识库"

        if len(data) > cfg.max_spliter_char_number:
            knowledge_chunks: list[str] = self.spliter.split_text(data)
        else:
            knowledge_chunks: list[str] = [data]

        # 元数据
        metadata = {
            "source": filename,
            "user_id": user_id,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 内容加载到向量库中
        self.chroma.add_texts(
            # iterable -> list
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        # 保存到 MD5里面
        save_md5(md5_hex)

        return "【成功】内容已经成功载入向量库"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    r = service.upload_by_str("周杰伦", "testfile", "user01")
    print(r)












