import json, os, re
from typing import Sequence
from langchain_core.messages import message_to_dict, BaseMessage, messages_from_dict
from langchain_core.chat_history import BaseChatMessageHistory

def _safe_name(name: str) -> str:
    """把用户/会话标识里的非法字符替换掉，避免路径穿越。"""
    return re.sub(r"[^\w\-.]", "_", name)

def get_history(session_id):
    return FileChatMessageHistory(session_id, "./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):

    def __init__(self, session_id, storage_path):
        self.session_id = _safe_name(session_id)
        self.storage_path = storage_path

        # file_path: 完整路径，例如 ./chat_history/user_001
        self.file_path = os.path.join(self.storage_path, self.session_id)
        # os.makedirs(..., exist_ok=True)：确保父目录存在。如果 ./chat_history 不存在，第一次写入会报 FileNotFoundError。
        # exist_ok=True 保证「已存在不报错」。
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        # 读出文件里已有的所有消息
        all_messages = list(self.messages)      # ① 读出已有历史
        all_messages.extend(messages)                        # ② 追加新消息
        # message_to_dict: BaseMessage 对象 → 可 JSON 序列化的 dict
        new_messages = [message_to_dict(m) for m in all_messages]
        # ④ 全量覆盖写
        with open(self.file_path, "w", encoding="utf-8") as f:
            # 覆盖写整个文件
            json.dump(new_messages, f, ensure_ascii=False, indent=2)

    # property: 让 history.messages 像属性一样访问（不用写 history.messages()）。这是基类要求的接口形式。
    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return messages_from_dict(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)