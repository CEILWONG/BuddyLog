from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class Message(BaseModel):
    """聊天消息模型"""
    content: str
    history: list = []  # 简单传递上下文
    mode: Optional[str] = None  # "article" 表示文章模式，None/其他为普通聊天


class ArticleDraftBody(BaseModel):
    """文章草稿保存请求模型"""
    content: str = ""


class ArchiveRequest(BaseModel):
    """归档请求模型"""
    conversation: list  # 对话历史


class ChatResponse(BaseModel):
    """聊天响应模型"""
    reply: str
    filename: Optional[str] = None
    diary_saved: bool = False


class ArchiveResponse(BaseModel):
    """归档响应模型"""
    success: bool
    filename: str
    structured_data: Dict[str, List[str]]
    diary_article: str