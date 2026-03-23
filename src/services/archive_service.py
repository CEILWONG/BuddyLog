import json
import os
from typing import Dict, List, Any
from datetime import date
import datetime
from src.utils.file_utils import (
    finalize_diary, 
    get_today_draft_file, 
    _load_md_file, 
    _extract_conversation_from_draft
)
from src.services.memory_service import MemoryService
import dashscope
from dashscope import Generation


class ArchiveService:
    """归档服务类"""
    
    def __init__(self, model: str):
        self.model = model
        self.memory_service = MemoryService(model)
    
    def extract_structured_data(self, conversation: list) -> Dict[str, List[str]]:
        """从对话中提取结构化信息"""
        extraction_prompt = """
        请从以下对话中提取结构化信息，格式为JSON：
        {
            "events": ["事件1", "事件2"],
            "people": ["人物1", "人物2"],
            "emotions": ["情绪1", "情绪2"],
            "ideas": ["想法1", "想法2"]
        }
        
        【提取原则】
        1. 重点关注用户（user）的发言内容，这是用户自己的经历、想法和感受
        2. AI（assistant）的回应仅作参考，用于理解上下文，但不要将AI的话作为用户的事件或想法提取
        3. 从用户的发言中提炼真实发生的事件、提到的人物、表达的情绪、提出的想法
        
        对话内容：
        """
        
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        
        extraction_messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": conversation_text}
        ]
        
        extraction_response = Generation.call(
            api_key=dashscope.api_key,
            model=self.model,
            messages=extraction_messages,
            result_format="message",
            enable_search=False,
            enable_thinking=False
        )
        
        if extraction_response.status_code != 200:
            raise Exception(f"API request failed: {extraction_response.message}")
        
        structured_data = json.loads(extraction_response.output.choices[0].message.content)
        return structured_data
    
    def generate_diary_article(self, conversation: list, date_str: str = None) -> str:
        """生成日记文章"""
        if not date_str:
            date_str = datetime.date.today().isoformat()

        # 提取用户发言（忽略Buddy的回应）
        user_messages = [
            f"[{msg.get('time', '--:--:--')}] {msg['content']}"
            for msg in conversation if msg['role'] == 'user'
        ]
        user_text = "\n".join(user_messages) if user_messages else "今日无记录"

        article_prompt = f"""你是一位积极心理学家，专门负责观察用户的日志和成长。

【日期】{date_str}

【用户的今日发言】
{user_text}

请生成一篇日志总结，要求：
1. 使用第三人称（"用户今天..."），客观描述用户的一天
2. 主要基于用户的发言内容，buddy的回应仅作参考
3. 核心内容简洁，语言流畅，不遗漏内容，方便日后快速回顾当天发生了什么
4. 最后从积极心理学和成长学角度，给用户几句正面且有价值的评价总结

格式：
【今日概览】简洁描述用户当天的日记内容
【AI评价】正面评价和建议"""

        article_messages = [
            {"role": "system", "content": article_prompt}
        ]

        article_response = Generation.call(
            api_key=dashscope.api_key,
            model=self.model,
            messages=article_messages,
            result_format="message",
            enable_search=False,
            enable_thinking=False
        )

        if article_response.status_code != 200:
            raise Exception(f"API request failed: {article_response.message}")

        diary_article = article_response.output.choices[0].message.content
        return diary_article
    
    def archive(self, conversation: list, draft_date: date = None, delete_draft: bool = False) -> Dict[str, Any]:
        """
        执行日记归档（通用函数，支持手动和自动归档）
        
        Args:
            conversation: 对话记录列表
            draft_date: 指定日期（None表示今天，用于自动归档）
            delete_draft: 是否删除草稿（手动归档时设为True）
        
        Returns:
            包含 filename, structured_data, diary_article 的字典
        """
        # 确定日期字符串
        if draft_date:
            date_str = draft_date.isoformat()
        else:
            date_str = datetime.date.today().isoformat()
        
        # 注释掉结构化提取以加速归档（如需恢复，取消下面两行注释）
        # structured_data = self.extract_structured_data(conversation)
        structured_data = {"events": [], "people": [], "emotions": [], "ideas": []}
        
        diary_article = self.generate_diary_article(conversation, date_str)
        
        filename = finalize_diary(structured_data, conversation, diary_article, 
                                  draft_date=draft_date, delete_draft=delete_draft)
        
        self.memory_service.update_memory(conversation, date_str)
        
        return {
            "success": True,
            "filename": filename,
            "structured_data": structured_data,
            "diary_article": diary_article
        }
    
    def process_archive(self, conversation: list = None) -> Dict[str, Any]:
        """手动归档今日（点击"完成今日"按钮）"""
        # 优先从草稿文件读取完整对话，确保不丢失任何消息
        draft_path = get_today_draft_file()
        if draft_path:
            content = _load_md_file(os.path.basename(draft_path))
            conversation = _extract_conversation_from_draft(content)

        # 如果草稿文件不存在或解析失败，使用传入的 conversation
        if not conversation:
            return {
                "success": False,
                "error": "今天还没有聊天记录，无法完成记录"
            }

        return self.archive(conversation, delete_draft=True)
    
    def auto_archive_from_draft(self, conversation: list, draft_date: date) -> Dict[str, Any]:
        """自动归档过期草稿（指定日期，由 auto_archive_expired_drafts 调用）"""
        return self.archive(conversation, draft_date=draft_date, delete_draft=False)