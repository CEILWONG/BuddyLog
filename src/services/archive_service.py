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
            result_format="message"
        )
        
        if extraction_response.status_code != 200:
            raise Exception(f"API request failed: {extraction_response.message}")
        
        structured_data = json.loads(extraction_response.output.choices[0].message.content)
        return structured_data
    
    def generate_diary_article(self, conversation: list, date_str: str = None) -> str:
        """生成日记文章"""
        if not date_str:
            date_str = datetime.date.today().isoformat()
        
        # 将对话转换为文本格式
        conversation_text = "\n".join([
            f"用户: {msg['content']}" if msg['role'] == 'user' else f"Buddy: {msg['content']}"
            for msg in conversation
        ])
        
        article_prompt = f"""
        请基于以下用户与Buddy的对话记录，生成一篇第一人称叙事散文风格的日记。
        
        【重要】这篇日记的日期是 {date_str}，请在文章开头明确标注此日期，不要编造其他日期。
        
        对话记录：
        {conversation_text}
        
        要求：
        - 第一人称视角（以用户的口吻）
        - 基于对话原文内容，提炼成流畅的日记叙述
        - 保留用户的真实表达和情感
        - 叙事自然，像真实的个人日记
        - 包含情感表达和当天的主要事件
        - 文章开头必须包含正确的日期：{date_str}
        """
        
        article_messages = [
            {"role": "system", "content": article_prompt}
        ]
        
        article_response = Generation.call(
            api_key=dashscope.api_key,
            model=self.model,
            messages=article_messages,
            result_format="message"
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
        
        structured_data = self.extract_structured_data(conversation)
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
            raise Exception("没有可归档的对话内容")
        
        return self.archive(conversation, delete_draft=True)
    
    def auto_archive_from_draft(self, conversation: list, draft_date: date) -> Dict[str, Any]:
        """自动归档过期草稿（指定日期，由 auto_archive_expired_drafts 调用）"""
        return self.archive(conversation, draft_date=draft_date, delete_draft=False)