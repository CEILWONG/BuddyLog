import json
from typing import Dict, List, Any
from datetime import date
from src.utils.file_utils import finalize_diary, finalize_diary_for_date
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
    
    def generate_diary_article(self, structured_data: Dict[str, List[str]]) -> str:
        """生成日记文章"""
        article_prompt = f"""
        基于以下结构化信息，生成一篇第一人称叙事散文风格的日记：
        {json.dumps(structured_data, ensure_ascii=False, indent=2)}
        
        要求：
        - 第一人称视角
        - 叙事流畅自然
        - 包含情感表达
        - 体现当天的主要事件和感受
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
    
    def _do_archive(self, conversation: list, draft_date: date = None) -> Dict[str, Any]:
        """执行归档（通用）"""
        structured_data = self.extract_structured_data(conversation)
        diary_article = self.generate_diary_article(structured_data)
        
        # 根据是否有 draft_date 选择归档方式
        if draft_date:
            filename = finalize_diary_for_date(structured_data, conversation, diary_article, draft_date)
        else:
            filename = finalize_diary(structured_data, conversation, diary_article)
        
        self.memory_service.update_memory(diary_article)
        
        return {
            "success": True,
            "filename": filename,
            "structured_data": structured_data,
            "diary_article": diary_article
        }
    
    def process_archive(self, conversation: list) -> Dict[str, Any]:
        """处理归档请求（手动归档今日）"""
        return self._do_archive(conversation)
    
    def auto_archive_from_draft(self, conversation: list, draft_date: date) -> Dict[str, Any]:
        """自动归档过期草稿（指定日期）"""
        return self._do_archive(conversation, draft_date)