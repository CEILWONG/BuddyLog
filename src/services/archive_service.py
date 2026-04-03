import json
import os
from typing import Dict, List, Any
from datetime import date
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.file_utils import (
    finalize_diary, 
    get_today_draft_file, 
    _load_md_file, 
    _extract_conversation_from_draft
)
from src.services.memory_service import MemoryService
from openai import OpenAI


class ArchiveService:
    """归档服务类"""
    
    def __init__(self, model: str, openai_client: OpenAI = None, enable_thinking: bool = False):
        self.model = model
        self.openai_client = openai_client
        self.enable_thinking = enable_thinking
        self.memory_service = MemoryService(model, openai_client, enable_thinking)
    
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
        
        # 调用API (OpenAI 格式)
        # 构建 extra_body 参数（明确传递 enable_thinking 控制深度思考）
        extra_body = {"enable_thinking": self.enable_thinking}
        
        try:
            extraction_response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=extraction_messages,
                extra_body=extra_body
            )
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")
        
        structured_data = json.loads(extraction_response.choices[0].message.content)
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

        article_prompt = f"""你是一位忠实的记录者，帮助用户整理当天的日记。请基于用户的真实发言，写出一篇简洁、真实的日记。

【日期】{date_str}

【用户的今日发言】
{user_text}

【写作要求】
1. **叙事视角**：使用第三人称（"TA今天..."），像一位朋友在客观记录TA的一天
2. **内容原则**：
   - 只记录用户真实提到的事情，不要添加虚构内容
   - 不要过度解读或揣测用户的想法
   - 不要编造场景、对话或细节
   - 保持真实，像用户自己写的日记一样朴素
3. **语言风格**：
   - 简洁、自然、口语化
   - 避免华丽的辞藻和文学修饰
   - 像普通人写日记一样，平实记录当天
4. **内容组织**：
   - 按时间顺序或事件顺序记录
   - 保留用户提到的关键信息和感受
   - 不遗漏重要内容，但也不刻意扩充

【结构建议】（必须严格使用以下标题格式）

**【日记文章】**
用第三人称（"TA今天..."）记录今天发生的事情，字数不限，根据用户聊天内容来定，真实、简洁即可。

**【AI评价】**
请从积极心理学和成长学的角度回应用户今天的一天，200-300字。要求：
- 运用积极心理学的一些知识分析，找到闪光点，鼓励赞扬用户
- 语气真诚客观，又显得很专业
- 如果用户提到了烦恼或困难，先共情、再鼓励，让TA感受到被理解
- 整体是为了提供情绪价值给用户，让人读完心情变好，拥有自信的力量"""

        article_messages = [
            {"role": "system", "content": article_prompt}
        ]

        # 调用API (OpenAI 格式)
        # 构建 extra_body 参数（明确传递 enable_thinking 控制深度思考）
        extra_body = {"enable_thinking": self.enable_thinking}
        
        try:
            article_response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=article_messages,
                extra_body=extra_body
            )
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")

        diary_article = article_response.choices[0].message.content
        return diary_article
    
    def archive(self, conversation: list, draft_date: date = None, delete_draft: bool = False,
                user_email: str = None) -> Dict[str, Any]:
        """
        执行日记归档（通用函数，支持手动和自动归档）
        
        Args:
            conversation: 对话记录列表
            draft_date: 指定日期（None表示今天，用于自动归档）
            delete_draft: 是否删除草稿（手动归档时设为True）
            user_email: 用户邮箱（可选，用于用户隔离）
        
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
        
        # 生成日记文章（必须等待完成，用于保存文件）
        diary_article = self.generate_diary_article(conversation, date_str)
        
        filename = finalize_diary(structured_data, conversation, diary_article, 
                                  draft_date=draft_date, delete_draft=delete_draft,
                                  user_email=user_email)
        
        # 长期记忆更新改为后台异步执行（不阻塞前端响应）
        self._background_update_memory(conversation, date_str, user_email)
        
        return {
            "success": True,
            "filename": filename,
            "structured_data": structured_data,
            "diary_article": diary_article
        }
    
    def _background_update_memory(self, conversation: list, date_str: str, user_email: str = None):
        """后台异步更新长期记忆（不阻塞主流程）"""
        def update_task():
            try:
                self.memory_service.update_memory(conversation, date_str, user_email)
                print(f"[Background] Memory update completed for user: {user_email}, date: {date_str}")
            except Exception as e:
                print(f"[Background] Memory update failed for user {user_email}: {e}")
        
        # 使用线程池后台执行
        import threading
        thread = threading.Thread(target=update_task, name=f"memory_update_{user_email}")
        thread.daemon = True
        thread.start()
    
    def process_archive(self, conversation: list = None, user_email: str = None) -> Dict[str, Any]:
        """手动归档今日（点击"完成今日"按钮）"""
        # 优先从草稿文件读取完整对话，确保不丢失任何消息
        draft_path = get_today_draft_file(user_email)
        if draft_path and os.path.exists(draft_path):
            with open(draft_path, "r", encoding="utf-8") as f:
                content = f.read()
            conversation = _extract_conversation_from_draft(content)

        # 如果草稿文件不存在或解析失败，使用传入的 conversation
        if not conversation:
            return {
                "success": False,
                "error": "今天还没有聊天记录，无法完成记录"
            }

        return self.archive(conversation, delete_draft=True, user_email=user_email)
    
    def auto_archive_from_draft(self, conversation: list, draft_date: date, user_email: str = None) -> Dict[str, Any]:
        """自动归档过期草稿（指定日期，由 auto_archive_expired_drafts 调用）"""
        return self.archive(conversation, draft_date=draft_date, delete_draft=False, user_email=user_email)