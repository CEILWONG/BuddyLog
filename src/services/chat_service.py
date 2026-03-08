import json
from typing import Dict, Any, Optional
from src.utils.file_utils import (
    load_memory, 
    load_recent_diaries, 
    append_to_draft,
    auto_archive_expired_drafts,
    extract_agent_persona,
    extract_profile_without_persona
)
import dashscope
from dashscope import Generation


class ChatService:
    """聊天服务类"""
    
    def __init__(self, model: str, archive_service=None):
        self.model = model
        self.archive_service = archive_service
    
    def set_archive_service(self, archive_service):
        """设置归档服务（用于自动归档过期草稿）"""
        self.archive_service = archive_service
    
    def build_system_prompt(self) -> str:
        """构建系统提示"""
        persona = extract_agent_persona()
        profile_info = extract_profile_without_persona()
        memory = load_memory()
        recent_diaries = load_recent_diaries()
        
        system_prompt = f"""{persona if persona else '你是 Buddy,一个陪用户聊每天日常的老朋友。'}

【记住的事】
{memory if memory else '暂时还不了解太多'}

【最近聊的】
{"\n---\n".join(recent_diaries[:7]) if recent_diaries else '刚认识不久'}

【用户档案】
{profile_info if profile_info else '慢慢了解中'}

输出JSON格式：{{"reply": "你的回复", "diary_entry": "日记摘要或null"}}"""
        
        return system_prompt
    
    def generate_response(self, content: str, history: list) -> Dict[str, Any]:
        """生成聊天响应"""
        # 构建系统提示
        system_prompt = self.build_system_prompt()
        
        # 构建消息数组，确保每个消息都有content字段
        messages = [
            {"role": "system", "content": system_prompt},
            *[msg for msg in history if msg.get("content")],  # 过滤掉没有content字段的消息
            {"role": "user", "content": content}
        ]
        
        # 调用API
        response = Generation.call(
            api_key=dashscope.api_key,
            model=self.model,
            messages=messages,
            result_format="message"
        )
        
        if response.status_code == 200:
            # 提取回复内容
            content = response.output.choices[0].message.content
            # 解析 JSON 响应
            result = json.loads(content)
            return result
        else:
            raise Exception(f"API request failed: {response.message}")
    
    def process_chat(self, content: str, history: list) -> Dict[str, Any]:
        """处理聊天请求"""
        # 1. 首先检查并自动归档过期草稿（如果存在且archive_service已设置）
        auto_archived = []
        if self.archive_service:
            auto_archived = auto_archive_expired_drafts(self.archive_service)
        
        # 2. 生成响应
        result = self.generate_response(content, history)
        
        reply_text = result.get("reply", "")
        diary_entry = result.get("diary_entry")
        
        # 3. 追加到今天的草稿文件
        filename = append_to_draft(diary_entry, content, reply_text)
        diary_saved = diary_entry is not None and diary_entry != ""
        
        response = {
            "reply": reply_text,
            "filename": filename,
            "diary_saved": diary_saved
        }
        
        # 4. 如果有自动归档的文件，添加到响应中
        if auto_archived:
            response["auto_archived"] = auto_archived
        
        return response