import json
from typing import Dict, Any, Optional
from src.utils.file_utils import load_profile, load_memory, load_recent_diaries, save_diary
import dashscope
from dashscope import Generation


class ChatService:
    """聊天服务类"""
    
    def __init__(self, model: str):
        self.model = model
    
    def build_system_prompt(self) -> str:
        """构建系统提示"""
        profile = load_profile()
        memory = load_memory()
        recent_diaries = load_recent_diaries()
        
        system_prompt = f"""
        {profile}

        # 长期记忆
        {memory}

        # 近期日记
        {"\n---\n".join(recent_diaries[:5])}  # 只包含最近5篇的内容

        任务：
        1. 像朋友一样回应用户的话。
        2. 从对话中提取关键信息，为用户生成一段简短的“日记摘要”。

        输出格式要求：
        请严格使用 JSON 格式返回，包含两个字段：
        {{
            "reply": "你对用户说的自然语言回复",
            "diary_entry": "如果对话中有值得记录的内容，生成一段 markdown 格式的日记摘要；如果没有，则为 null"
        }}
        """
        
        # 确保system_prompt不为空
        if not system_prompt:
            system_prompt = "你叫 Buddy，是用户的亲密好友兼私人日记助手。"
        
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
        # 生成响应
        result = self.generate_response(content, history)
        
        reply_text = result.get("reply", "")
        diary_entry = result.get("diary_entry")
        
        # 如果有日记内容，保存到本地 .md 文件
        filename = None
        if diary_entry:
            filename = save_diary(diary_entry, history, reply_text)
        
        return {
            "reply": reply_text,
            "filename": filename,
            "diary_saved": bool(filename)
        }