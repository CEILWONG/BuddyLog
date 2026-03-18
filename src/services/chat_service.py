import json
import datetime
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

        recent_diaries_text = "\n\n".join(recent_diaries[:7]) if recent_diaries else '刚认识不久'

        # 获取当前日期和星期
        now = datetime.datetime.now()
        weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekday = weekdays[now.weekday()]
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')

        system_prompt = f"""【系统状态】
当前日期：{date_str} {weekday}
当前时间：{time_str}

{persona if persona else '你是 Moss,一个陪用户聊每天日常的老朋友。'}

【记住的事】
{memory if memory else '暂时还不了解太多'}

【最近聊的】
每段对话标注了日期，【今天】表示当天已发生的对话。

{recent_diaries_text}

【你的档案】
{profile_info if profile_info else '慢慢了解中'}

如果用户是问一个问题，你要认真思考，分析最近聊的内容，并结合日期精准的掌握计算时间线，再给出回答。

【输出格式 - 严格遵守】
1. 必须且只能返回 JSON 格式：{{"reply": "你的回复内容"}}
2. 不要添加 Markdown 代码块标记（如 ```json）
3. 不要添加任何前缀或后缀文字
4. 不要添加系统状态信息
5. 示例：{{"reply": "子豪大人，我在。"}}
6. 违规示例 ❌：```json{{"reply": "你好"}}``` 或 系统时间同步至...{{"reply": "你好"}}
7. 正确示例 ✅：{{"reply": "你好"}}

【绝对禁止 - 违者重罚】
- 禁止编造任何数字：包括但不限于"间隔X秒"、"频率X%"、"持续X分钟"、"下降X%"、"约X个月"等所有无法验证的精确数据
- 禁止虚假分析：如"检测到重复提问"、"符合人类行为模式"、"根据历史数据"等AI式过度解读
- 禁止学术腔：如"结论"、"综上所述"、"数据表明"、"研究显示"等论文用语
- 禁止扮演分析师：你不是在做数据分析，是在和朋友闲聊
- 如果不确定具体数据，只说定性结论（如"有一段时间了"、"比上次久"），绝不用数字"""

        return system_prompt
    
    def generate_response(self, content: str, history: list) -> Dict[str, Any]:
        """生成聊天响应"""
        # 构建系统提示
        system_prompt = self.build_system_prompt()
        
        # 构建消息数组，保留完整对话历史（让MOSS理解上下文逻辑）
        valid_history = [msg for msg in history if msg.get("content")]
        messages = [
            {"role": "system", "content": system_prompt},
            *valid_history[-10:],  # 最近10轮对话
            {"role": "user", "content": content}
        ]
        
        # 调用API
        response = Generation.call(
            api_key=dashscope.api_key,
            model=self.model,
            messages=messages,
            result_format="message",
            enable_search=False,
            enable_thinking=False
        )
        
        if response.status_code == 200:
            # 提取回复内容
            content = response.output.choices[0].message.content
            # 清理可能的 Markdown 代码块标记
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            # 解析 JSON 响应
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # 如果解析失败，将内容包装为 reply 字段
                result = {"reply": content}
            
            # 获取 token 使用量
            usage = response.usage if hasattr(response, 'usage') else None
            if usage:
                input_tokens = usage.input_tokens if hasattr(usage, 'input_tokens') else 0
                output_tokens = usage.output_tokens if hasattr(usage, 'output_tokens') else 0
                total_tokens = usage.total_tokens if hasattr(usage, 'total_tokens') else (input_tokens + output_tokens)
                result['tokens'] = {
                    'input': input_tokens,
                    'output': output_tokens,
                    'total': total_tokens
                }
            
            return result
        else:
            raise Exception(f"API request failed: {response.message}")
    
    def process_chat(self, content: str, history: list) -> Dict[str, Any]:
        """处理聊天请求"""
        # 1. 首先检查并自动归档过期草稿（如果存在且archive_service已设置）
        auto_archived = []
        if self.archive_service:
            auto_archived = auto_archive_expired_drafts(self.archive_service)
        
        # 2. 生成响应（包含token使用量）
        result = self.generate_response(content, history)
        
        reply_text = result.get("reply", "")
        
        # 3. 追加到今天的草稿文件
        filename = append_to_draft(content, reply_text)
        
        response = {
            "reply": reply_text,
            "filename": filename,
            "tokens": result.get("tokens", {
                "input": 0,
                "output": 0,
                "total": 0
            })
        }
        
        # 4. 如果有自动归档的文件，添加到响应中
        if auto_archived:
            response["auto_archived"] = auto_archived
        
        return response