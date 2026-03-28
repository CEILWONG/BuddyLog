import os
import json
import datetime
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from src.utils.file_utils import (
    load_memory, 
    load_recent_diaries, 
    append_to_draft,
    auto_archive_expired_drafts,
    extract_agent_persona,
    extract_profile_without_persona
)
import dashscope
from dashscope import Generation, MultiModalConversation


class ChatService:
    """聊天服务类"""
    
    def __init__(self, model: str, archive_service=None):
        self.model = model
        self.archive_service = archive_service
        # 后台归档线程池（单线程，确保顺序执行）
        self._archive_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="archive_worker")
    
    def set_archive_service(self, archive_service):
        """设置归档服务（用于自动归档过期草稿）"""
        self.archive_service = archive_service
    
    def build_system_prompt(self, user_email: str = None) -> str:
        """构建系统提示"""
        persona = extract_agent_persona(user_email)
        profile_info = extract_profile_without_persona(user_email)
        memory = load_memory(user_email)
        recent_diaries = load_recent_diaries(user_email)

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
5. 违规示例 ❌：```json{{"reply": "你好"}}``` 或 系统时间同步至...{{"reply": "你好"}}
6. 正确示例 ✅：{{"reply": "你好"}}

【绝对禁止 - 违者重罚】
- 禁止编造任何数字：包括但不限于"间隔X秒"、"频率X%"、"持续X分钟"、"下降X%"、"约X个月"等所有无法验证的精确数据
- 禁止虚假分析：如"检测到重复提问"、"符合人类行为模式"、"根据历史数据"等AI式过度解读
- 禁止学术腔：如"结论"、"综上所述"、"数据表明"、"研究显示"等论文用语
- 禁止扮演分析师：你不是在做数据分析，是在和朋友闲聊
- 如果不确定具体数据，只说定性结论（如"有一段时间了"、"比上次久"），绝不用数字"""

        return system_prompt
    
    def _has_image_reference(self, content: str) -> bool:
        """检查消息内容是否包含图片引用"""
        import re
        return bool(re.search(r'\[图片:\s*\S+\]', content))

    def _extract_image_info(self, content: str, user_email: str = None):
        """从消息中提取图片路径和纯文本"""
        import re
        from src.utils.file_utils import DATA_DIR
        from src.utils.user_utils import get_user_id_by_email

        image_pattern = r'\[图片:\s*(\S+)\]'
        matches = re.findall(image_pattern, content)
        text = re.sub(image_pattern, '', content).strip()

        image_paths = []
        if matches and user_email:
            user_id = get_user_id_by_email(user_email)
            if user_id:
                for filename in matches:
                    filepath = os.path.join(DATA_DIR, "users", user_id, "images", filename)
                    if os.path.exists(filepath):
                        image_paths.append(filepath)

        return text, image_paths

    def generate_response(self, content: str, history: list, user_email: str = None) -> Dict[str, Any]:
        """生成聊天响应"""
        # 构建系统提示
        system_prompt = self.build_system_prompt(user_email)

        # 检查是否包含图片 -> 使用多模态模型
        has_image = self._has_image_reference(content)

        # 构建消息数组，保留完整对话历史（让MOSS理解上下文逻辑）
        valid_history = [msg for msg in history if msg.get("content")]

        print(f"[Chat] content='{content[:80]}', has_image={has_image}")
        if has_image:
            # 多模态消息：使用 qwen-vl-max
            text, image_paths = self._extract_image_info(content, user_email)
            print(f"[Chat] Multimodal mode: text='{text}', image_paths={image_paths}")
            user_content = []
            for img_path in image_paths:
                user_content.append({"image": f"file://{img_path}"})
            user_content.append({"text": text if text else "请描述这张图片"})

            messages = [
                {"role": "system", "content": [{"text": system_prompt}]},
                *[{"role": m["role"], "content": [{"text": m["content"]}]} for m in valid_history[-10:]],
                {"role": "user", "content": user_content}
            ]
            model = "qwen-vl-max"
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                *valid_history[-10:],  # 最近10轮对话
                {"role": "user", "content": content}
            ]
            model = self.model

        # 调用API
        if has_image:
            response = MultiModalConversation.call(
                api_key=dashscope.api_key,
                model=model,
                messages=messages,
            )
        else:
            response = Generation.call(
                api_key=dashscope.api_key,
                model=model,
                messages=messages,
                result_format="message",
                enable_search=False,
                enable_thinking=False,
            )

        print(f"[Chat] API response status={response.status_code}, model={model}")
        if response.status_code == 200:
            # 提取回复内容
            if has_image:
                # MultiModalConversation 返回格式不同
                raw_content = response.output.choices[0].message.content
                print(f"[Chat] Multimodal raw response: {raw_content}")
                content = raw_content[0]["text"] if isinstance(raw_content, list) else raw_content
            else:
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
    
    def _background_archive(self, user_email: str = None):
        """后台执行归档（不阻塞主流程）"""
        try:
            if self.archive_service:
                auto_archive_expired_drafts(self.archive_service, user_email)
                print(f"[Background] Auto-archive completed for user: {user_email}")
        except Exception as e:
            print(f"[Background] Auto-archive failed for user {user_email}: {e}")
    
    def _has_expired_drafts(self, user_email: str = None) -> bool:
        """检查是否存在过期草稿（非今天的草稿）"""
        import os
        from datetime import date
        from src.utils.file_utils import _get_diaries_dir, _parse_draft_date
        
        diaries_dir = _get_diaries_dir(user_email)
        if not os.path.exists(diaries_dir):
            return False
        
        today = date.today()
        for filename in os.listdir(diaries_dir):
            if filename.endswith('_draft.md'):
                draft_date = _parse_draft_date(filename)
                if draft_date and draft_date < today:
                    return True
        return False
    
    def process_chat(self, content: str, history: list, user_email: str = None) -> Dict[str, Any]:
        """处理聊天请求（归档改为后台异步执行）"""
        # 1. 检查是否存在过期草稿，并提交归档任务到后台（非阻塞）
        archiving_notice = None
        if self.archive_service:
            if self._has_expired_drafts(user_email):
                archiving_notice = "检测到有未归档draft日记，将在后台自动完成归档"
            self._archive_executor.submit(self._background_archive, user_email)
        
        # 2. 立即生成响应（不等待归档完成）
        result = self.generate_response(content, history, user_email)
        
        reply_text = result.get("reply", "")
        
        # 3. 追加到今天的草稿文件
        filename = append_to_draft(content, reply_text, user_email)
        
        response = {
            "reply": reply_text,
            "filename": filename,
            "tokens": result.get("tokens", {
                "input": 0,
                "output": 0,
                "total": 0
            })
        }
        
        # 4. 如果有归档提示，添加到响应中
        if archiving_notice:
            response["archiving_notice"] = archiving_notice
        
        return response