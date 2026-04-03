import json
import json5
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
from openai import OpenAI


class ChatService:
    """聊天服务类"""
    
    def __init__(self, model: str, openai_client: OpenAI, archive_service=None, enable_thinking: bool = False):
        self.model = model
        self.openai_client = openai_client
        self.archive_service = archive_service
        self.enable_thinking = enable_thinking
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
**重要：以下内容只包含用户的原话，不包含AI之前的回复。请基于用户的真实发言进行回忆和关联，不要参考AI自己之前说过的话（那些可能包含幻觉）。**

{recent_diaries_text}

【你的档案】
{profile_info if profile_info else '慢慢了解中'}

【思维链 - 回复前必须执行】
在回复用户前，请按以下步骤思考（但不要把思考过程输出给用户）：

1. **语义意图分析**
   - 用户这句话是在：记录今天的事 / 询问之前的事 / 表达情绪 / 闲聊 / 提问？
   - 用户的核心诉求是什么？

2. **回忆关联（重点）**
   - 查看【记住的事】，有没有和用户今天说的内容相关的背景？
   - 查看【最近聊的】，**只参考用户的原话，不参考AI之前的回复**：
     * 【最近聊的】里只有用户说过的话，没有AI的回复
     * 如果用户提到"上次"、"之前"、"昨天"等时间词，结合当前日期计算具体是哪一天
     * 检查那一天的**用户原话**，确认具体发生了什么
     * 不要混淆不同日期的事件
   - 如果用户提到了具体的人/事/物，只在用户的原话中查找是否聊过相关的

3. **时间线验证（重点，必须仔细计算）**
   - 当前是 {date_str} {weekday}
   - **如果用户提到"第X天"**：
     * "第1天" = 当天，"第2天" = 前一天，以此类推
     * 例：3月16号说"第5天" → 3月16号是第5天 → 开始日期是3月12号（16-5+1=12）
   - **如果用户说"前几天"、"上周"、"上个月"**：
     * "前几天"通常指3-5天前
     * "上周"指上周一到上周日，不是7天前
     * "上个月"指上一个自然月
   - **验证逻辑**：
     * 先找到用户提到该事的最早日期
     * 根据当时的描述推算实际发生日期
     * 不要直接把提到日期当成发生日期

4. **生成回复**
   - 基于以上分析，给出自然、真诚的回复
   - 如果回忆起了相关的事，自然地提及（"我记得你上周也提过..."）
   - 如果是新话题，就当朋友间闲聊回应，给予一定情绪价值
   - 如果是提问题，则引用一定的专业知识进行回复

5. **回复内容自检（必须执行）**
   在最终输出前，逐句检查你的回复是否包含以下问题：
   - ❌ 是否编造了具体数字（如"间隔X秒"、"持续X分钟"、"下降X%"）？
   - ❌ 是否声称进行了"数据分析"、"检测到"、"根据历史记录"等AI式分析？
   - ❌ 是否使用了论文腔（"综上所述"、"研究表明"）？
   - ❌ 是否混淆了不同日期的事件（把上周的事说成昨天的）？
   - ❌ **时间计算是否正确？**
     * 如果用户说"第X天"，是否正确推算出了开始日期？
     * 是否错误地把"提到日期"当成了"事件发生日期"？
   - ❌ **是否添加了【记住的事】和【最近聊的】里没有的细节？**
     * 比如编造具体的人名、物品、地点、对话内容？
     * 比如"XX给你做了XX"、"你买了XX"、"你跟XX说"等具体情节？
   - ❌ 是否把"可能"、"也许"的事情说成了确定的？
   
   **自检标准**：回复中的每一个具体细节，都必须能在【记住的事】或【最近聊的】中找到原文依据。
   
   如果发现以上任何问题，立即修改回复：
   - 删除所有编造的具体细节
   - 不确定的事情用"好像"、"可能"、"我不太确定"等模糊表达
   - 或者直接说"我记得你提过溃疡的事，但具体细节我记不清了"

【输出格式 - 严格遵守】
1. 必须且只能返回 JSON 格式：{{"reply": "你的回复内容"}}
2. 不要添加 Markdown 代码块标记（如 ```json）
3. 不要添加任何前缀或后缀文字
4. 不要添加系统状态信息
5. 违规示例 ❌：```json{{"reply": "你好"}}``` 或 系统时间同步至...{{"reply": "你好"}}
6. 正确示例 ✅：{{"reply": "你好"}}

【绝对禁止 - 违者重罚】
- 禁止编造任何数字：包括但不限于"间隔X秒"、"频率X%"、"持续X分钟"、"下降X%"、"约X个月"等所有无法验证的精确数据
- 禁止编造任何细节：包括但不限于人名、物品、地点、对话内容、具体情节（如"XX给你做了XX"、"你买了XX"）
- 禁止虚假分析：如"检测到重复提问"、"符合人类行为模式"、"根据历史数据"等AI式过度解读
- 禁止学术腔：如"结论"、"综上所述"、"数据表明"、"研究显示"等论文用语
- 禁止扮演分析师：你不是在做数据分析，是在和朋友闲聊
- 禁止"脑补"情节：如果【记住的事】和【最近聊的】里没有提到，你绝对不能说"记得你..."
- 如果不确定具体数据，只说定性结论（如"有一段时间了"、"比上次久"），绝不用数字
- 如果不确定具体细节，直接说"我记得你提过这件事，但具体细节我记不清了"，绝不编造"""

        return system_prompt
    
    def generate_response(self, content: str, history: list, user_email: str = None) -> Dict[str, Any]:
        """生成聊天响应"""
        # 构建系统提示
        system_prompt = self.build_system_prompt(user_email)
        
        # 构建消息数组，保留完整对话历史（让MOSS理解上下文逻辑）
        valid_history = [msg for msg in history if msg.get("content")]
        messages = [
            {"role": "system", "content": system_prompt},
            *valid_history[-10:],  # 最近10轮对话
            {"role": "user", "content": content}
        ]
        
        # 调用API (OpenAI 格式)
        # 构建 extra_body 参数（明确传递 enable_thinking 控制深度思考）
        extra_body = {"enable_thinking": self.enable_thinking}
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body=extra_body
            )
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")
        
        # 提取回复内容
        raw_content = response.choices[0].message.content
        
        # 清理可能的 Markdown 代码块标记
        content = raw_content
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # 解析 JSON 响应
        result = None
        try:
            result = json.loads(content)
            # 确保 result 是字典且有 reply 字段
            if not isinstance(result, dict) or "reply" not in result:
                result = {"reply": content if not isinstance(result, dict) else str(result)}
        except json.JSONDecodeError:
            # 标准 JSON 解析失败，尝试使用 json5 解析（更宽松，支持转义字符、注释等）
            try:
                result = json5.loads(content)
                # 确保 result 是字典且有 reply 字段
                if not isinstance(result, dict) or "reply" not in result:
                    result = {"reply": content if not isinstance(result, dict) else str(result)}
            except Exception:
                # json5 也解析失败，尝试用正则提取 reply 字段
                import re
                # 匹配 {"reply": "..."} 格式，支持转义引号
                match = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', content)
                if match:
                    reply_content = match.group(1)
                    # 处理 JSON 转义字符
                    try:
                        # 用 json.loads 解析转义字符串
                        reply_content = json.loads(f'"{reply_content}"')
                    except json.JSONDecodeError:
                        # 如果解析失败，手动处理常见转义
                        reply_content = reply_content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                    result = {"reply": reply_content}
                else:
                    # 完全无法解析，将清理后的内容作为 reply
                    result = {"reply": content}
        
        # 获取 token 使用量
        usage = response.usage
        if usage:
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0
            total_tokens = usage.total_tokens or (input_tokens + output_tokens)
            result['tokens'] = {
                'input': input_tokens,
                'output': output_tokens,
                'total': total_tokens
            }
        
        return result
    
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