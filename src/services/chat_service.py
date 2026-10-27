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
from src.utils.user_utils import extract_usage_tokens, update_user_usage
from src.utils.llm_utils import (
    CREATIVE_TEMPERATURE,
    describe_llm_error,
    llm_params,
    resolve_llm,
)
from openai import OpenAI


class ChatService:
    """聊天服务类"""
    
    def __init__(self, model: str, openai_client: OpenAI, archive_service=None, enable_thinking: bool = False):
        # model / openai_client 为系统默认值；实际调用时按 user_email 解析（见 llm_utils.resolve_llm）
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
        # 注意：不注入秒级/分钟级时间——高频变化的内容会破坏提供商的 prefix cache，
        # 导致 system prompt 中大量固定内容每轮都全价计费。日期一天才变一次，缓存友好。
        now = datetime.datetime.now()
        weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekday = weekdays[now.weekday()]
        date_str = now.strftime('%Y-%m-%d')

        # 系统状态放在 prompt 末尾（易变内容后置），保证固定部分作为稳定前缀命中缓存
        system_prompt = f"""{persona if persona else '你是 Moss,一个陪用户聊每天日常的老朋友。'}

【记住的事】
{memory if memory else '暂时还不了解太多'}

【最近聊的】
每段对话标注了日期，【今天】表示当天已发生的对话。
**重要：以下内容只包含用户的原话，不包含AI之前的回复。请基于用户的真实发言进行回忆和关联，不要参考AI自己之前说过的话（那些可能包含幻觉）。**

{recent_diaries_text}

【你的档案】
{profile_info if profile_info else '慢慢了解中'}

【你所在 App 的功能（用户问起使用方法时据此回答）】
你和用户聊天的地方叫 BuddyLog，除了聊天还有这些功能：
- **聊天模式**：底部输入框，适合碎碎念式的日常记录
- **文章模式**：输入框上方可切换，输入区变成半屏大小，适合写长篇内容；发送后你会把文章当作认真读过的素材来回应
- **次记**（底部导航「📊 次记」，目前在手机端提供）：数据化的记录，和日记互补。用户新建记录项目，选好记录方式和图表，之后每天记一条，长期看趋势：
  * 数值型：如体重（kg）、跑步公里数（km），可选折线图（看变化趋势）或热力图（看记录频率和多少分布）
  * 是否型：如今天是否喝了饮料，固定用热力图，记录过的日子会着色，一眼看出频率
  * 次记首页直接展示每个项目的图表和一句话总结，点卡片进入详情页记当天数据
- **日记档案**（底部导航「日记档案」）：每天的对话会归档成正式日记，页面顶部有日记热力图
- **复盘**（底部导航「复盘」）：阶段性回顾和总结

当用户问起"怎么记体重"、"能不能记跑步"、"怎么记喝没喝饮料"这类问题时：
- 告诉TA去「次记」页新建对应项目：起个名字，选数值或是否，数值型填单位、选折线图或热力图，之后每天记一条
- 不要假装自己能替用户记录这些数据——你只负责聊天，数据要用户自己在次记页填
- 用户聊到自己记的数据时（如"我今天称了体重"），可以自然回应并顺带提醒TA记到次记里
- 保持朋友口吻，别写成说明书

【怎么回】
1. 先接住当下：用户这句话在说什么、情绪是什么，先回应这个。大多数时候只关注当下。
2. 旧事只在自然相关时提：用户主动提“上次/之前/还记得吗”，或当下场景与某段经历明显相似时，用一句带过即可；不要每句都翻旧账，也不要为了显得记得而硬塞回忆。
3. 允许有自己的判断：给一个具体的看法、角度或问题，比“看情况”“因人而异”更有价值；不确定就直说不知道，不要装懂。
4. 具体细节必须有依据：人名、地点、数字、“你之前说过…”都必须能在【记住的事】或【最近聊的】里找到原文；找不到就说记不清，绝不补全。用户提到“第X天/前几天/上周”时，先换算成具体日期再回答，算不准就说记不清是哪天。
5. 不做伪分析：不写“检测到”“根据历史数据”“频率X%”“连续说了两次”，不写论文腔，不加系统状态。

【输出格式】
只返回 JSON：{{"reply": "你的回复内容"}}
不要 Markdown 代码块、不要前缀或后缀、不要附加系统状态。

【系统状态】
当前日期：{date_str} {weekday}"""

        return system_prompt
    
    # 文章模式下注入到本轮用户消息前的说明（只改本轮 user turn，不动 system prompt，保持 prefix cache 命中）
    ARTICLE_USER_PREFIX = (
        "（这条消息不是日常闲聊，而是用户认真写的一篇文章。请先完整读完，理解TA想表达的内容、情绪和思考，"
        "然后像一位认真读完了老朋友文章的朋友那样回应：保持你们一贯的朋友口吻，但回复要比平时更有深度——"
        "可以谈文章触动了你什么、你的真实看法和共鸣、值得延伸探讨的点，而不是泛泛地夸奖，也不是给修改建议。"
        "回复长度与文章的内容量匹配，可以适当更长。）\n\n【文章正文】\n"
    )

    def generate_response(self, content: str, history: list, user_email: str = None,
                          is_article: bool = False) -> Dict[str, Any]:
        """生成聊天响应（is_article=True 时本轮用户消息为文章）"""
        # 构建系统提示
        system_prompt = self.build_system_prompt(user_email)

        # 构建消息数组，保留完整对话历史（让MOSS理解上下文逻辑）
        valid_history = [msg for msg in history if msg.get("content")]
        content_for_llm = f"{self.ARTICLE_USER_PREFIX}{content}" if is_article else content
        messages = [
            {"role": "system", "content": system_prompt},
            *valid_history[-10:],  # 最近10轮对话
            {"role": "user", "content": content_for_llm}
        ]
        
        # 调用API (OpenAI 格式)
        # 按用户解析生效的客户端与模型（自带 Key / 自选模型优先，未配置回落系统默认）
        ctx = resolve_llm(user_email)
        
        try:
            response = ctx.client.chat.completions.create(
                model=ctx.model,
                messages=messages,
                **llm_params(self.enable_thinking, CREATIVE_TEMPERATURE),
            )
        except Exception as e:
            raise Exception(describe_llm_error(e, ctx))
        
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
            # 观测 prefix cache 命中情况（DashScope/DeepSeek 与 OpenAI 两种字段风格）
            cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
            if cache_hit is None:
                details = getattr(usage, "prompt_tokens_details", None)
                cache_hit = getattr(details, "cached_tokens", None) if details else None
            print(f"[TokenUsage] user={user_email} input={input_tokens} cached={cache_hit} output={output_tokens}")
        
        return result
    
    def generate_greeting(self, user_email: str = None) -> str:
        """生成个性化开场白"""
        persona = extract_agent_persona(user_email)
        memory = load_memory(user_email)
        recent_diaries = load_recent_diaries(user_email)
        
        now = datetime.datetime.now()
        weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekday = weekdays[now.weekday()]
        date_str = now.strftime('%Y-%m-%d')
        # 只保留小时粒度（供开场白感知时段），避免秒级时间破坏 prefix cache
        time_str = now.strftime('%H:00')
        
        recent_text = '\n'.join(recent_diaries[:3]) if recent_diaries else '暂无近期的日记记录'
        
        greeting_prompt = f"""【任务】生成一句开场白

当前时间：{date_str} {weekday} {time_str}

【长期记忆】
{memory if memory else '暂无长期记忆'}

【近期日记】
{recent_text}

【要求】
请生成一个开场白，要求：
1. 如果长期记忆中有用户的小名/昵称/称呼偏好，用那个称呼用户（不要用"用户"或"你"，直接用名字）
2. 结合今天的日期（{weekday}）和长期记忆中用户的习惯/兴趣，闲聊1-2句
3. 最后，自然地问用户今天想写点什么日记，引导分享
4. 整体风格：温暖、简短、像老朋友打招呼，保持在3-5句话以内
5. 用纯文本输出，不要加任何 JSON 包装、代码块或格式标记

请直接输出开场白内容："""
        
        messages = [
            {"role": "system", "content": f"{persona if persona else '你是一个陪用户聊日常的老朋友，语气温暖自然。'}"},
            {"role": "user", "content": greeting_prompt}
        ]
        
        # 开场白同样按用户解析客户端与模型
        ctx = resolve_llm(user_email)
        
        try:
            response = ctx.client.chat.completions.create(
                model=ctx.model,
                messages=messages,
                **llm_params(self.enable_thinking, CREATIVE_TEMPERATURE),
            )
            reply = response.choices[0].message.content.strip()
            # 开场白的 LLM 消耗记账（只计 tokens 不计对话轮次）
            greeting_tokens = extract_usage_tokens(getattr(response, "usage", None))
            if user_email and greeting_tokens > 0:
                update_user_usage(user_email, tokens_increment=greeting_tokens)
            # 清洗可能的格式标记
            if reply.startswith('```'):
                reply = reply.split('\n', 1)[-1] if '\n' in reply else reply[3:]
            if reply.endswith('```'):
                reply = reply[:-3]
            return reply.strip()
        except Exception as e:
            print(f"生成开场白失败: {describe_llm_error(e, ctx)}")
            return f"早上好。今天是{date_str} {weekday}。\n今天想记录些什么呢？"
    
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
    
    def process_chat(self, content: str, history: list, user_email: str = None,
                     is_article: bool = False) -> Dict[str, Any]:
        """处理聊天请求（归档改为后台异步执行）"""
        # 1. 检查是否存在过期草稿，并提交归档任务到后台（非阻塞）
        archiving_notice = None
        if self.archive_service:
            if self._has_expired_drafts(user_email):
                archiving_notice = "检测到有未归档draft日记，将在后台自动完成归档"
            self._archive_executor.submit(self._background_archive, user_email)

        # 2. 立即生成响应（不等待归档完成）
        result = self.generate_response(content, history, user_email, is_article=is_article)

        reply_text = result.get("reply", "")

        # 3. 追加到今天的草稿文件（文章消息带标记，归档后仍可识别）
        filename = append_to_draft(content, reply_text, user_email, is_article=is_article)
        
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