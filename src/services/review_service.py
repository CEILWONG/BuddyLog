import re
import datetime
from typing import Dict, Any, Optional

from openai import OpenAI

from src.utils.file_utils import (
    load_review,
    save_review,
    load_all_archived_diaries_for_review,
)
from src.utils.user_utils import (
    get_user_by_email,
    get_user_review_meta,
    update_user_review_meta,
)


# 更新条件常量
REQUIRED_DAYS = 1
REQUIRED_DELTA = 5


class ReviewService:
    """复盘服务：基于全部归档日记生成三段式复盘内容"""

    def __init__(self, model: str, openai_client: OpenAI = None, enable_thinking: bool = False):
        self.model = model
        self.openai_client = openai_client
        self.enable_thinking = enable_thinking

    # ---------------- 公共 API ----------------

    def get_review(self, user_email: str) -> Dict[str, Any]:
        """获取当前用户的复盘内容、元数据与可更新状态"""
        content = load_review(user_email)
        meta = get_user_review_meta(user_email)
        sections = self._split_sections(content) if content else {}
        can_update_info = self.can_update_review(user_email)

        return {
            "exists": bool(content),
            "content": content,
            "sections": sections,
            "meta": meta,
            "can_update_info": can_update_info,
        }

    def can_update_review(self, user_email: str) -> Dict[str, Any]:
        """判断当前用户是否可以更新复盘"""
        user_info = get_user_by_email(user_email) or {}
        usage = user_info.get("usage", {}) or {}
        total_conversations = int(usage.get("total_conversations", 0) or 0)

        meta = get_user_review_meta(user_email)
        generated_date_str = meta.get("generated_date")
        baseline = int(meta.get("baseline_total_conversations", 0) or 0)

        # 首次（含存量用户无 review 字段）
        if not generated_date_str:
            return {
                "can_update": True,
                "reason": "首次复盘",
                "days_since": 0,
                "conversations_delta": total_conversations,
                "required_days": REQUIRED_DAYS,
                "required_delta": REQUIRED_DELTA,
                "is_first": True,
            }

        try:
            generated_date = datetime.date.fromisoformat(generated_date_str)
        except Exception:
            # 元数据脏数据，按首次处理
            return {
                "can_update": True,
                "reason": "元数据异常，可重新生成",
                "days_since": 0,
                "conversations_delta": total_conversations,
                "required_days": REQUIRED_DAYS,
                "required_delta": REQUIRED_DELTA,
                "is_first": True,
            }

        today = datetime.date.today()
        days_since = (today - generated_date).days
        conversations_delta = total_conversations - baseline

        if days_since >= REQUIRED_DAYS and conversations_delta >= REQUIRED_DELTA:
            return {
                "can_update": True,
                "reason": "满足更新条件",
                "days_since": days_since,
                "conversations_delta": conversations_delta,
                "required_days": REQUIRED_DAYS,
                "required_delta": REQUIRED_DELTA,
                "is_first": False,
            }

        # 不满足
        unmet = []
        if days_since < REQUIRED_DAYS:
            unmet.append(f"距上次复盘需≥{REQUIRED_DAYS}天（当前{days_since}天）")
        if conversations_delta < REQUIRED_DELTA:
            unmet.append(f"新增对话需≥{REQUIRED_DELTA}轮（当前{conversations_delta}轮）")
        return {
            "can_update": False,
            "reason": "；".join(unmet),
            "days_since": days_since,
            "conversations_delta": conversations_delta,
            "required_days": REQUIRED_DAYS,
            "required_delta": REQUIRED_DELTA,
            "is_first": False,
        }

    def generate_review(self, user_email: str) -> Dict[str, Any]:
        """生成（或重新生成）复盘内容

        说明：
        - 仅以用户原始发言（所有归档日记中 **用户**: 行）作为投喂语料。
        - 不投喂长期记忆（memory.md 是增量刷新产物，可能带进偏差）。
        - 不投喂 AI 回复与日记总结（在 _extract_conversation_from_archive 已过滤）。
        """
        diaries = load_all_archived_diaries_for_review(user_email)
        if not diaries:
            return {
                "success": False,
                "error": "还没有归档日记，先去聊聊吧",
            }

        diary_text = "\n\n".join(
            f"【{item['date']}】\n{item['user_text']}" for item in diaries
        )

        prompt = self._build_prompt(diary_text)
        messages = [{"role": "system", "content": prompt}]

        extra_body = {"enable_thinking": self.enable_thinking}

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body=extra_body,
            )
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")

        review_content = (response.choices[0].message.content or "").strip()
        if not review_content:
            return {
                "success": False,
                "error": "AI 未返回有效内容，请稍后再试",
            }

        # 写入正文
        save_review(review_content, user_email)

        # 写入元数据
        user_info = get_user_by_email(user_email) or {}
        usage = user_info.get("usage", {}) or {}
        total_conversations = int(usage.get("total_conversations", 0) or 0)
        now = datetime.datetime.now()
        meta = {
            "generated_at": now.isoformat(timespec="seconds"),
            "generated_date": now.date().isoformat(),
            "baseline_total_conversations": total_conversations,
            "diary_count": len(diaries),
        }
        update_user_review_meta(user_email, meta)

        sections = self._split_sections(review_content)
        return {
            "success": True,
            "exists": True,
            "content": review_content,
            "sections": sections,
            "meta": meta,
            "can_update_info": self.can_update_review(user_email),
        }

    # ---------------- 内部工具 ----------------

    def _build_prompt(self, diary_text: str) -> str:
        """构建复盘 system prompt，强约束三段式输出

        说明：仅以用户原始发言为唯一语料，不投喂长期记忆、AI 回复、日记总结等交叉产物，
        避免多轮推论偏差。"""
        return f"""你是一位高级心理成长学专家，同时也是这位用户的私人观察者。请仅基于用户亲口说出的原始话语（日记中的本人发言），为TA撰写一篇高质量的复盘报告。

【输入资料】

# 用户历史日记（按日期升序，仅含用户本人发言）
{diary_text}

【输出要求 - 严格遵守】
1. 严格输出 Markdown，且只能包含以下三个二级标题（## 开头），顺序与标题文案不得变动：
   - `## 人物画像`
   - `## 回忆高光`
   - `## 洞察`
2. 三段以外不得出现任何标题、前言、后记、致谢、结语等内容。
3. 严禁编造日记中没有提到的事件、人物或细节，所有内容必须可以在日记中找到依据。
4. 使用第三人称（“TA”）或第二人称（“你”）的温和口吻，禁止使用“我”代指用户。

【三段具体要求】

## 人物画像
- 仅基于用户日记原话，凝练 3-5 条客观画像，覆盖性格特质、当前关注点、近期状态、价值倾向等。
- 每条一行，使用无序列表（- 开头）。
- 描述需具体而克制，避免空洞的赞美或贴标签。

## 回忆高光
- 从所有日记中挑选值得被记住的好时刻，分点列出（无序列表）。
- 每条一行，格式建议：`- 【YYYY-MM-DD】简洁描述具体事件或场景`。
- 数量 4-8 条，覆盖不同主题或时间段，按时间倒序（新→旧）。
- 只挑选日记中真实出现过的高光时刻，不要泛泛而谈。

## 洞察
- 先用 1-2 段（每段 2-4 句）分析最近的性格倾向、情绪波动和主要关注点，语气真诚、专业、富同理心。
- 然后另起一行，使用 `### 发散性问题` 三级小节，列出 1-2 个值得用户深思的开放式问题（无序列表）。
- 然后另起一行，使用 `### 未来一段时间的建议` 三级小节，给出 3-5 条针对生活/工作的可执行建议（无序列表，每条一行）。
- 全部基于日记真实内容推演，不要给出与用户处境无关的通用鸡汤。

请直接开始输出 Markdown，不要重复以上要求。"""

    @staticmethod
    def _split_sections(content: str) -> Dict[str, str]:
        """按三个固定 H2 标题切分 Markdown，返回各段正文（不含标题）"""
        if not content:
            return {}

        # 匹配三个标题位置
        titles = ["人物画像", "回忆高光", "洞察"]
        # 用正则按 ## 标题切片
        pattern = re.compile(r'^##\s+(.+?)\s*$', re.MULTILINE)
        matches = list(pattern.finditer(content))
        if not matches:
            return {}

        sections: Dict[str, str] = {}
        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[start:end].strip()
            if title in titles:
                sections[title] = body

        return sections
