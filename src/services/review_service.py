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
    extract_usage_tokens,
    update_user_usage,
)


# 更新条件常量：至少间隔 2 天且新增 20 轮对话才能再次触发复盘
REQUIRED_DAYS = 2
REQUIRED_DELTA = 20


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
        - 仅以用户原始发言（归档日记中 **用户**: 行）作为投喂语料。
        - 不投喂长期记忆（memory.md 是增量刷新产物，可能带进偏差）。
        - 不投喂 AI 回复与日记总结（在 _extract_conversation_from_archive 已过滤）。
        - 存在上次复盘且基线有效时走「增量模式」：上次复盘结果 + 新增日记，大幅降低 token 消耗；
          首次复盘 / 元数据缺基线时回退全量模式。
        """
        previous_content = load_review(user_email)
        meta = get_user_review_meta(user_email)
        baseline_count = meta.get("diary_count")

        # 增量模式需要完整未截断的日记列表来按基线切片，故传大 max_chars
        diaries = load_all_archived_diaries_for_review(user_email, max_chars=10**9)
        if not diaries:
            return {
                "success": False,
                "error": "还没有归档日记，先去聊聊吧",
            }

        new_diaries = (
            diaries[baseline_count:]
            if isinstance(baseline_count, int) and 0 < baseline_count < len(diaries)
            else []
        )
        incremental = bool(previous_content) and bool(new_diaries)
        # 未截断列表的完整日记数，作为下次增量切片的基线（全量分支会重新赋值）
        total_diary_count = len(diaries)

        if incremental:
            # 增量模式：上次复盘（承载历史结论）+ 新增用户原话（仅新日记）
            if meta.get("earliest_date"):
                earliest = meta["earliest_date"]
            else:
                earliest = diaries[0].get("date") or datetime.date.today().isoformat()
            latest = new_diaries[-1].get("date") or datetime.date.today().isoformat()
            unique_months = sorted({
                (d.get("date") or "")[:7] for d in diaries if d.get("date")
            })
            span_months = len(unique_months) or 1
            prompt = self._build_incremental_prompt(
                previous_content,
                new_diaries,
                today_str=datetime.date.today().isoformat(),
                earliest=earliest,
                latest=latest,
                span_months=span_months,
            )
        else:
            if previous_content and not new_diaries:
                # 满足触发条件但没有新归档日记：新对话尚未归档，提示用户而非无米之炊
                return {
                    "success": False,
                    "error": "新增对话尚未归档，请先完成归档再更新复盘",
                }
            # 全量模式：首次复盘或存量用户无基线，保留原有截断策略控成本
            total_diary_count = len(diaries)
            diaries = load_all_archived_diaries_for_review(user_email)
            diary_text = "\n\n".join(
                f"【{item['date']}】\n{item['user_text']}" for item in diaries
            )
            today = datetime.date.today()
            diary_dates = [it["date"] for it in diaries if it.get("date")]
            earliest = diary_dates[0] if diary_dates else today.isoformat()
            latest = diary_dates[-1] if diary_dates else today.isoformat()
            unique_months = sorted({d[:7] for d in diary_dates}) if diary_dates else []
            span_months = len(unique_months)
            prompt = self._build_prompt(
                diary_text,
                today_str=today.isoformat(),
                earliest=earliest,
                latest=latest,
                span_months=span_months,
            )

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

        # 复盘的 LLM 消耗记账（只计 tokens 不计对话轮次）
        review_tokens = extract_usage_tokens(getattr(response, "usage", None))
        if review_tokens > 0:
            update_user_usage(user_email, tokens_increment=review_tokens)

        # 写入正文
        save_review(review_content, user_email)

        # 写入元数据（额外记录日记日期范围，供下次增量复盘重建时间锚点）
        user_info = get_user_by_email(user_email) or {}
        usage = user_info.get("usage", {}) or {}
        total_conversations = int(usage.get("total_conversations", 0) or 0)
        now = datetime.datetime.now()
        all_dates = [d.get("date") for d in diaries if d.get("date")]
        meta = {
            "generated_at": now.isoformat(timespec="seconds"),
            "generated_date": now.date().isoformat(),
            "baseline_total_conversations": total_conversations,
            "diary_count": total_diary_count,
            "earliest_date": all_dates[0] if all_dates else "",
            "latest_date": all_dates[-1] if all_dates else "",
            "mode": "incremental" if incremental else "full",
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

    def _build_prompt(
        self,
        diary_text: str,
        today_str: str = "",
        earliest: str = "",
        latest: str = "",
        span_months: int = 1,
    ) -> str:
        """构建复盘 system prompt，强约束三段式输出

        说明：仅以用户原始发言为唯一语料，不投喂长期记忆、AI 回复、日记总结等交叉产物，
        避免多轮推论偏差。同时注入时间锚点与分布约束，防止「高光焦集于早期月份」的偏斜问题。"""

        distribution_rule = self._distribution_rule(span_months)

        date_anchor = ""
        if today_str:
            date_anchor = (
                f"\n【时间锚点】今天是 {today_str}；"
                f"日记时间跨度：{earliest} → {latest}（共 {span_months} 个月份）。\n"
                "请以今天为参考系，明确区分「近期 / 远期」，不要被早期信息量大的月份吊走注意力。"
            )

        return f"""你是一位高级心理成长学专家，同时也是这位用户的私人观察者。请仅基于用户亲口说出的原始话语（日记中的本人发言），为TA撰写一篇高质量的复盘报告。
{date_anchor}
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
- 「近期状态」部分必须反映最近 30 天内的表达，不要拿几个月前的状态充数。

## 回忆高光
- 从所有日记中挑选值得被记住的好时刻，分点列出（无序列表）。
- 每条一行，格式严格为：`- 【YYYY-MM-DD】简洁描述具体事件或场景`，日期必须取自日记原始日期。
- 数量 4-8 条，覆盖不同主题与不同时间段。
{distribution_rule}
- 只挑选日记中真实出现过的高光时刻，不要泛泛而谈。
- 输出前请自检：高光日期是否集中于某一两个月？若是，必须重新调整。

## 洞察
- 先用 1-2 段（每段 2-4 句）分析最近的性格倾向、情绪波动和主要关注点，语气真诚、专业、富同理心。「最近」严格指最近 30 天内的日记。
- 然后另起一行，使用 `### 发散性问题` 三级小节，列出 1-2 个值得用户深思的开放式问题（无序列表）。
- 然后另起一行，使用 `### 未来一段时间的建议` 三级小节，给出 3-5 条针对生活/工作的可执行建议（无序列表，每条一行）。
- 全部基于日记真实内容推演，不要给出与用户处境无关的通用鸡汤。

请直接开始输出 Markdown，不要重复以上要求。"""

    @staticmethod
    def _distribution_rule(span_months: int) -> str:
        """动态生成「时间分布」约束文案（全量/增量提示词共用）"""
        if span_months >= 3:
            return (
                f"- 【时间分布硬约束】日记跨越 {span_months} 个月份，不得出现“高光集中某一个月”的现象：\n"
                f"  * 同一月份最多 2 条高光；\n"
                f"  * 至少覆盖 3 个不同月份；\n"
                f"  * 必须至少包含 2 条来自最近 30 天内的高光（即不早于「今天减 30 天」的日期）。如日记中近 30 天确无高光，可改为最近 60 天内的 2 条；仍无则明确说明。\n"
                f"- 按时间倒序排列（新→旧），第一条必须是最近期的。"
            )
        if span_months == 2:
            return (
                "- 【时间分布硬约束】日记跨越 2 个月份，两个月都必须有代表高光且数量差异不得超过 2 条；\n"
                "- 按时间倒序排列（新→旧），第一条必须是近期的。"
            )
        return (
            "- 按时间倒序排列（新→旧），优先选近期高光。"
        )

    def _build_incremental_prompt(
        self,
        previous_review: str,
        new_diaries: list,
        today_str: str,
        earliest: str,
        latest: str,
        span_months: int,
    ) -> str:
        """构建增量复盘 system prompt：上次复盘结果 + 新增用户原话，合并产出新一版三段式复盘。

        语料约束：新增部分仍只含用户本人发言；上次复盘作为「历史结论的浓缩载体」参与合并，
        但所有新增结论必须能在新增日记中找到依据，不得从旧复盘中无根据地推演新事实。"""
        new_diary_text = "\n\n".join(
            f"【{item['date']}】\n{item['user_text']}" for item in new_diaries
        )
        distribution_rule = self._distribution_rule(span_months)

        return f"""你是一位高级心理成长学专家，同时也是这位用户的私人观察者。你之前已经为TA写过一版复盘，现在用户又积累了新的对话，请基于「上一版复盘 + 新增对话」合并更新出新一版复盘报告。
【时间锚点】今天是 {today_str}；用户日记总时间跨度：{earliest} → {latest}（共 {span_months} 个月份）。
【输入资料】
# 上一版复盘（历史结论的浓缩载体）
{previous_review}

# 新增对话（自上次复盘之后的用户本人发言，按日期升序）
{new_diary_text}

【更新原则 - 严格遵守】
1. 严格输出 Markdown，且只能包含以下三个二级标题（## 开头），顺序与标题文案不得变动：
   - `## 人物画像`
   - `## 回忆高光`
   - `## 洞察`
2. 三段以外不得出现任何标题、前言、后记、致谢、结语等内容。
3. 新增结论（新画像、新高光、新洞察）必须能在新增对话中找到依据，严禁编造；上一版复盘中已有的内容默认视为已经过核实的可靠结论，可保留或修订。
4. 使用第三人称（“TA”）或第二人称（“你”）的温和口吻，禁止使用“我”代指用户。

【三段具体合并规则】

## 人物画像
- 在上一版画像基础上合并更新：有新证据支撑的条目保留并可细化，被新对话反映出的新特质/新关注点补充进来，已过时的表述修订或删除。
- 总数保持 3-5 条，每条一行，无序列表（- 开头）；「近期状态」必须反映新增对话中的表达。
- 描述具体而克制，避免空洞赞美或贴标签。

## 回忆高光
- 保留上一版中仍然有价值的高光条目（格式不变：`- 【YYYY-MM-DD】描述`），并从新增对话中增补真正值得记住的新时刻。
- 总数控制在 4-8 条；超出时优先淘汰时间最早或代表性最弱的旧条目，不得为了凑数而保留。
- 新增高光的日期必须取自新增对话的原始日期，严禁编造。
{distribution_rule}
- 输出前请自检：高光日期是否集中于某一两个月？若是，必须重新调整。

## 洞察
- 先用 1-2 段（每段 2-4 句）分析最近的性格倾向、情绪波动和主要关注点，语气真诚、专业、富同理心。「最近」以新增对话为主，可结合上一版洞察作对比：若状态发生变化，要点出变化轨迹。
- 然后另起一行，使用 `### 发散性问题` 三级小节，列出 1-2 个值得用户深思的开放式问题（无序列表）。
- 然后另起一行，使用 `### 未来一段时间的建议` 三级小节，给出 3-5 条针对生活/工作的可执行建议（无序列表，每条一行）；上一版建议中已落实或不再适用的可替换。
- 全部基于真实内容推演，不要给出与用户处境无关的通用鸡汤。

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
