import datetime
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import json5

from src.utils.memory_store import (
    apply_diff,
    apply_lifecycle,
    commit_staged,
    discard_staged,
    empty_store,
    format_user_messages,
    is_empty,
    load_or_bootstrap_store,
    load_store,
    render_markdown,
    save_markdown,
    save_staged,
    save_store,
)
from src.utils.file_utils import _get_user_base_dir, load_archived_conversations_by_date
from src.utils.user_utils import extract_usage_tokens, update_user_usage
from src.utils.llm_utils import (
    FACTUAL_TEMPERATURE,
    describe_llm_error,
    llm_params,
    resolve_llm,
)
from openai import OpenAI

# 同一用户的记忆写入必须串行：归档是后台线程，并发读旧写新会互相覆盖
_USER_LOCKS = {}
_USER_LOCKS_GUARD = threading.Lock()


def _get_user_lock(user_email) -> threading.Lock:
    key = user_email or "__default__"
    with _USER_LOCKS_GUARD:
        return _USER_LOCKS.setdefault(key, threading.Lock())


MEMORY_DIFF_PROMPT = """你是长期记忆整理助手。唯一任务：从用户本轮亲口说的话里抽取可长期复用的事实，输出增量 JSON。

【日期】{date_str}

【现有记忆（JSON，仅供去重与判断冲突，不要复述）】
{existing}

【本轮用户原话】
{user_text}

【抽取规则 - 严格遵守】
1. 只记录用户明确说出的内容。禁止根据常识、推测、上下文气氛或“AI 可能说过的话”补全任何细节。
2. 只记录长期有效的事实：身份、关系、偏好、目标、进行中的事项、明确发生的事件。
3. 一次性的情绪、当天临时安排、客套话、对 AI 的提问，不记录。
4. 每条不超过 40 字，用用户原话或最小改写；不写评价，不写“标志着/体现了/意味着/说明”这类解读。
5. 与现有记忆重复的不输出；与现有记忆冲突的，用 corrections 输出旧→新。
6. 拿不准是否真实、是否长期有效的，一律不记录。宁可少记，不要编。

【输出字段】
- core：对象，键只能是 职业/关系/喜好/当前状态/姓名/昵称，值是简洁的新事实（覆盖旧值）
- goals_add / tasks_add / values_add：字符串数组，新增条目
- goals_done / tasks_done / tasks_cancel：字符串数组，填现有记忆里对应条目的原文开头即可
- events：数组，元素 {{"date": "YYYY-MM-DD", "items": ["发生了什么"]}}
- corrections：数组，元素 {{"old": "旧内容片段", "new": "修正后内容"}}

【输出格式】只输出一个 JSON 对象，不要 Markdown 代码块，不要任何解释。没有可新增的内容就输出 {{}}。
"""


MEMORY_BATCH_PROMPT = """你是长期记忆整理助手。下面是用户多天的原话，按日期分段。任务：只从这些原话中抽取可长期复用的事实，按天输出增量 JSON。

【用户原话（按日期分段）】
{user_text}

【抽取规则 - 严格遵守】
1. 只记录用户明确说出的内容。禁止根据常识、推测、上下文气氛或“AI 可能说过的话”补全任何细节。
2. 只记录长期有效的事实：身份、关系、偏好、目标、进行中的事项、明确发生的事件。
3. 一次性的情绪、当天临时安排、客套话、对 AI 的提问，不记录。
4. 每条不超过 40 字，用用户原话或最小改写；不写评价，不写“标志着/体现了/意味着/说明”这类解读。
5. 事实必须归到它出现的那一天：events 的 date 要用那一段原文的日期；跨天都成立的身份、关系、偏好放进 core。
6. 目标/事项必须带状态：用户新说出的写 open；用户明确说做完了写 done；明确说不做了写 cancelled。完成/放弃的日期写用户说这句话的那天。
7. 不要根据时间推移自己推断状态（例如“已经过去两个月了，应该做完了”是禁止的）。
8. 同一件事在不同日期重复出现时，只在最早那天记一次；后续日期如果状态变化或说法改变，单独记一条。
9. 同一天内前后矛盾时，以最后一次说法为准，用 corrections 记录旧→新。
10. 拿不准是否真实、是否长期有效的，一律不记录。宁可少记，不要编。

【输出字段】
{{"days": [
  {{"date": "YYYY-MM-DD",
    "core": {{"职业": "...", "关系": "...", "喜好": "...", "当前状态": "...", "姓名": "...", "昵称": "..."}},
    "goals": [{{"text": "目标原文", "status": "open|done|cancelled", "date": "YYYY-MM-DD"}}],
    "tasks": [{{"text": "事项原文", "status": "open|done|cancelled", "date": "YYYY-MM-DD"}}],
    "values_add": [],
    "events": [{{"date": "YYYY-MM-DD", "items": ["发生了什么"]}}],
    "corrections": [{{"old": "旧内容片段", "new": "修正后内容"}}]}}
]}}

没有可新增内容的日期直接不输出。整个批次都没有内容就输出 {{"days": []}}。
只输出一个 JSON 对象，不要 Markdown 代码块，不要任何解释。
"""


def _parse_diff(raw: str):
    """解析模型返回的 JSON；容忍代码块包裹和 json5 语法。"""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    for parser in (json.loads, json5.loads):
        try:
            data = parser(text)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


class MemoryService:
    """记忆服务类（增量抽取 + 结构化合并，不再整篇重写）"""

    def __init__(self, model: str, openai_client: OpenAI = None, enable_thinking: bool = False):
        # model / openai_client 为系统默认值；实际调用时按 user_email 解析（见 llm_utils.resolve_llm）
        self.model = model
        self.openai_client = openai_client
        self.enable_thinking = enable_thinking

    def _call_extract(self, prompt: str, user_email: str = None, max_tokens: int = 1500):
        """调用模型做一次抽取。

        返回 (diff, problem)：成功时 problem 为 None；
        失败时 diff 为 None，problem 取 "api" / "parse" / "truncated"，
        以便重建时区分“该重试”和“该拆小批次”。
        """
        ctx = resolve_llm(user_email)
        try:
            response = ctx.client.chat.completions.create(
                model=ctx.model,
                messages=[{"role": "system", "content": prompt}],
                **llm_params(self.enable_thinking, FACTUAL_TEMPERATURE, max_tokens=max_tokens),
            )
        except Exception as e:
            print(f"[Memory] 抽取失败 user={user_email}: {describe_llm_error(e, ctx)}")
            return None, "api"

        choice = response.choices[0]
        raw = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)
        diff = _parse_diff(raw)
        if diff is None:
            problem = "truncated" if finish_reason == "length" else "parse"
            print(f"[Memory] 输出无法解析 user={user_email} ({problem}): {raw[:200]}")
            return None, problem

        memory_tokens = extract_usage_tokens(getattr(response, "usage", None))
        if user_email and memory_tokens > 0:
            update_user_usage(user_email, tokens_increment=memory_tokens)

        return diff, None

    def update_memory(self, conversation: list, date_str: str, user_email: str = None):
        """从用户原话中抽取事实增量，合并进结构化记忆。

        注意：只读取 role=user 的消息。AI 自己的回复不能作为事实来源，
        否则模型的自述与揣测会被永久写进长期记忆，并在后续对话里被当成真的。
        """
        user_text = format_user_messages(conversation)
        if not user_text.strip():
            return

        # 重建期间暂停日常增量：线上记忆保持不动，记下日期等切换后补跑，
        # 否则重建结束时的整体切换会把这些增量静默覆盖掉。
        if is_rebuild_active(user_email):
            _mark_pending_date(user_email, date_str)
            print(f"[Memory] 重建进行中，增量延后处理 user={user_email}, date={date_str}")
            return

        lock = _get_user_lock(user_email)
        with lock:
            store = load_or_bootstrap_store(user_email)
            existing = json.dumps(store, ensure_ascii=False, separators=(",", ":"))
            prompt = MEMORY_DIFF_PROMPT.format(
                date_str=date_str, existing=existing, user_text=user_text
            )

            diff, problem = self._call_extract(prompt, user_email, max_tokens=1500)
            if diff is None:
                return

            # 模型没有提取到新内容时不动文件，避免无意义的整篇渲染
            if not diff:
                print(f"[Memory] 无可新增事实，跳过写入 user={user_email}, date={date_str}")
                return

            new_store = apply_diff(store, diff, date_str)
            # 先写渲染视图再写事实源：让 memory.json 的 mtime 保持最新，
            # 避免被 load_or_bootstrap_store 误判成“用户手工编辑过 memory.md”。
            save_markdown(render_markdown(new_store), user_email)
            save_store(new_store, user_email)

            print(
                f"[Memory] 增量合并完成 user={user_email}, date={date_str}, "
                f"keys={sorted(diff.keys())}"
            )


# ==================== 按需重建 ====================

_REBUILD_STATES = {}
_REBUILD_GUARD = threading.Lock()

# 单批上限：天数与字符数取先到者，避免个别超长文章把单次输入撑爆
BATCH_MAX_DAYS = 15
BATCH_MAX_CHARS = 8000
# 并行批次数：几百天也能在几轮内跑完，同时避免打爆供应商限流
REBUILD_CONCURRENCY = max(1, int(os.getenv("MEMORY_REBUILD_CONCURRENCY", "3")))
# 成功率低于该阈值时放弃切换，线上记忆保持原样
REBUILD_MIN_SUCCESS_RATIO = float(os.getenv("MEMORY_REBUILD_MIN_SUCCESS_RATIO", "0.9"))
# 单批解析失败时拆半重试的最大深度
REBUILD_SPLIT_DEPTH = 2
REBUILD_STATE_FILENAME = "rebuild_state.json"


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _state_email(key):
    return None if key == "__default__" else key


def _state_path(key: str) -> str:
    return os.path.join(_get_user_base_dir(_state_email(key)), REBUILD_STATE_FILENAME)


def _persist_state(key: str, state: dict):
    """把重建状态落盘，进程重启后仍可判断“上次是否被中断”。"""
    try:
        path = _state_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[Memory] 重建状态落盘失败 user={_state_email(key)}: {e}")


def _load_persisted_state(key: str):
    try:
        with open(_state_path(key), "r", encoding="utf-8") as f:
            state = json.load(f)
        return state if isinstance(state, dict) else None
    except Exception:
        return None


def get_rebuild_status(user_email: str = None) -> dict:
    key = user_email or "__default__"
    with _REBUILD_GUARD:
        state = _REBUILD_STATES.get(key)
        if state:
            return dict(state)

        persisted = _load_persisted_state(key)
        if persisted and persisted.get("status") == "running":
            # 暂存区还没切换，线上记忆必然完整
            persisted["status"] = "interrupted"
            persisted["message"] = "服务重启，重建已中断，原记忆未改动"
            persisted["finished_at"] = _timestamp()
            _persist_state(key, persisted)
        return persisted or {"status": "idle"}


def is_rebuild_active(user_email: str = None) -> bool:
    """只认进程内状态：持久化的 running 可能是重启前遗留的。"""
    key = user_email or "__default__"
    with _REBUILD_GUARD:
        state = _REBUILD_STATES.get(key)
        return bool(state and state.get("status") == "running")


def _mark_pending_date(user_email: str = None, date_str: str = ""):
    key = user_email or "__default__"
    with _REBUILD_GUARD:
        state = _REBUILD_STATES.get(key)
        if not state:
            return
        pending = set(state.get("pending_dates") or [])
        if date_str:
            pending.add(date_str)
        state["pending_dates"] = sorted(pending)


def _make_batches(days):
    """按天数和字符预算把整段历史切成若干批，每批一次模型调用。"""
    batches = []
    current = []
    current_chars = 0

    for date_str, text in days:
        size = len(text)
        if current and (len(current) >= BATCH_MAX_DAYS or current_chars + size > BATCH_MAX_CHARS):
            batches.append(current)
            current = []
            current_chars = 0
        current.append((date_str, text))
        current_chars += size

    if current:
        batches.append(current)
    return batches


def _extract_batch(service: "MemoryService", batch, user_email: str = None):
    """把一批多天文本一次发给模型，返回 (diff, problem)，diff 统一为 {"days": [...]}。"""
    block = "\n\n".join(f"【{date_str}】\n{text}" for date_str, text in batch)
    prompt = MEMORY_BATCH_PROMPT.format(user_text=block)
    diff, problem = service._call_extract(prompt, user_email, max_tokens=3000)

    if isinstance(diff, dict) and diff and not isinstance(diff.get("days"), list):
        diff = {"days": [{"date": batch[0][0], **diff}]}
    return diff, problem


def _extract_batch_safe(service: "MemoryService", batch, user_email: str = None, depth: int = 0):
    """批次失败（尤其是输出被截断）时对半拆开重试，避免整批日期直接丢失。"""
    diff, problem = _extract_batch(service, batch, user_email)
    if diff is not None:
        return diff, None

    if depth < REBUILD_SPLIT_DEPTH and len(batch) > 1:
        mid = len(batch) // 2
        left, left_problem = _extract_batch_safe(service, batch[:mid], user_email, depth + 1)
        right, right_problem = _extract_batch_safe(service, batch[mid:], user_email, depth + 1)
        if left is None and right is None:
            return None, problem or left_problem or right_problem
        merged = {"days": []}
        for part in (left, right):
            if part:
                merged["days"].extend(part.get("days") or [])
        return merged, None

    return None, problem


def _iter_day_diffs(diff, batch):
    """把批次结果拆成 (日期, 单日 diff)；兼容模型直接返回扁平结构。"""
    if not isinstance(diff, dict) or not diff:
        return

    dates = [date_str for date_str, _ in batch]
    day_list = diff.get("days")

    if isinstance(day_list, list):
        for entry in day_list:
            if not isinstance(entry, dict):
                continue
            date_str = entry.get("date")
            payload = {k: v for k, v in entry.items() if k != "date"}
            if not payload:
                continue
            if date_str not in dates:
                date_str = dates[0]
            yield date_str, payload
    else:
        yield dates[0], diff


def _merge_batch_days(store, pairs):
    """按日期顺序把各批次结果合并进同一份记忆。

    抽取可以并行（每批是独立时间段），但合并必须有序：
    core 靠“后写覆盖前写”体现变化，目标/事项靠 apply_lifecycle 落定状态。
    """
    for date_str, payload in pairs:
        goals = payload.pop("goals", None)
        tasks = payload.pop("tasks", None)
        store = apply_diff(store, payload, date_str)
        store = apply_lifecycle(store, goals, "goals", date_str)
        store = apply_lifecycle(store, tasks, "open_tasks", date_str)
    return store


def start_rebuild(service: "MemoryService", user_email: str = None) -> dict:
    """按时间顺序重放全部已归档日记，重建长期记忆（后台线程）。

    线上记忆在整个过程中保持不动，成功后再原子切换。
    """
    key = user_email or "__default__"
    days = load_archived_conversations_by_date(user_email)

    with _REBUILD_GUARD:
        state = _REBUILD_STATES.get(key)
        if state and state.get("status") == "running":
            return {"started": False, "error": "长期记忆正在重建中，请稍后再试"}
        if not days:
            return {"started": False, "error": "没有可用的历史日记，暂无法重建"}

        batches = _make_batches(days)
        _REBUILD_STATES[key] = {
            "status": "running",
            "phase": "extract",
            "total": len(days),
            "done": 0,
            "failed": 0,
            "failed_ranges": [],
            "total_batches": len(batches),
            "done_batches": 0,
            "pending_dates": [],
            "current_date": "",
            "started_at": _timestamp(),
            "finished_at": "",
            "message": "",
        }

    # 上次异常退出可能留下暂存文件，先清掉
    discard_staged(user_email)
    _persist_state(key, dict(_REBUILD_STATES[key]))

    thread = threading.Thread(
        target=_run_rebuild,
        args=(service, user_email, key, batches),
        daemon=True,
        name="memory_rebuild",
    )
    thread.start()
    return {"started": True, "total": len(days), "batches": len(batches)}


def _update_state(key: str, **changes):
    with _REBUILD_GUARD:
        state = _REBUILD_STATES.get(key)
        if not state:
            return None
        state.update(changes)
        snapshot = dict(state)
    _persist_state(key, snapshot)
    return snapshot


def _run_rebuild(service, user_email, key, batches):
    total_days = sum(len(batch) for batch in batches)
    committed = False
    try:
        results = [None] * len(batches)
        failed_days = 0
        failed_ranges = []

        if batches:
            workers = max(1, min(REBUILD_CONCURRENCY, len(batches)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_extract_batch_safe, service, batch, user_email): idx
                    for idx, batch in enumerate(batches)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    batch = batches[idx]
                    try:
                        diff, problem = future.result()
                    except Exception as e:
                        diff, problem = None, str(e)

                    results[idx] = diff
                    if diff is None:
                        failed_days += len(batch)
                        failed_ranges.append(f"{batch[0][0]}~{batch[-1][0]}")
                        print(f"[Memory] 批次最终失败 user={user_email}, "
                              f"{batch[0][0]}~{batch[-1][0]}, reason={problem}")

                    _update_state(
                        key,
                        done=_REBUILD_STATES[key]["done"] + len(batch),
                        done_batches=_REBUILD_STATES[key]["done_batches"] + 1,
                        failed=failed_days,
                        failed_ranges=list(failed_ranges),
                        current_date=f"{batch[0][0]} → {batch[-1][0]}",
                    )

        _update_state(key, phase="merge", current_date="")

        # 按日期顺序合并，保证“较新的说法覆盖较旧的说法”
        pairs = []
        for batch, diff in zip(batches, results):
            if diff:
                pairs.extend(_iter_day_diffs(diff, batch))
        pairs.sort(key=lambda pair: pair[0])

        # 重建从空白开始：历史事实由这段日记重新抽取，不叠加旧记忆（旧记忆可能已被污染）
        store = _merge_batch_days(empty_store(), pairs)

        success_ratio = 1.0 if total_days == 0 else (total_days - failed_days) / total_days
        if success_ratio < REBUILD_MIN_SUCCESS_RATIO:
            discard_staged(user_email)
            message = (
                f"仅 {total_days - failed_days}/{total_days} 天抽取成功，"
                f"低于 {int(REBUILD_MIN_SUCCESS_RATIO * 100)}% 阈值，已取消切换，原记忆未改动"
            )
            _update_state(key, status="error", phase="done", message=message,
                          failed=failed_days, failed_ranges=list(failed_ranges),
                          current_date="", finished_at=_timestamp())
            print(f"[Memory] 重建放弃 user={user_email}: {message}")
            return

        # 先写暂存区，线上文件仍未被触碰
        save_staged(store, user_email)
        _update_state(key, phase="commit")

        pending_dates = []
        stamp = ""
        with _get_user_lock(user_email):
            stamp = commit_staged(user_email)
            committed = True
            try:
                pending_dates = _replay_pending(service, user_email, key)
            except Exception as e:
                # 已经切换完成，补跑失败不能再报“原记忆未改动”
                print(f"[Memory] 重建后补跑增量失败 user={user_email}: {e}")
                pending_dates = []

        empty = is_empty(store)
        messages = []
        if failed_days:
            messages.append(f"{failed_days} 天抽取失败，这些日期未写入新记忆")
        if pending_dates:
            messages.append(f"已补跑重建期间的 {len(pending_dates)} 天增量")
        if empty:
            messages.append("历史日记里没有提取到长期事实")

        _update_state(
            key,
            status="done",
            phase="done",
            current_date="",
            failed=failed_days,
            failed_ranges=list(failed_ranges),
            pending_dates=[],
            committed_at=stamp,
            message="；".join(messages),
            finished_at=_timestamp(),
        )

        print(
            f"[Memory] 重建完成 user={user_email}, days={total_days}, "
            f"batches={len(batches)}, failed_days={failed_days}, empty={empty}, stamp={stamp}"
        )
    except Exception as e:
        discard_staged(user_email)
        message = (
            f"新记忆已切换，但收尾步骤异常：{e}" if committed
            else f"重建异常，原记忆未改动：{e}"
        )
        _update_state(
            key,
            status="error",
            phase="done",
            message=message,
            current_date="",
            finished_at=_timestamp(),
        )
        print(f"[Memory] 重建异常 user={user_email}: {e}")


def _replay_pending(service: "MemoryService", user_email: str = None, key: str = None) -> list:
    """切换完成后补跑重建窗口内被推迟的增量。调用方需持有用户锁。"""
    key = key or (user_email or "__default__")
    with _REBUILD_GUARD:
        pending = list((_REBUILD_STATES.get(key) or {}).get("pending_dates") or [])
    if not pending:
        return []

    archived = dict(load_archived_conversations_by_date(user_email))
    store = load_store(user_email) or empty_store()
    replayed = []

    for date_str in pending:
        text = archived.get(date_str)
        if not text:
            # 例如今天的草稿还没归档，等它归档时会正常更新
            continue
        diff, _ = _extract_batch_safe(service, [(date_str, text)], user_email)
        if not diff:
            continue
        for day, payload in _iter_day_diffs(diff, [(date_str, text)]):
            goals = payload.pop("goals", None)
            tasks = payload.pop("tasks", None)
            store = apply_diff(store, payload, day)
            store = apply_lifecycle(store, goals, "goals", day)
            store = apply_lifecycle(store, tasks, "open_tasks", day)
        replayed.append(date_str)

    if replayed:
        save_markdown(render_markdown(store), user_email)
        save_store(store, user_email)
        print(f"[Memory] 已补跑重建期间的增量 user={user_email}, dates={replayed}")

    return replayed
