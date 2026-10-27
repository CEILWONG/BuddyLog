"""结构化长期记忆存储。

设计目标：
- 记忆的“事实源”是 memory.json（结构化条目），memory.md 只是渲染视图；
- 更新时只接收增量 diff（新增/修正），由 Python 合并，不让模型整篇重写，
  避免长文复现漂移、润色和输出截断；
- 首次启用时把旧的 memory.md 原样拆分进 JSON（不调用模型、不丢内容），
  并保留一份 legacy 备份。
"""

import datetime
import json
import os
import re
import shutil
from typing import Any, Dict, Optional

from src.utils.file_utils import _get_user_base_dir

STORE_FILENAME = "memory.json"
MARKDOWN_FILENAME = "memory.md"

SECTION_ORDER = [
    ("core", "用户核心画像"),
    ("goals", "长期目标"),
    ("open_tasks", "未完结事项"),
    ("values", "价值观"),
    ("events", "重要事件"),
]

# 单次更新允许写入的条目上限，防止模型异常输出撑爆文件
MAX_ITEMS_PER_UPDATE = 20
MAX_ITEM_LENGTH = 200


def get_store_path(user_email: Optional[str] = None) -> str:
    return os.path.join(_get_user_base_dir(user_email), STORE_FILENAME)


def get_markdown_path(user_email: Optional[str] = None) -> str:
    return os.path.join(_get_user_base_dir(user_email), MARKDOWN_FILENAME)


def empty_store() -> Dict[str, Any]:
    return {
        "version": 1,
        "updated_at": "",
        "core": [],
        "goals": [],
        "open_tasks": [],
        "values": [],
        "events": [],
    }


def _read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text_atomic(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, path)


def load_store(user_email: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = get_store_path(user_email)
    if not os.path.exists(path):
        return None
    try:
        data = json.loads(_read_text(path))
        if not isinstance(data, dict):
            return None
        store = empty_store()
        store.update(data)
        return store
    except Exception as e:
        print(f"[MemoryStore] 读取失败，将回退到 memory.md：{e}")
        return None


def save_store(store: Dict[str, Any], user_email: Optional[str] = None):
    store["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_text_atomic(
        get_store_path(user_email),
        json.dumps(store, ensure_ascii=False, indent=2),
    )


def _markdown_edited_after_store(md_path: str, json_path: str) -> bool:
    """判断 memory.md 是否被用户手工改过（比事实源更新）。"""
    if not (os.path.exists(md_path) and os.path.exists(json_path)):
        return False
    try:
        return os.path.getmtime(md_path) > os.path.getmtime(json_path)
    except OSError:
        return False


def load_or_bootstrap_store(user_email: Optional[str] = None) -> Dict[str, Any]:
    """读取 JSON 事实源。

    - JSON 不存在：从旧 memory.md 无模型迁移一次；
    - memory.md 比 JSON 新：视为用户手工修正过记忆，以 Markdown 为准重建事实源。
      这样用户可以直接编辑 memory.md 删除不真实的条目，且不会被下次更新覆盖回去。
    """
    md_path = get_markdown_path(user_email)
    json_path = get_store_path(user_email)

    store = load_store(user_email)
    if store is not None:
        if _markdown_edited_after_store(md_path, json_path):
            rebuilt = bootstrap_from_markdown(_read_text(md_path))
            if not is_empty(rebuilt):
                try:
                    shutil.copy2(json_path, f"{json_path}.bak")
                except Exception as e:
                    print(f"[MemoryStore] 事实源备份失败：{e}")
                save_store(rebuilt, user_email)
                print(f"[MemoryStore] 检测到 memory.md 手工修改，已重建事实源 user={user_email}")
                return rebuilt
        return store

    legacy_text = _read_text(md_path)
    store = bootstrap_from_markdown(legacy_text)

    if legacy_text.strip():
        backup_path = f"{md_path}.legacy.bak"
        if not os.path.exists(backup_path):
            try:
                shutil.copy2(md_path, backup_path)
            except Exception as e:
                print(f"[MemoryStore] 旧记忆备份失败：{e}")

    save_store(store, user_email)
    return store


# ==================== 旧格式迁移 ====================


def _clean_item(line: str) -> str:
    text = line.strip()
    text = re.sub(r"^[-*+]\s+", "", text)
    return text.strip()


def _split_sections(markdown_text: str) -> Dict[str, str]:
    sections = {}
    current = None
    buffer = []
    for line in markdown_text.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*$", line)
        if heading:
            if current is not None:
                sections[current] = "\n".join(buffer)
            current = heading.group(1).strip()
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer)
    return sections


def _normalize_date(text: str) -> str:
    m = re.search(r"(\d{4})[年/\-.](\d{1,2})[月/\-.](\d{1,2})", text)
    if not m:
        return ""
    year, month, day = m.groups()
    try:
        return datetime.date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return ""


def _split_date_header(content: str):
    """把“**2026-04-29**：正文”拆成（日期, 正文），兼容正文直接跟在日期后。

    只有日期出现在条目开头时才视为日期标题；像“回溯病程：2026-04-07 晨起低烧…”
    这类日期在句中的条目必须原样保留，不能截断前半句。
    """
    match = re.match(r"^[\*\s]*(\d{4}[年/\-.]\d{1,2}[月/\-.]\d{1,2})", content)
    if not match:
        return "", content
    tail = content[match.end():].lstrip("*").strip()
    tail = re.sub(r"^[:：]\s*", "", tail).strip()
    return _normalize_date(match.group(1)), tail


def _parse_events(section_text: str):
    events = []
    current = None
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bullet = re.match(r"^-\s+(.*)$", line)
        if bullet:
            content = bullet.group(1).strip()
            date_str, remainder = _split_date_header(content)
            if date_str:
                current = {"date": date_str, "items": []}
                events.append(current)
                if remainder:
                    current["items"].append(remainder)
            else:
                if current is None:
                    current = {"date": "", "items": []}
                    events.append(current)
                current["items"].append(_clean_item(content))
            continue
        sub_bullet = re.match(r"^\s+[-*+]\s+(.*)$", raw_line)
        if sub_bullet and current is not None:
            current["items"].append(_clean_item(sub_bullet.group(1)))
            continue
        # 旧格式里也可能出现无项目符号的日期标题或叙述段落
        date_str, remainder = _split_date_header(line)
        if date_str and not remainder:
            current = {"date": date_str, "items": []}
            events.append(current)
        elif current is not None:
            current["items"].append(line)
        else:
            current = {"date": "", "items": [line]}
            events.append(current)
    return [e for e in events if e.get("items")]


def bootstrap_from_markdown(markdown_text: str) -> Dict[str, Any]:
    store = empty_store()
    if not markdown_text.strip():
        return store

    sections = _split_sections(markdown_text)

    for key, heading in SECTION_ORDER:
        text = sections.get(heading, "")
        if not text.strip():
            continue
        if key == "events":
            store["events"] = _parse_events(text)
            continue
        items = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if re.match(r"^[-*+]\s+", stripped):
                cleaned = _clean_item(stripped)
            else:
                # 旧记忆里也有整段叙述，不能只迁移列表项
                cleaned = stripped
            if cleaned:
                items.append(cleaned)
        store[key] = items

    return store


# ==================== 增量合并 ====================


def _norm(text: str) -> str:
    return re.sub(r"[\s，。,\.、；;：:！!？?（）()【】\[\]「」“”\"']", "", text or "").lower()


def _core_key(item: str) -> str:
    text = _clean_item(item)
    m = re.match(r"^\*\*(.+?)\*\*\s*[:：]", text)
    if m:
        return m.group(1).strip()
    m = re.match(r"^([^:：]{1,12})[:：]", text)
    if m:
        return m.group(1).strip()
    return ""


def _clip(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip()[:MAX_ITEM_LENGTH]


def _append_unique(items, text: str):
    clean = _clip(text)
    if not clean:
        return
    norm = _norm(clean)
    for existing in items:
        other = _norm(existing)
        if norm == other or (len(norm) > 4 and (norm in other or other in norm)):
            return
    items.append(clean)


def _replace_in_list(items, old: str, new: str, preserve_key: bool = False):
    old_norm = _norm(old)
    if len(old_norm) < 2:
        return
    for idx, existing in enumerate(items):
        if old_norm in _norm(existing):
            replacement = _clip(new) or existing
            if preserve_key:
                key = _core_key(existing)
                if key:
                    replacement = f"**{key}**：{replacement}"
            items[idx] = replacement
            return


def _mark_status(items, target: str, suffix: str):
    """给匹配到的条目追加状态后缀；已标记过的不重复追加。"""
    target_norm = _norm(target)
    if len(target_norm) < 2:
        return
    for idx, existing in enumerate(items):
        if target_norm in _norm(existing):
            if "已完成" in existing or "已取消" in existing:
                return
            items[idx] = f"{existing}{suffix}"
            return


def _bigrams(text: str):
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _similar(a: str, b: str) -> bool:
    """判断两条记忆是否在说同一件事（容忍“学 Kubernetes”和“学完 Kubernetes”这类差异）。"""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    if len(na) < 4 or len(nb) < 4:
        return False
    ga, gb = _bigrams(na), _bigrams(nb)
    if not ga or not gb:
        return False
    return len(ga & gb) / min(len(ga), len(gb)) >= 0.6


def apply_lifecycle(store: Dict[str, Any], entries, bucket: str, date_str: str) -> Dict[str, Any]:
    """合并带状态的目标/事项条目。

    重建时各批次是并行抽取的，后一批看不到前一批建立的事项，
    因此模型只能输出“这件事在说什么 + 状态”，由这里按日期顺序做去重和状态落定。
    """
    if not isinstance(entries, list):
        return store

    opened = []
    changed = []
    for entry in entries[:MAX_ITEMS_PER_UPDATE]:
        if not isinstance(entry, dict):
            continue
        text = _clip(entry.get("text") or entry.get("item") or entry.get("content"))
        if not text:
            continue
        status = (entry.get("status") or "open").strip().lower()
        if status == "open":
            opened.append(text)
            continue
        item_date = _normalize_date(str(entry.get("date") or "")) or date_str
        changed.append((text, status, item_date))

    for text in opened:
        _append_unique(store[bucket], text)

    for text, status, item_date in changed:
        matched = False
        for idx, existing in enumerate(store[bucket]):
            if "已完成" in existing or "已取消" in existing:
                continue
            if _similar(existing, text):
                if status == "done":
                    store[bucket][idx] = f"{existing}（已完成 {item_date}）"
                else:
                    store[bucket][idx] = f"{existing}（已取消）"
                matched = True
                break
        if not matched:
            # 找不到此前建立过的条目时也要保留下这条事实，避免整条丢失
            label = f"（已完成 {item_date}）" if status == "done" else "（已取消）"
            _append_unique(store[bucket], f"{text}{label}")

    return store


def apply_diff(store: Dict[str, Any], diff: Dict[str, Any], date_str: str) -> Dict[str, Any]:
    """把模型输出的增量 diff 合并进结构化记忆。"""
    if not isinstance(diff, dict):
        return store

    # 1. 核心画像：按 key 覆盖，不重写其他条目
    core_updates = diff.get("core") or {}
    if isinstance(core_updates, dict):
        for key, value in list(core_updates.items())[:MAX_ITEMS_PER_UPDATE]:
            key = _clip(key)
            value = _clip(value)
            if not key or not value:
                continue
            new_item = f"**{key}**：{value}"
            replaced = False
            for idx, existing in enumerate(store["core"]):
                if _core_key(existing) == key:
                    store["core"][idx] = new_item
                    replaced = True
                    break
            if not replaced:
                _append_unique(store["core"], new_item)

    # 2. 目标 / 未完结事项 / 价值观
    for src_key, dst_key in [
        ("goals_add", "goals"),
        ("tasks_add", "open_tasks"),
        ("values_add", "values"),
    ]:
        for item in (diff.get(src_key) or [])[:MAX_ITEMS_PER_UPDATE]:
            _append_unique(store[dst_key], item)

    for item in (diff.get("goals_done") or [])[:MAX_ITEMS_PER_UPDATE]:
        _mark_status(store["goals"], item, f"（已完成 {date_str}）")

    for item in (diff.get("tasks_done") or [])[:MAX_ITEMS_PER_UPDATE]:
        _mark_status(store["open_tasks"], item, f"（已完成 {date_str}）")

    for item in (diff.get("tasks_cancel") or [])[:MAX_ITEMS_PER_UPDATE]:
        _mark_status(store["open_tasks"], item, "（已取消）")

    # 3. 事件：按日期分组去重
    for event in (diff.get("events") or [])[:MAX_ITEMS_PER_UPDATE]:
        if not isinstance(event, dict):
            continue
        event_date = _normalize_date(str(event.get("date") or "")) or date_str
        items = event.get("items") or []
        if isinstance(items, str):
            items = [items]
        target = None
        for block in store["events"]:
            if block.get("date") == event_date:
                target = block
                break
        if target is None:
            target = {"date": event_date, "items": []}
            store["events"].append(target)
        for item in items[:MAX_ITEMS_PER_UPDATE]:
            _append_unique(target["items"], item)

    # 4. 事实修正：在所有文本列表里做旧→新替换
    corrections = diff.get("corrections") or []
    for pair in corrections[:MAX_ITEMS_PER_UPDATE]:
        if not isinstance(pair, dict):
            continue
        old = _clip(pair.get("old"))
        new = _clip(pair.get("new"))
        if not old or not new:
            continue
        for key in ("core", "goals", "open_tasks", "values"):
            _replace_in_list(store[key], old, new, preserve_key=(key == "core"))
        for block in store["events"]:
            for idx, item in enumerate(block.get("items", [])):
                if _norm(old) and _norm(old) in _norm(item):
                    block["items"][idx] = new

    store["events"].sort(key=lambda e: (e.get("date") or "0000-00-00"), reverse=True)
    return store


# ==================== 渲染 ====================


def is_empty(store: Dict[str, Any]) -> bool:
    return not any(store.get(k) for k in ("core", "goals", "open_tasks", "values", "events"))


def render_markdown(store: Dict[str, Any]) -> str:
    if is_empty(store):
        return ""

    lines = ["# 长期记忆", ""]

    for key, heading in SECTION_ORDER:
        items = store.get(key) or []
        if not items:
            continue
        lines.append(f"## {heading}")
        if key == "events":
            lines.append("")
            for block in items:
                date_str = block.get("date") or ""
                block_items = block.get("items") or []
                if not block_items:
                    continue
                if date_str:
                    lines.append(f"- **{date_str}**：")
                    lines.extend(f"  - {item}" for item in block_items)
                else:
                    lines.extend(f"- {item}" for item in block_items)
        else:
            lines.extend(f"- {item}" for item in items)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def backup_and_clear(user_email: Optional[str] = None, stamp: Optional[str] = None) -> str:
    """备份并清空长期记忆，返回备份时间戳。

    先写渲染视图再写事实源，保证 memory.json 的 mtime 最新，
    避免下一次更新把空文件误判为“用户手工编辑过 memory.md”。
    """
    stamp = stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in (get_markdown_path(user_email), get_store_path(user_email)):
        if os.path.exists(path):
            try:
                shutil.copy2(path, f"{path}.bak-{stamp}")
            except Exception as e:
                print(f"[MemoryStore] 备份失败 {path}: {e}")

    save_markdown("", user_email)
    save_store(empty_store(), user_email)
    return stamp


def save_markdown(markdown_text: str, user_email: Optional[str] = None):
    _write_text_atomic(get_markdown_path(user_email), markdown_text)


# ==================== 旁路重建（暂存 + 原子切换） ====================


def get_staged_store_path(user_email: Optional[str] = None) -> str:
    return os.path.join(_get_user_base_dir(user_email), "memory.staged.json")


def get_staged_markdown_path(user_email: Optional[str] = None) -> str:
    return os.path.join(_get_user_base_dir(user_email), "memory.staged.md")


def save_staged(store: Dict[str, Any], user_email: Optional[str] = None):
    """重建结果先写暂存区，线上文件完全不动。"""
    # 先写 md 再写 json，保证切换后 memory.json 的 mtime 更新
    _write_text_atomic(get_staged_markdown_path(user_email), render_markdown(store))
    store["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_text_atomic(
        get_staged_store_path(user_email),
        json.dumps(store, ensure_ascii=False, indent=2),
    )


def discard_staged(user_email: Optional[str] = None):
    for path in (get_staged_store_path(user_email), get_staged_markdown_path(user_email)):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                print(f"[MemoryStore] 清理暂存文件失败 {path}: {e}")


def commit_staged(user_email: Optional[str] = None) -> str:
    """把暂存区原子切换为线上记忆，旧文件留时间戳备份。"""
    staged_store = get_staged_store_path(user_email)
    staged_markdown = get_staged_markdown_path(user_email)
    if not os.path.exists(staged_store):
        raise FileNotFoundError("暂存记忆不存在，无法切换")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for real, staged in (
        (get_markdown_path(user_email), staged_markdown),
        (get_store_path(user_email), staged_store),
    ):
        if os.path.exists(real):
            try:
                shutil.copy2(real, f"{real}.bak-{stamp}")
            except Exception as e:
                print(f"[MemoryStore] 切换前备份失败 {real}: {e}")
        if os.path.exists(staged):
            os.replace(staged, real)

    discard_staged(user_email)
    return stamp


# ==================== 对话文本 ====================


def format_user_messages(conversation) -> str:
    """只保留用户原话；AI 回复不进入记忆事实源。"""
    lines = []
    for msg in conversation or []:
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        time_str = msg.get("time")
        prefix = f"[{time_str}] " if time_str else ""
        if msg.get("is_article"):
            prefix += "（文章）"
        lines.append(f"{prefix}{content}")
    return "\n\n".join(lines)
