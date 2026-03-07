import os
import datetime
import json
import re

DATA_DIR = "data"


def ensure_data_dir():
    """确保数据目录存在"""
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_md_file(filename: str) -> str:
    """通用：加载markdown文件内容"""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def load_profile():
    """加载profile.md文件内容"""
    return _load_md_file("profile.md")


def load_memory():
    """加载memory.md文件内容"""
    return _load_md_file("memory.md")


def load_recent_diaries():
    """加载最近日记文件内容（包含草稿）"""
    diary_files = [f for f in os.listdir(DATA_DIR) if f.startswith("diary_") and f.endswith('.md')]
    diary_files.sort(reverse=True)
    return [_load_md_file(f) for f in diary_files[:7]]


def get_today_draft_file():
    """获取今天的草稿文件路径"""
    today = datetime.date.today().isoformat()
    draft_path = os.path.join(DATA_DIR, f"diary_{today}_draft.md")
    return draft_path if os.path.exists(draft_path) else None


def append_to_draft(diary_entry, user_msg, assistant_reply):
    """追加内容到今天的草稿文件"""
    today = datetime.date.today().isoformat()
    draft_path = os.path.join(DATA_DIR, f"diary_{today}_draft.md")
    
    # 如果草稿不存在，创建初始结构
    if not os.path.exists(draft_path):
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(f"# 日记 - {today}\n## 对话记录\n")
    
    now = datetime.datetime.now().strftime('%H:%M:%S')
    with open(draft_path, "a", encoding="utf-8") as f:
        f.write(f"\n**[{now}] 用户**: {user_msg}\n\n")
        f.write(f"**[{now}] Buddy**: {assistant_reply}\n\n")
        if diary_entry:
            f.write(f"**[{now}] 日记摘要**: {diary_entry}\n\n---\n")
    
    return os.path.basename(draft_path)


def _get_next_diary_index(date_str: str) -> int:
    """获取指定日期下一个日记索引"""
    diary_files = [f for f in os.listdir(DATA_DIR) 
                   if f.startswith(f"diary_{date_str}") and f.endswith('.md') and '_draft' not in f]
    return len(diary_files) + 1


def _write_diary_file(filepath: str, date_str: str, idx: int, structured_data: dict, 
                      conversation: list, diary_article: str):
    """通用：写入日记文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 日记 - {date_str}\n")
        f.write(f"## 元数据\n- 日期: {date_str}\n- 时间: {datetime.datetime.now().strftime('%H:%M:%S')}\n- 索引: {idx}\n\n")
        f.write("## 结构化摘要\n```json\n")
        f.write(json.dumps(structured_data, ensure_ascii=False, indent=2))
        f.write("\n```\n\n## 对话记录\n")
        for msg in conversation:
            role = "用户" if msg["role"] == "user" else "Buddy"
            f.write(f"**{role}**: {msg['content']}\n\n")
        f.write(f"## 日记文章\n{diary_article}\n\n---\n")


def finalize_diary(structured_data: dict, conversation: list, diary_article: str) -> str:
    """完成今日归档（删除草稿）"""
    today = datetime.date.today().isoformat()
    idx = _get_next_diary_index(today)
    filename = f"diary_{today}_{idx}.md"
    filepath = os.path.join(DATA_DIR, filename)
    
    _write_diary_file(filepath, today, idx, structured_data, conversation, diary_article)
    
    # 删除草稿
    draft_path = get_today_draft_file()
    if draft_path:
        os.remove(draft_path)
    
    return filename


def finalize_diary_for_date(structured_data: dict, conversation: list, diary_article: str, draft_date) -> str:
    """为指定日期归档（用于自动归档）"""
    date_str = draft_date.isoformat()
    idx = _get_next_diary_index(date_str)
    filename = f"diary_{date_str}_{idx}.md"
    filepath = os.path.join(DATA_DIR, filename)
    
    _write_diary_file(filepath, date_str, idx, structured_data, conversation, diary_article)
    return filename


def _parse_draft_date(filename: str):
    """从草稿文件名解析日期"""
    match = re.search(r'diary_(\d{4}-\d{2}-\d{2})_draft\.md', filename)
    if match:
        return datetime.date.fromisoformat(match.group(1))
    return None


def _extract_conversation_from_draft(content: str) -> list:
    """从草稿内容提取对话记录"""
    conversation = []
    user_msg = None
    for line in content.split('\n'):
        user_match = re.search(r'\*\*\[\d{2}:\d{2}:\d{2}\] 用户\*\*: (.+)', line)
        buddy_match = re.search(r'\*\*\[\d{2}:\d{2}:\d{2}\] Buddy\*\*: (.+)', line)
        if user_match:
            user_msg = user_match.group(1)
        elif buddy_match and user_msg:
            conversation.append({"role": "user", "content": user_msg})
            conversation.append({"role": "assistant", "content": buddy_match.group(1)})
            user_msg = None
    return conversation


def auto_archive_expired_drafts(archive_service):
    """自动归档过期的草稿（非今天的）"""
    today = datetime.date.today()
    archived = []
    
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith('_draft.md'):
            continue
        draft_date = _parse_draft_date(filename)
        if not draft_date or draft_date >= today:
            continue
        
        # 读取并归档
        content = _load_md_file(filename)
        conversation = _extract_conversation_from_draft(content)
        result = archive_service.auto_archive_from_draft(conversation, draft_date)
        
        # 删除原草稿
        os.remove(os.path.join(DATA_DIR, filename))
        archived.append({'old_draft': filename, 'new_file': result['filename']})
    
    return archived


def update_memory_file(new_memory):
    """更新memory.md文件"""
    memory_path = os.path.join(DATA_DIR, "memory.md")
    with open(memory_path, "w", encoding="utf-8") as f:
        f.write(new_memory)


def list_diary_files():
    """获取已有的日记文件列表"""
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.md')], reverse=True)
    return files


def get_diary_file_path(filename):
    """获取日记文件路径"""
    return os.path.join(DATA_DIR, filename)