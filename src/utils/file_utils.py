import os
import datetime
import json
import re
from typing import Optional

# 获取项目根目录（main.py 所在目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))


def ensure_data_dir():
    """确保数据目录存在"""
    os.makedirs(DATA_DIR, exist_ok=True)


def _get_user_base_dir(user_email: Optional[str] = None) -> str:
    """获取用户数据基础目录"""
    if user_email:
        # 导入在这里避免循环依赖
        from src.utils.user_utils import get_user_data_dir
        user_dir = get_user_data_dir(user_email)
        if user_dir:
            return user_dir
    return DATA_DIR


def _get_diaries_dir(user_email: Optional[str] = None) -> str:
    """获取日记目录"""
    if user_email:
        from src.utils.user_utils import get_user_diaries_dir
        diaries_dir = get_user_diaries_dir(user_email)
        if diaries_dir:
            return diaries_dir
    return DATA_DIR


def _load_md_file(filename: str, user_email: Optional[str] = None) -> str:
    """通用：加载markdown文件内容"""
    base_dir = _get_user_base_dir(user_email)
    filepath = os.path.join(base_dir, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def load_profile(user_email: Optional[str] = None):
    """加载profile.md文件内容"""
    # 如果用户设置了自定义profile，使用用户的
    if user_email:
        from src.utils.user_utils import get_user_settings
        settings = get_user_settings(user_email)
        custom_profile = settings.get("profile_file")
        if custom_profile and os.path.exists(custom_profile):
            with open(custom_profile, "r", encoding="utf-8") as f:
                return f.read()
    # 否则使用系统默认profile
    return _load_md_file("profile.md")


def extract_agent_persona(user_email: Optional[str] = None) -> str:
    """从profile.md中提取角色设定部分"""
    content = load_profile(user_email)
    if not content:
        return ""
    
    # 查找 "## 角色设定" 部分
    match = re.search(r'## 角色设定\s*\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
    if match:
        persona = match.group(1).strip()
        return persona
    return ""


def extract_profile_without_persona(user_email: Optional[str] = None) -> str:
    """从profile.md中提取除角色设定外的其他内容"""
    content = load_profile(user_email)
    if not content:
        return ""
    
    # 移除角色设定部分，保留其他内容
    # 匹配从 "## 角色设定" 开始到下一个 "## " 之前的内容并移除
    cleaned = re.sub(r'## 角色设定\s*\n.*?(?=\n## |\Z)', '', content, flags=re.DOTALL)
    return cleaned.strip()


def load_memory(user_email: Optional[str] = None):
    """加载memory.md文件内容"""
    if user_email:
        from src.utils.user_utils import get_user_memory_path
        memory_path = get_user_memory_path(user_email)
        if memory_path and os.path.exists(memory_path):
            with open(memory_path, "r", encoding="utf-8") as f:
                return f.read()
    # 回退到默认路径
    return _load_md_file("memory.md")


def _extract_date_from_archive(filename: str) -> str:
    """从归档文件名提取日期"""
    match = re.search(r'diary_(\d{4}-\d{2}-\d{2})_\d+\.md', filename)
    if match:
        return match.group(1)
    return ""


def _extract_conversation_from_archive(content: str) -> str:
    """从归档文件内容提取用户对话记录（只保留用户消息，去掉MOSS回复）"""
    match = re.search(r'## 对话记录\s*\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
    if match:
        conversation = match.group(1).strip()
        # 只保留用户消息（以**用户**:开头的行）
        user_lines = []
        for line in conversation.split('\n'):
            if line.strip().startswith('**用户**:'):
                # 去掉前缀，只保留内容
                user_content = line.strip().replace('**用户**:', '').strip()
                user_lines.append(user_content)
        return '\n'.join(user_lines)
    return ""


def _sort_diary_files(filename: str):
    """日记文件排序键：按日期和索引排序，草稿文件排在最前"""
    # 草稿文件：diary_YYYY-MM-DD_draft.md
    draft_match = re.search(r'diary_(\d{4}-\d{2}-\d{2})_draft\.md', filename)
    if draft_match:
        date_str = draft_match.group(1)
        return (date_str, 999)  # 草稿用最大索引，确保排在当天最前
    
    # 归档文件：diary_YYYY-MM-DD_N.md
    archive_match = re.search(r'diary_(\d{4}-\d{2}-\d{2})_(\d+)\.md', filename)
    if archive_match:
        date_str, idx = archive_match.groups()
        return (date_str, int(idx))
    return ("", 0)


def load_recent_diaries(user_email: Optional[str] = None):
    """加载最近日记的对话记录（带日期标注）"""
    diaries_dir = _get_diaries_dir(user_email)
    today = datetime.date.today().isoformat()
    
    if not os.path.exists(diaries_dir):
        return []
    
    diary_files = [f for f in os.listdir(diaries_dir)
                   if f.startswith("diary_") and f.endswith('.md')]
    diary_files.sort(key=_sort_diary_files, reverse=True)

    conversations = []
    for filename in diary_files[:7]:
        filepath = os.path.join(diaries_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            date_str = _extract_date_from_archive(filename)
            conversation = _extract_conversation_from_archive(content)

            if conversation:
                # 如果是今天的对话，特殊标注
                if date_str == today:
                    header = f"【今天】"
                else:
                    header = f"【{date_str}】"
                conversations.append(f"{header}\n{conversation}")

    return conversations


def get_today_draft_file(user_email: Optional[str] = None):
    """获取今天的草稿文件路径"""
    diaries_dir = _get_diaries_dir(user_email)
    today = datetime.date.today().isoformat()
    draft_path = os.path.join(diaries_dir, f"diary_{today}_draft.md")
    return draft_path if os.path.exists(draft_path) else None


def append_to_draft(user_msg, assistant_reply, user_email: Optional[str] = None):
    """追加内容到今天的草稿文件"""
    diaries_dir = _get_diaries_dir(user_email)
    today = datetime.date.today().isoformat()
    draft_path = os.path.join(diaries_dir, f"diary_{today}_draft.md")
    
    # 确保目录存在
    os.makedirs(diaries_dir, exist_ok=True)
    
    # 如果草稿不存在，创建初始结构
    if not os.path.exists(draft_path):
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(f"# 日记 - {today}\n## 对话记录\n")
    
    now = datetime.datetime.now().strftime('%H:%M:%S')
    with open(draft_path, "a", encoding="utf-8") as f:
        f.write(f"\n**[{now}] 用户**: {user_msg}\n\n")
        f.write(f"**[{now}] Buddy**: {assistant_reply}\n\n")
    
    return os.path.basename(draft_path)


def _get_next_diary_index(date_str: str, user_email: Optional[str] = None) -> int:
    """获取指定日期下一个日记索引"""
    diaries_dir = _get_diaries_dir(user_email)
    if not os.path.exists(diaries_dir):
        return 1
    diary_files = [f for f in os.listdir(diaries_dir) 
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
            # 保留时间戳（如果有）
            time_str = msg.get("time", "")
            if msg["role"] == "user":
                # 用户内容：引用块 + emoji + 加粗，更醒目
                if time_str:
                    f.write(f"> 🙋 **[{time_str}] {role}**: **{msg['content']}**\n\n")
                else:
                    f.write(f"> 🙋 **{role}**: **{msg['content']}**\n\n")
            else:
                # Buddy内容：emoji + 普通格式
                if time_str:
                    f.write(f"🤖 **[{time_str}] {role}**: {msg['content']}\n\n")
                else:
                    f.write(f"🤖 **{role}**: {msg['content']}\n\n")
        f.write(f"## 日记文章\n{diary_article}\n\n---\n")


def finalize_diary(structured_data: dict, conversation: list, diary_article: str, 
                   draft_date: datetime.date = None, delete_draft: bool = False,
                   user_email: Optional[str] = None) -> str:
    """
    完成日记归档（通用函数）
    
    Args:
        structured_data: 结构化摘要数据
        conversation: 对话记录列表
        diary_article: AI生成的日记文章
        draft_date: 指定日期（None表示今天）
        delete_draft: 是否删除草稿文件
        user_email: 用户邮箱（可选，用于用户隔离）
    
    Returns:
        生成的文件名
    """
    diaries_dir = _get_diaries_dir(user_email)
    
    # 确定日期
    if draft_date:
        date_str = draft_date.isoformat()
    else:
        date_str = datetime.date.today().isoformat()
    
    # 确保目录存在
    os.makedirs(diaries_dir, exist_ok=True)
    
    # 获取索引并生成文件名
    idx = _get_next_diary_index(date_str, user_email)
    filename = f"diary_{date_str}_{idx}.md"
    filepath = os.path.join(diaries_dir, filename)
    
    # 写入文件
    _write_diary_file(filepath, date_str, idx, structured_data, conversation, diary_article)
    
    # 按需删除草稿
    if delete_draft:
        draft_path = get_today_draft_file(user_email)
        if draft_path:
            os.remove(draft_path)
    
    return filename


def _parse_draft_date(filename: str):
    """从草稿文件名解析日期"""
    match = re.search(r'diary_(\d{4}-\d{2}-\d{2})_draft\.md', filename)
    if match:
        return datetime.date.fromisoformat(match.group(1))
    return None


def _extract_conversation_from_draft(content: str) -> list:
    """从草稿内容提取对话记录（保留时间戳）"""
    conversation = []
    user_msg = None
    user_time = None
    for line in content.split('\n'):
        user_match = re.search(r'\*\*\[(\d{2}:\d{2}:\d{2})\] 用户\*\*: (.+)', line)
        buddy_match = re.search(r'\*\*\[(\d{2}:\d{2}:\d{2})\] Buddy\*\*: (.+)', line)
        if user_match:
            user_time = user_match.group(1)
            user_msg = user_match.group(2)
        elif buddy_match and user_msg:
            buddy_time = buddy_match.group(1)
            conversation.append({"role": "user", "content": user_msg, "time": user_time})
            conversation.append({"role": "assistant", "content": buddy_match.group(2), "time": buddy_time})
            user_msg = None
            user_time = None
    return conversation


def auto_archive_expired_drafts(archive_service, user_email: Optional[str] = None):
    """自动归档过期的草稿（非今天的）"""
    diaries_dir = _get_diaries_dir(user_email)
    today = datetime.date.today()
    archived = []
    
    if not os.path.exists(diaries_dir):
        return archived
    
    for filename in os.listdir(diaries_dir):
        if not filename.endswith('_draft.md'):
            continue
        draft_date = _parse_draft_date(filename)
        if not draft_date or draft_date >= today:
            continue
        
        # 读取并归档
        filepath = os.path.join(diaries_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        conversation = _extract_conversation_from_draft(content)
        result = archive_service.auto_archive_from_draft(conversation, draft_date, user_email)
        
        # 删除原草稿
        os.remove(filepath)
        archived.append({'old_draft': filename, 'new_file': result['filename']})
    
    return archived


def update_memory_file(new_memory, user_email: Optional[str] = None):
    """更新memory.md文件"""
    if user_email:
        from src.utils.user_utils import get_user_memory_path
        memory_path = get_user_memory_path(user_email)
        if memory_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(memory_path), exist_ok=True)
            with open(memory_path, "w", encoding="utf-8") as f:
                f.write(new_memory)
            return
    # 回退到默认路径
    memory_path = os.path.join(DATA_DIR, "memory.md")
    with open(memory_path, "w", encoding="utf-8") as f:
        f.write(new_memory)


def list_diary_files(user_email: Optional[str] = None):
    """获取已有的日记文件列表（包括系统文件）"""
    files = []
    
    # 添加系统文件（memory.md, profile.md）
    base_dir = _get_user_base_dir(user_email)
    system_files = ['memory.md', 'profile.md']
    for sys_file in system_files:
        sys_path = os.path.join(base_dir, sys_file)
        if os.path.exists(sys_path):
            files.append(sys_file)
    
    # 添加日记文件
    diaries_dir = _get_diaries_dir(user_email)
    if os.path.exists(diaries_dir):
        diary_files = [f for f in os.listdir(diaries_dir) if f.endswith('.md')]
        files.extend(diary_files)
    
    return sorted(files, reverse=True)


def get_diary_file_path(filename, user_email: Optional[str] = None):
    """获取日记文件路径"""
    # 系统文件（memory.md, profile.md）存放在用户根目录
    system_files = ['memory.md', 'profile.md']
    if filename in system_files:
        base_dir = _get_user_base_dir(user_email)
        return os.path.join(base_dir, filename)

    # 日记文件存放在 diaries 子目录
    diaries_dir = _get_diaries_dir(user_email)
    return os.path.join(diaries_dir, filename)