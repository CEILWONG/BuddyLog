import os
import datetime
import json
import re

DATA_DIR = "data"


def get_current_date():
    """获取当前东八区日期"""
    tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz).date()


def get_current_datetime():
    """获取当前东八区日期时间"""
    tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz)


def ensure_data_dir():
    """确保数据目录存在"""
    os.makedirs(DATA_DIR, exist_ok=True)


def load_profile():
    """加载profile.md文件内容"""
    profile_path = os.path.join(DATA_DIR, "profile.md")
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def load_memory():
    """加载memory.md文件内容"""
    memory_path = os.path.join(DATA_DIR, "memory.md")
    if os.path.exists(memory_path):
        with open(memory_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def load_recent_diaries():
    """加载最近5-10篇日记文件内容"""
    # 只加载已归档的日记（非draft文件）
    diary_files = [f for f in os.listdir(DATA_DIR) 
                   if f.startswith("diary_") 
                   and f.endswith('.md')
                   and '_draft' not in f
                   and f != "memory.md" 
                   and f != "profile.md"]
    # 按日期和索引排序，最新的在前
    diary_files.sort(reverse=True)
    recent_diaries = []
    for file in diary_files[:10]:  # 最多加载10篇
        filepath = os.path.join(DATA_DIR, file)
        with open(filepath, "r", encoding="utf-8") as f:
            recent_diaries.append(f.read())
    return recent_diaries


def get_today_draft_file():
    """获取今天的草稿文件路径，如果不存在则返回None"""
    today = get_current_date().isoformat()
    draft_filename = f"diary_{today}_draft.md"
    draft_path = os.path.join(DATA_DIR, draft_filename)
    if os.path.exists(draft_path):
        return draft_path
    return None


def create_draft_file():
    """创建今天的草稿文件"""
    today = get_current_date().isoformat()
    draft_filename = f"diary_{today}_draft.md"
    draft_path = os.path.join(DATA_DIR, draft_filename)
    
    # 写入初始结构
    with open(draft_path, "w", encoding="utf-8") as f:
        f.write(f"# 日记 - {today}\n")
        f.write(f"## 元数据\n")
        f.write(f"- 日期: {today}\n")
        f.write(f"- 状态: 进行中\n\n")
        
        f.write("## 对话记录\n")
    
    return draft_path


def append_to_draft(diary_entry, user_msg, assistant_reply):
    """追加内容到今天的草稿文件"""
    draft_path = get_today_draft_file()
    
    # 如果草稿文件不存在，创建它
    if not draft_path:
        draft_path = create_draft_file()
    
    now = get_current_datetime().strftime('%H:%M:%S')
    
    # 追加内容
    with open(draft_path, "a", encoding="utf-8") as f:
        # 记录对话
        f.write(f"\n**[{now}] 用户**: {user_msg}\n\n")
        f.write(f"**[{now}] Buddy**: {assistant_reply}\n\n")
        
        # 记录日记摘要
        if diary_entry:
            f.write(f"**[{now}] 日记摘要**:\n")
            f.write(f"{diary_entry}\n\n")
            f.write("---\n")
    
    return os.path.basename(draft_path)


def parse_draft_date(draft_filename):
    """从草稿文件名解析日期"""
    match = re.search(r'diary_(\d{4}-\d{2}-\d{2})_draft\.md', draft_filename)
    if match:
        date_str = match.group(1)
        return datetime.date.fromisoformat(date_str)
    return None


def get_all_draft_files():
    """获取所有草稿文件列表"""
    draft_files = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("diary_") and f.endswith("_draft.md"):
            draft_path = os.path.join(DATA_DIR, f)
            draft_date = parse_draft_date(f)
            if draft_date:
                draft_files.append({
                    'filename': f,
                    'path': draft_path,
                    'date': draft_date
                })
    return draft_files


def extract_conversation_from_draft(draft_content):
    """从草稿内容中提取对话记录"""
    conversation = []
    lines = draft_content.split('\n')
    
    current_user_msg = None
    current_assistant_msg = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 匹配用户消息: **[HH:MM:SS] 用户**: 内容
        user_match = re.search(r'\*\*\[\d{2}:\d{2}:\d{2}\] 用户\*\*: (.+)', line)
        if user_match:
            if current_user_msg and current_assistant_msg:
                conversation.append({"role": "user", "content": current_user_msg})
                conversation.append({"role": "assistant", "content": current_assistant_msg})
                current_user_msg = None
                current_assistant_msg = None
            current_user_msg = user_match.group(1)
            continue
        
        # 匹配Buddy消息: **[HH:MM:SS] Buddy**: 内容
        buddy_match = re.search(r'\*\*\[\d{2}:\d{2}:\d{2}\] Buddy\*\*: (.+)', line)
        if buddy_match:
            current_assistant_msg = buddy_match.group(1)
            continue
    
    # 添加最后一组对话
    if current_user_msg and current_assistant_msg:
        conversation.append({"role": "user", "content": current_user_msg})
        conversation.append({"role": "assistant", "content": current_assistant_msg})
    
    return conversation


def finalize_diary_for_date(structured_data, conversation, diary_article, date):
    """为指定日期完成归档（用于自动归档过期草稿）"""
    date_str = date.isoformat()
    
    # 计算当天的正式日记索引
    diary_files = [f for f in os.listdir(DATA_DIR) 
                   if f.startswith(f"diary_{date_str}") 
                   and f.endswith('.md')
                   and '_draft' not in f]
    idx = len(diary_files) + 1
    
    final_filename = f"diary_{date_str}_{idx}.md"
    final_path = os.path.join(DATA_DIR, final_filename)
    
    # 写入正式文件
    with open(final_path, "w", encoding="utf-8") as f:
        # 元数据
        f.write(f"# 日记 - {date_str}\n")
        f.write(f"## 元数据\n")
        f.write(f"- 日期: {date_str}\n")
        f.write(f"- 时间: {get_current_datetime().strftime('%H:%M:%S')}\n")
        f.write(f"- 索引: {idx}\n")
        f.write(f"- 状态: 已归档\n\n")
        
        # 结构化摘要
        f.write("## 结构化摘要\n")
        f.write("```json\n")
        f.write(json.dumps(structured_data, ensure_ascii=False, indent=2))
        f.write("\n```\n\n")
        
        # 精简对话流
        f.write("## 对话记录\n")
        for msg in conversation:
            role = "用户" if msg["role"] == "user" else "Buddy"
            f.write(f"**{role}**: {msg['content']}\n\n")
        
        # AI 生成的完整日记文章
        f.write("## 日记文章\n")
        f.write(f"{diary_article}\n\n")
        
        # 结束标记
        f.write("---\n")
    
    return final_filename


def finalize_diary(structured_data, conversation, diary_article):
    """完成今日记录，将草稿文件转为正式归档文件"""
    today = get_current_date().isoformat()
    draft_path = get_today_draft_file()
    
    # 计算当天的正式日记索引
    diary_files = [f for f in os.listdir(DATA_DIR) 
                   if f.startswith(f"diary_{today}") 
                   and f.endswith('.md')
                   and '_draft' not in f]
    idx = len(diary_files) + 1
    
    final_filename = f"diary_{today}_{idx}.md"
    final_path = os.path.join(DATA_DIR, final_filename)
    
    # 写入正式文件
    with open(final_path, "w", encoding="utf-8") as f:
        # 元数据
        f.write(f"# 日记 - {today}\n")
        f.write(f"## 元数据\n")
        f.write(f"- 日期: {today}\n")
        f.write(f"- 时间: {get_current_datetime().strftime('%H:%M:%S')}\n")
        f.write(f"- 索引: {idx}\n")
        f.write(f"- 状态: 已归档\n\n")
        
        # 结构化摘要
        f.write("## 结构化摘要\n")
        f.write("```json\n")
        f.write(json.dumps(structured_data, ensure_ascii=False, indent=2))
        f.write("\n```\n\n")
        
        # 精简对话流
        f.write("## 对话记录\n")
        for msg in conversation:
            role = "用户" if msg["role"] == "user" else "Buddy"
            f.write(f"**{role}**: {msg['content']}\n\n")
        
        # AI 生成的完整日记文章
        f.write("## 日记文章\n")
        f.write(f"{diary_article}\n\n")
        
        # 结束标记
        f.write("---\n")
    
    # 删除草稿文件
    if draft_path and os.path.exists(draft_path):
        os.remove(draft_path)
    
    return final_filename


def auto_archive_expired_drafts(archive_service):
    """自动归档过期的草稿文件（非今天的草稿）"""
    today = get_current_date()
    drafts = get_all_draft_files()
    
    archived_files = []
    for draft in drafts:
        draft_date = draft['date']
        # 如果草稿日期早于今天，则自动归档
        if draft_date < today:
            try:
                # 读取草稿内容
                with open(draft['path'], "r", encoding="utf-8") as f:
                    draft_content = f.read()
                
                # 从草稿中提取对话记录
                conversation = extract_conversation_from_draft(draft_content)
                
                # 使用 archive_service 进行归档
                result = archive_service.auto_archive_from_draft(conversation, draft_date)
                
                # 删除原草稿文件
                os.remove(draft['path'])
                
                archived_files.append({
                    'old_draft': draft['filename'],
                    'new_file': result['filename']
                })
                print(f"Auto-archived expired draft: {draft['filename']} -> {result['filename']}")
            except Exception as e:
                print(f"Failed to auto-archive draft {draft['filename']}: {str(e)}")
    
    return archived_files


def update_memory_file(new_memory):
    """更新memory.md文件"""
    memory_path = os.path.join(DATA_DIR, "memory.md")
    with open(memory_path, "w", encoding="utf-8") as f:
        f.write(new_memory)


def list_diary_files():
    """获取已有的日记文件列表（包含草稿、memory.md、profile.md）"""
    files = sorted([f for f in os.listdir(DATA_DIR)
                    if f.endswith('.md')], reverse=True)
    return files


def get_diary_file_path(filename):
    """获取日记文件路径"""
    return os.path.join(DATA_DIR, filename)