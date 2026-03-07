import os
import datetime
import json

DATA_DIR = "data"


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
    diary_files = [f for f in os.listdir(DATA_DIR) if f.startswith("diary_") and f.endswith('.md')]
    # 按日期和索引排序，最新的在前
    diary_files.sort(reverse=True)
    recent_diaries = []
    for file in diary_files[:10]:  # 最多加载10篇
        filepath = os.path.join(DATA_DIR, file)
        with open(filepath, "r", encoding="utf-8") as f:
            recent_diaries.append(f.read())
    return recent_diaries


def save_diary(content, history, reply_text):
    """保存日记到文件"""
    today = datetime.date.today().isoformat()
    
    # 计算当天的日记索引
    diary_files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"diary_{today}") and f.endswith('.md')]
    idx = len(diary_files) + 1
    
    filename = f"diary_{today}_{idx}.md"
    filepath = os.path.join(DATA_DIR, filename)

    # 生成结构化摘要
    structured_summary = {
        "events": [],
        "people": [],
        "emotions": [],
        "ideas": []
    }

    # 写入标准格式的日记文件
    with open(filepath, "w", encoding="utf-8") as f:
        # 元数据
        f.write(f"# 日记 - {today}\n")
        f.write(f"## 元数据\n")
        f.write(f"- 日期: {today}\n")
        f.write(f"- 时间: {datetime.datetime.now().strftime('%H:%M:%S')}\n")
        f.write(f"- 索引: {idx}\n\n")
        
        # 结构化摘要
        f.write("## 结构化摘要\n")
        f.write("```json\n")
        f.write(json.dumps(structured_summary, ensure_ascii=False, indent=2))
        f.write("\n```\n\n")
        
        # 精简对话流
        f.write("## 对话记录\n")
        for msg in history:
            role = "用户" if msg["role"] == "user" else "Buddy"
            f.write(f"**{role}**: {msg['content']}\n\n")
        
        # AI 生成的完整日记文章
        f.write("## 日记文章\n")
        f.write(f"{content}\n\n")
        
        # 结束标记
        f.write("---\n")
    
    return filename


def save_archived_diary(structured_data, conversation, diary_article):
    """保存归档日记到文件"""
    today = datetime.date.today().isoformat()
    diary_files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"diary_{today}") and f.endswith('.md')]
    idx = len(diary_files) + 1
    
    filename = f"diary_{today}_{idx}.md"
    filepath = os.path.join(DATA_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        # 元数据
        f.write(f"# 日记 - {today}\n")
        f.write(f"## 元数据\n")
        f.write(f"- 日期: {today}\n")
        f.write(f"- 时间: {datetime.datetime.now().strftime('%H:%M:%S')}\n")
        f.write(f"- 索引: {idx}\n\n")
        
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
    
    return filename


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