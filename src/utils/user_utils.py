import os
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from src.services.auth_service import get_password_hash

# 获取项目根目录（main.py 所在目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
USER_INDEX_FILE = os.path.join(DATA_DIR, "user_index.json")
USERS_DIR = os.path.join(DATA_DIR, "users")

# 每日最大对话次数限制（默认值20）
# 优先级：用户设置 > 环境变量 > 默认值20
DEFAULT_DAILY_MAX_CONVERSATIONS = 20

# 默认长期记忆模板
DEFAULT_MEMORY_TEMPLATE = """# 用户长期记忆

## 用户核心画像
- 姓名/昵称：
- 职业：
- 当前状态：
- 兴趣爱好：

## 长期目标

## 未完结事项

## 价值观

## 重要事件
"""


def _ensure_users_dir():
    """确保用户目录存在"""
    os.makedirs(USERS_DIR, exist_ok=True)


def _load_user_index() -> Dict[str, Any]:
    """加载用户索引"""
    if os.path.exists(USER_INDEX_FILE):
        with open(USER_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_user_index(index: Dict[str, Any]):
    """保存用户索引"""
    _ensure_users_dir()
    with open(USER_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _generate_user_id() -> str:
    """生成用户ID"""
    return f"u_{uuid.uuid4().hex[:8]}"


def _get_user_dir(user_id: str) -> str:
    """获取用户数据目录路径"""
    return os.path.join(USERS_DIR, user_id)


def _create_user_data_dir(user_id: str):
    """创建用户数据目录及初始文件"""
    user_dir = _get_user_dir(user_id)
    diaries_dir = os.path.join(user_dir, "diaries")
    
    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(diaries_dir, exist_ok=True)
    
    # 创建空的长期记忆文件
    memory_path = os.path.join(user_dir, "memory.md")
    if not os.path.exists(memory_path):
        with open(memory_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_MEMORY_TEMPLATE)


def create_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    创建新用户
    
    Args:
        email: 用户邮箱
        password: 明文密码
        
    Returns:
        用户信息字典，如果邮箱已存在返回 None
    """
    index = _load_user_index()
    
    # 检查邮箱是否已存在
    if email in index:
        return None
    
    # 生成用户ID和创建时间
    user_id = _generate_user_id()
    created_at = datetime.now().isoformat()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 创建用户信息
    user_info = {
        "user_id": user_id,
        "password_hash": get_password_hash(password),
        "created_at": created_at,
        "settings": {
            "daily_max_conversations": None,
            "selected_model": None,
            "api_key": None,
            "profile_file": None
        },
        "usage": {
            "total_conversations": 0,
            "total_tokens": 0,
            "today_conversations": 0,
            "today_tokens": 0,
            "last_reset_date": today
        }
    }
    
    # 保存到索引
    index[email] = user_info
    _save_user_index(index)
    
    # 创建用户数据目录
    _create_user_data_dir(user_id)
    
    return user_info


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """通过邮箱获取用户信息"""
    index = _load_user_index()
    user_info = index.get(email)
    if user_info:
        user_info["email"] = email
    return user_info


def get_user_id_by_email(email: str) -> Optional[str]:
    """通过邮箱获取用户ID"""
    user_info = get_user_by_email(email)
    return user_info.get("user_id") if user_info else None


def update_user_settings(email: str, settings: Dict[str, Any]) -> bool:
    """更新用户设置"""
    index = _load_user_index()
    if email not in index:
        return False
    
    # 只更新非None的字段
    current_settings = index[email].get("settings", {})
    for key, value in settings.items():
        if value is not None:
            current_settings[key] = value
    
    index[email]["settings"] = current_settings
    _save_user_index(index)
    return True


def update_user_usage(email: str, conversations_increment: int = 0, tokens_increment: int = 0) -> bool:
    """
    更新用户使用统计
    
    Args:
        email: 用户邮箱
        conversations_increment: 对话次数增量
        tokens_increment: token使用量增量
    """
    index = _load_user_index()
    if email not in index:
        return False
    
    today = datetime.now().strftime("%Y-%m-%d")
    usage = index[email].get("usage", {})
    
    # 检查是否需要重置今日统计
    if usage.get("last_reset_date") != today:
        usage["today_conversations"] = 0
        usage["today_tokens"] = 0
    
    # 更新日期标记
    usage["last_reset_date"] = today
    
    # 更新累计统计
    usage["total_conversations"] = usage.get("total_conversations", 0) + conversations_increment
    usage["total_tokens"] = usage.get("total_tokens", 0) + tokens_increment
    
    # 更新今日统计
    usage["today_conversations"] = usage.get("today_conversations", 0) + conversations_increment
    usage["today_tokens"] = usage.get("today_tokens", 0) + tokens_increment
    
    index[email]["usage"] = usage
    _save_user_index(index)
    return True


def check_user_limit(email: str) -> Dict[str, Any]:
    """
    检查用户是否超出使用限制
    
    优先级：用户设置 > 环境变量 > 默认值20
    
    Returns:
        {
            "allowed": bool,
            "reason": str,  # 如果不允许，说明原因
            "current": int,
            "limit": int
        }
    """
    user_info = get_user_by_email(email)
    if not user_info:
        return {"allowed": False, "reason": "用户不存在", "current": 0, "limit": 0}
    
    settings = user_info.get("settings", {})
    usage = user_info.get("usage", {})
    
    # 获取最大对话次数限制（优先级：用户设置 > 环境变量 > 默认值20）
    user_limit = settings.get("daily_max_conversations")
    env_limit = os.getenv("DAILY_MAX_CONVERSATIONS")
    
    if user_limit is not None:
        max_conversations = int(user_limit)
    elif env_limit is not None:
        max_conversations = int(env_limit)
    else:
        max_conversations = DEFAULT_DAILY_MAX_CONVERSATIONS
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 检查是否需要重置
    if usage.get("last_reset_date") != today:
        return {"allowed": True, "reason": "", "current": 0, "limit": max_conversations}
    
    current = usage.get("today_conversations", 0)
    
    if current >= max_conversations:
        return {
            "allowed": False,
            "reason": f"今日对话次数已达上限 ({max_conversations}次)",
            "current": current,
            "limit": max_conversations
        }
    
    return {
        "allowed": True,
        "reason": "",
        "current": current,
        "limit": max_conversations
    }


def get_effective_daily_limit(email: str) -> int:
    """
    获取用户实际生效的每日对话限制
    
    优先级：用户设置 > 环境变量 > 默认值20
    
    Returns:
        实际生效的限制值
    """
    user_info = get_user_by_email(email)
    if not user_info:
        return DEFAULT_DAILY_MAX_CONVERSATIONS
    
    settings = user_info.get("settings", {})
    user_limit = settings.get("daily_max_conversations")
    env_limit = os.getenv("DAILY_MAX_CONVERSATIONS")
    
    if user_limit is not None:
        return int(user_limit)
    elif env_limit is not None:
        return int(env_limit)
    else:
        return DEFAULT_DAILY_MAX_CONVERSATIONS


def get_user_data_dir(email: str) -> Optional[str]:
    """获取用户数据目录"""
    user_id = get_user_id_by_email(email)
    if not user_id:
        return None
    return _get_user_dir(user_id)


def get_user_memory_path(email: str) -> Optional[str]:
    """获取用户长期记忆文件路径"""
    user_dir = get_user_data_dir(email)
    if not user_dir:
        return None
    return os.path.join(user_dir, "memory.md")


def get_user_diaries_dir(email: str) -> Optional[str]:
    """获取用户日记目录路径"""
    user_dir = get_user_data_dir(email)
    if not user_dir:
        return None
    return os.path.join(user_dir, "diaries")


def get_user_settings(email: str) -> Dict[str, Any]:
    """获取用户设置"""
    user_info = get_user_by_email(email)
    if not user_info:
        return {}
    return user_info.get("settings", {})
