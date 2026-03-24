from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any
from datetime import datetime


class UserRegister(BaseModel):
    """用户注册请求"""
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=50)


class UserLogin(BaseModel):
    """用户登录请求"""
    email: EmailStr
    password: str


class UserSettings(BaseModel):
    """用户设置"""
    daily_max_conversations: Optional[int] = None  # 每天最大对话次数，null表示无限制
    selected_model: Optional[str] = None  # 选用的模型，null使用系统默认
    api_key: Optional[str] = None  # 用户自己的API Key，null使用系统默认
    profile_file: Optional[str] = None  # 自定义profile文件路径，null使用系统默认


class UserUsage(BaseModel):
    """用户使用统计"""
    total_conversations: int = 0
    total_tokens: int = 0
    today_conversations: int = 0
    today_tokens: int = 0
    last_reset_date: str  # YYYY-MM-DD


class UserInfo(BaseModel):
    """用户信息（存储在user_index.json中的结构）"""
    user_id: str
    password_hash: str
    created_at: str
    settings: UserSettings = Field(default_factory=UserSettings)
    usage: UserUsage


class TokenResponse(BaseModel):
    """登录/注册成功响应"""
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str


class UserProfileResponse(BaseModel):
    """获取用户信息响应"""
    email: str
    user_id: str
    created_at: str
    settings: UserSettings
    usage: UserUsage


class SettingsUpdateRequest(BaseModel):
    """更新设置请求"""
    daily_max_conversations: Optional[int] = None
    selected_model: Optional[str] = None
    api_key: Optional[str] = None
    profile_file: Optional[str] = None
