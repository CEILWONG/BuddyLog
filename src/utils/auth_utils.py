from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

# JWT 配置
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

# 内置管理员账户（不写入用户数据，仅代码/环境变量定义）
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "wzhH1234")

# 角色常量
ROLE_USER = "user"
ROLE_ADMIN = "admin"

# 安全方案
security = HTTPBearer()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """创建 JWT Token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[str]:
    """验证 JWT Token，返回用户邮箱"""
    payload = get_token_payload(token)
    if payload is None:
        return None
    return payload.get("sub")


def get_token_payload(token: str) -> Optional[dict]:
    """验证 JWT Token，返回完整 payload；无效时返回 None"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") is None:
            return None
        return payload
    except JWTError:
        return None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """获取当前登录用户邮箱（作为依赖使用）"""
    token = credentials.credentials
    email = verify_token(token)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return email


async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """获取当前管理员邮箱（校验 role == admin，否则 403）"""
    token = credentials.credentials
    payload = get_token_payload(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if payload.get("role") != ROLE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限",
        )
    return payload.get("sub")


async def get_optional_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """可选的认证，用于兼容旧接口"""
    try:
        token = credentials.credentials
        return verify_token(token)
    except:
        return None
