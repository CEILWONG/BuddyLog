import hashlib
import os

# 使用简单的 SHA256 + salt 方案，避免 bcrypt 版本兼容问题

def _get_salt() -> str:
    """获取环境变量中的 salt，如果没有则使用默认值"""
    return os.getenv("PASSWORD_SALT", "buddylog-default-salt-change-in-production")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return get_password_hash(plain_password) == hashed_password


def get_password_hash(password: str) -> str:
    """获取密码哈希 (SHA256 + salt)"""
    salt = _get_salt()
    # 组合密码和 salt，进行多次哈希增加安全性
    hashed = password + salt
    for _ in range(1000):  # 迭代1000次
        hashed = hashlib.sha256(hashed.encode()).hexdigest()
    return hashed
