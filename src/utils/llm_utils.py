"""LLM 接入层：统一解析每次调用实际使用的客户端与模型

设计要点：
- 系统默认客户端由环境变量 OPENAI_API_KEY / DASHSCOPE_API_KEY + OPENAI_BASE_URL 构建；
- 用户可在设置中填入自己的 api_key 与 selected_model，调用时按 user_email 动态生效，
  未配置则自然回落到系统默认，不影响存量用户；
- OpenAI 客户端内部持有连接池，按 api_key 缓存复用，避免每次调用重建；
- base_url 仍由部署方统一配置，因此用户自带的 Key 必须属于同一兼容端点。
"""

import os
import threading
from typing import NamedTuple, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# 保证任意入口导入本模块时环境变量都已加载（load_dotenv 可重复调用）
load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
SYSTEM_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "qwen-plus")

# 保存 Key 时是否做一次真实校验；个别兼容端点不支持极短补全时可置为 false
VERIFY_KEY_ON_SAVE = os.getenv("LLM_KEY_VERIFY", "true").lower() == "true"

# 系统默认客户端
_system_client = OpenAI(api_key=SYSTEM_API_KEY, base_url=OPENAI_BASE_URL)

# api_key -> OpenAI 客户端
_client_cache = {}
_cache_lock = threading.Lock()
_MAX_CACHED_CLIENTS = 64


class LLMContext(NamedTuple):
    """一次 LLM 调用实际使用的客户端与模型"""
    client: OpenAI
    model: str
    is_custom_key: bool


def get_system_client() -> OpenAI:
    """系统默认客户端（环境变量中的 Key）"""
    return _system_client


def get_client_for_key(api_key: Optional[str]) -> OpenAI:
    """按 api_key 取客户端，空值回落到系统默认客户端"""
    key = (api_key or "").strip()
    if not key:
        return _system_client
    with _cache_lock:
        client = _client_cache.get(key)
        if client is None:
            # 容量兜底：极端情况下整体重建，避免缓存无限增长
            if len(_client_cache) >= _MAX_CACHED_CLIENTS:
                _client_cache.clear()
            client = OpenAI(api_key=key, base_url=OPENAI_BASE_URL)
            _client_cache[key] = client
        return client


def invalidate_client_cache(api_key: Optional[str] = None):
    """Key 变更后失效客户端缓存；不传参数则全部清空"""
    with _cache_lock:
        if api_key is None:
            _client_cache.clear()
        else:
            _client_cache.pop((api_key or "").strip(), None)


def mask_api_key(api_key: Optional[str]) -> str:
    """脱敏展示：仅保留首尾各 4 位，明文不出后端"""
    key = (api_key or "").strip()
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}****{key[-4:]}"


def resolve_llm(user_email: Optional[str] = None) -> LLMContext:
    """解析某次调用生效的客户端与模型

    优先级：用户设置的 api_key > 系统默认（环境变量）。
    自选模型遵循「模型跟随 Key」策略：仅当用户配置了自己的 api_key 时
    selected_model 才生效；无 Key 一律回落系统默认模型，杜绝用户不花自己的钱
    却用系统 Key 跑贵模型。用户不存在或未配置时回落系统默认，不抛异常。
    """
    if not user_email:
        return LLMContext(_system_client, DEFAULT_MODEL, False)

    # 延迟导入：user_utils 依赖 auth_service，避免模块级循环导入
    from src.utils.user_utils import get_user_settings

    settings = get_user_settings(user_email) or {}
    api_key = (settings.get("api_key") or "").strip()
    # 模型跟随 Key：无自定义 Key 时忽略 selected_model
    model = (settings.get("selected_model") or "").strip() if api_key else ""
    return LLMContext(get_client_for_key(api_key), model or DEFAULT_MODEL, bool(api_key))


def describe_llm_error(err: Exception, ctx: LLMContext) -> str:
    """把底层异常翻译成对使用者有意义的提示"""
    if ctx.is_custom_key:
        return f"自定义 API Key 调用失败（模型 {ctx.model}），请检查 Key 是否有效、余额是否充足、模型名是否正确：{err}"
    return f"API request failed: {err}"


def verify_llm_credentials(api_key: Optional[str], model: Optional[str]) -> Tuple[bool, str]:
    """用一次最小补全校验 (api_key, model) 组合是否真实可用

    Returns:
        (是否通过, 失败原因)
    """
    if not VERIFY_KEY_ON_SAVE:
        return True, ""
    target_model = (model or "").strip() or DEFAULT_MODEL
    client = get_client_for_key(api_key)
    try:
        client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return True, ""
    except Exception as e:
        return False, f"模型 {target_model} 校验失败：{e}"
