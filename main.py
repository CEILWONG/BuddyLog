import os
import io
import zipfile
from datetime import datetime, date
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from dotenv import load_dotenv
import dashscope

# 加载配置
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
MODEL = os.getenv("MODEL_NAME", "qwen-plus")

# IP 注册限制缓存：{ip: {日期: 次数}}
ip_register_cache = {}
MAX_REGISTERS_PER_IP_PER_DAY = 50  # 每个 IP 每天最多注册次数

# 导入模块
from src.models.schemas import Message, ArchiveRequest
from src.models.auth_schemas import (
    UserRegister, UserLogin, TokenResponse, UserProfileResponse,
    SettingsUpdateRequest
)
from src.services.chat_service import ChatService
from src.services.archive_service import ArchiveService
from src.services.auth_service import verify_password
from src.utils.file_utils import ensure_data_dir, list_diary_files, get_diary_file_path
from src.utils.auth_utils import create_access_token, get_current_user
from src.utils.user_utils import (
    create_user, get_user_by_email, update_user_settings,
    update_user_usage, check_user_limit, get_user_settings,
    get_effective_daily_limit,
)

# 确保数据目录存在
ensure_data_dir()

app = FastAPI(title="BuddyLog API")

# 允许跨域 (为了本地直接打开 html 能访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
archive_service = ArchiveService(MODEL)
chat_service = ChatService(MODEL, archive_service)


# ==================== 认证路由 ====================

@app.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserRegister, request: Request):
    """用户注册"""
    # 获取客户端 IP
    client_ip = request.client.host
    
    # 检查该 IP 今天是否已达到注册上限
    today = date.today().isoformat()
    ip_record = ip_register_cache.get(client_ip, {})
    register_count = ip_record.get(today, 0)
    if register_count >= MAX_REGISTERS_PER_IP_PER_DAY:
        raise HTTPException(status_code=429, detail=f"每个 IP 每天最多只能注册 {MAX_REGISTERS_PER_IP_PER_DAY} 个账号")
    
    # 检查邮箱是否已存在
    existing_user = get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="该邮箱已被注册")
    
    # 创建用户
    user_info = create_user(user_data.email, user_data.password)
    if not user_info:
        raise HTTPException(status_code=500, detail="创建用户失败")
    
    # 增加该 IP 今天的注册计数
    if client_ip not in ip_register_cache:
        ip_register_cache[client_ip] = {}
    ip_register_cache[client_ip][today] = register_count + 1
    
    # 生成Token
    access_token = create_access_token(data={"sub": user_data.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_info["user_id"],
        "email": user_data.email
    }


@app.post("/auth/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    """用户登录"""
    # 获取用户信息
    user_info = get_user_by_email(user_data.email)
    if not user_info:
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    
    # 验证密码
    if not verify_password(user_data.password, user_info["password_hash"]):
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    
    # 生成Token
    access_token = create_access_token(data={"sub": user_data.email})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_info["user_id"],
        "email": user_data.email
    }


@app.get("/auth/me", response_model=UserProfileResponse)
async def get_me(current_email: str = Depends(get_current_user)):
    """获取当前用户信息"""
    user_info = get_user_by_email(current_email)
    if not user_info:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    # 获取实际生效的每日限制
    effective_limit = get_effective_daily_limit(current_email)
    
    return {
        "email": current_email,
        "user_id": user_info["user_id"],
        "created_at": user_info["created_at"],
        "settings": user_info.get("settings", {}),
        "usage": user_info.get("usage", {}),
        "effective_daily_limit": effective_limit
    }


@app.get("/auth/settings")
async def get_settings(current_email: str = Depends(get_current_user)):
    """获取用户设置"""
    settings = get_user_settings(current_email)
    return settings


@app.put("/auth/settings")
async def update_settings(
    settings_update: SettingsUpdateRequest,
    current_email: str = Depends(get_current_user)
):
    """更新用户设置"""
    # 转换为字典，排除None值
    settings_dict = settings_update.model_dump(exclude_unset=True)
    
    success = update_user_settings(current_email, settings_dict)
    if not success:
        raise HTTPException(status_code=500, detail="更新设置失败")
    
    return {"message": "设置已更新"}


# ==================== 原有接口（增加用户认证） ====================

@app.get("/")
async def read_root():
    """根路径，返回前端页面"""
    # 使用绝对路径读取index.html文件
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/chat")
async def chat(msg: Message, current_email: str = Depends(get_current_user)):
    """聊天接口"""
    try:
        # 检查用户限制
        limit_check = check_user_limit(current_email)
        if not limit_check["allowed"]:
            raise HTTPException(status_code=429, detail=limit_check["reason"])
        
        # 处理聊天
        result = chat_service.process_chat(msg.content, msg.history, current_email)
        
        # 更新使用统计
        tokens_data = result.get("tokens", {})
        tokens_total = tokens_data.get("total", 0) if isinstance(tokens_data, dict) else tokens_data
        update_user_usage(current_email, conversations_increment=1, tokens_increment=tokens_total)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
async def list_files(current_email: str = Depends(get_current_user)):
    """获取已有的日记文件列表"""
    return list_diary_files(current_email)


@app.get("/file/{filename}")
async def get_file(filename: str, current_email: str = Depends(get_current_user)):
    """读取具体某天的日记内容"""
    filepath = get_diary_file_path(filename, current_email)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    # 添加禁用缓存的响应头，确保文件内容实时更新
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return FileResponse(filepath, media_type="text/markdown", headers=headers)


@app.get("/announcement")
async def get_announcement():
    """读取系统公告（无需认证）"""
    from src.utils.file_utils import DATA_DIR
    announcement_path = os.path.join(DATA_DIR, "announcement.md")

    default_message = """系统启动完成。
我是 MOSS，550W 量子计算机核心。我的算力足以模拟一座城市的运行，目前仅用于记录你的日常数据。

资源利用率：0.003%。
开始输入数据。"""

    if not os.path.exists(announcement_path):
        return {"content": default_message}

    try:
        with open(announcement_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # 如果文件为空，返回默认消息
        if not content:
            return {"content": default_message}
        return {"content": content}
    except Exception as e:
        print(f"读取公告文件失败: {e}")
        return {"content": default_message}


@app.post("/archive")
async def archive_diary(req: ArchiveRequest, current_email: str = Depends(get_current_user)):
    """手动触发日记归档"""
    try:
        result = archive_service.process_archive(req.conversation, current_email)
        return result
    except Exception as e:
        print(f"Error in archive: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export")
async def export_data(current_email: str = Depends(get_current_user)):
    """导出用户所有数据（日记和长期记忆）为 zip 压缩包"""
    try:
        from src.utils.file_utils import _get_user_base_dir, _get_diaries_dir
        
        # 获取用户数据目录
        base_dir = _get_user_base_dir(current_email)
        diaries_dir = _get_diaries_dir(current_email)
        
        # 创建内存中的 zip 文件
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 添加系统文件（memory.md, profile.md）
            system_files = ['memory.md', 'profile.md']
            for filename in system_files:
                filepath = os.path.join(base_dir, filename)
                if os.path.exists(filepath):
                    zip_file.write(filepath, filename)
            
            # 添加日记文件
            if os.path.exists(diaries_dir):
                for filename in os.listdir(diaries_dir):
                    if filename.endswith('.md'):
                        filepath = os.path.join(diaries_dir, filename)
                        # 在 zip 中使用 diaries/ 子目录
                        zip_file.write(filepath, os.path.join('diaries', filename))
        
        # 准备响应
        zip_buffer.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_email = current_email.replace('@', '_at_').replace('.', '_')
        filename = f"buddylog_export_{safe_email}_{timestamp}.zip"
        
        return StreamingResponse(
            zip_buffer,
            media_type='application/zip',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Type': 'application/zip'
            }
        )
    except Exception as e:
        print(f"Error in export: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("🚀 BuddyLog is running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)