import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv
import dashscope

# 加载配置
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
MODEL = os.getenv("MODEL_NAME", "qwen-plus")

# 导入模块
from src.models.schemas import Message, ArchiveRequest
from src.services.chat_service import ChatService
from src.services.archive_service import ArchiveService
from src.utils.file_utils import ensure_data_dir, list_diary_files, get_diary_file_path

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
chat_service = ChatService(MODEL)
archive_service = ArchiveService(MODEL)


@app.get("/")
async def read_root():
    """根路径，返回前端页面"""
    # 使用绝对路径读取index.html文件
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/chat")
async def chat(msg: Message):
    """聊天接口"""
    try:
        result = chat_service.process_chat(msg.content, msg.history)
        return result
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
async def list_files():
    """获取已有的日记文件列表"""
    return list_diary_files()


@app.get("/file/{filename}")
async def get_file(filename: str):
    """读取具体某天的日记内容"""
    filepath = get_diary_file_path(filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, media_type="text/markdown")


@app.post("/archive")
async def archive_diary(req: ArchiveRequest):
    """手动触发日记归档"""
    try:
        result = archive_service.process_archive(req.conversation)
        return result
    except Exception as e:
        print(f"Error in archive: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("🚀 BuddyLog is running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)