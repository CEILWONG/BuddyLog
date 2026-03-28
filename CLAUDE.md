# BuddyLog - AI 聊天日记系统

## 项目简介

BuddyLog 是一款 AI 聊天日记应用，用户通过与 Buddy（AI 角色）自然对话，系统自动将对话归档为日记文章。支持文本、图片、语音三种输入方式。

## 技术栈

- **后端**: Python FastAPI, uvicorn
- **前端**: 纯 HTML/CSS/JS 单页应用（index.html），无框架
- **AI**: 阿里云 DashScope API
  - 文本对话: qwen-plus (通过 `Generation` API)
  - 图片理解: qwen-vl-max (通过 `MultiModalConversation` API)
  - 语音识别: paraformer-realtime-v2 (通过 `Recognition` + callback 流式 API)
- **存储**: 本地文件系统，Markdown 文件，按用户隔离
- **认证**: JWT (python-jose)

## 项目架构

```
BuddyLog/
├── main.py                          # FastAPI 入口，所有路由定义
├── index.html                       # 前端单页应用（~2800行，含 CSS/JS）
├── requirements.txt                 # Python 依赖
├── .env                             # 环境变量（不入库）
├── src/
│   ├── models/
│   │   ├── schemas.py               # 数据模型: Message, ArchiveRequest
│   │   └── auth_schemas.py          # 认证模型: UserRegister, UserLogin
│   ├── services/
│   │   ├── chat_service.py          # 聊天服务: 构建 prompt, 调 LLM, 多模态处理
│   │   ├── archive_service.py       # 归档服务: 结构化提取, 日记生成
│   │   ├── memory_service.py        # 记忆服务: LLM 融合长期记忆
│   │   └── auth_service.py          # 认证服务: 密码哈希
│   └── utils/
│       ├── file_utils.py            # 文件操作: 日记读写, 草稿管理, memory 更新
│       ├── user_utils.py            # 用户管理: 创建用户, 设置, 用量统计
│       └── auth_utils.py            # JWT 工具: token 生成与验证
└── data/                            # 数据目录（不入库）
    ├── user_index.json              # 用户凭证索引
    └── users/{user_id}/
        ├── memory.md                # 长期记忆
        ├── profile.md               # 用户画像
        ├── images/                  # 上传的图片
        ├── audio/                   # 录音文件（WAV）
        └── diaries/
            ├── diary_YYYY-MM-DD_draft.md   # 当天草稿
            └── diary_YYYY-MM-DD_N.md       # 归档日记
```

## API 路由

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| GET | / | 前端页面 | 否 |
| POST | /auth/register | 用户注册 | 否 |
| POST | /auth/login | 用户登录 | 否 |
| GET | /auth/me | 当前用户信息 | 是 |
| GET/PUT | /auth/settings | 用户设置 | 是 |
| POST | /chat | 聊天（文本/图片） | 是 |
| POST | /archive | 手动归档今日日记 | 是 |
| GET | /files | 日记文件列表 | 是 |
| GET | /file/{filename} | 读取日记内容 | 是 |
| GET | /export | 导出所有数据为 ZIP | 是 |
| GET | /announcement | 系统公告 | 否 |
| POST | /upload-image | 上传图片 | 是 |
| GET | /images/{user_id}/{filename} | 访问图片 | 否 |
| POST | /transcribe | 语音转文字 | 是 |
| GET | /audio/{user_id}/{filename} | 访问录音 | 否 |

## 核心流程

### 聊天流程
1. 前端发送 `{content, history}` 到 POST /chat
2. ChatService 构建 system prompt（角色设定 + 长期记忆 + 最近日记）
3. 检测是否含 `[图片: filename]` 标记 -> 选择文本模型或多模态模型
4. 调用 DashScope API 生成回复
5. 追加到当天草稿文件 `diary_YYYY-MM-DD_draft.md`
6. 后台线程检查并自动归档过期草稿

### 归档流程
1. 从草稿提取对话记录
2. LLM 生成日记文章（第三人称叙事 + AI 评价）
3. LLM 融合更新长期记忆 memory.md
4. 写入正式日记文件，删除草稿

### 多模态消息处理
- 消息含 `[图片: filename]` -> 用 `MultiModalConversation` + `qwen-vl-max`
- 消息含 `[语音: filename]` -> 仅作为标记保存，不影响 LLM 调用
- 纯文本 -> 用 `Generation` + `qwen-plus`

## 开发规范

### 代码风格
- 后端遵循现有风格：中文注释，函数命名用英文
- 前端全在 index.html 内，CSS 在 `<style>` 中，JS 在 `<script>` 中
- 桌面端和移动端有两套 HTML 结构，改 UI 时两处都要改

### 环境变量
- `DASHSCOPE_API_KEY` - 阿里云 DashScope API Key（必须）
- `MODEL_NAME` - 文本模型名（默认 qwen-plus）
- `DATA_DIR` - 数据存储目录（默认 ./data）

### 注意事项
- `.env` 文件不要提交，已在 .gitignore 中
- `data/` 目录不入库，包含用户数据
- 前端使用相对路径调 API，部署时后端和前端同源
- 图片上传限制 10MB，支持 JPG/PNG/GIF/WebP
- 语音录制前端直接输出 WAV（16kHz 单声道），不依赖 ffmpeg
- 多模态模型 qwen-vl-max 要求图片宽高 > 10px
- 修改 chat_service.py 时注意 system prompt 要求模型返回 `{"reply": "..."}` JSON 格式
- 归档服务的结构化提取（extract_structured_data）当前被注释掉以加速，如需恢复在 archive_service.py 中取消注释
