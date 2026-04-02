from src.utils.file_utils import load_memory, update_memory_file
from openai import OpenAI


class MemoryService:
    """记忆服务类"""
    
    def __init__(self, model: str, openai_client: OpenAI = None, enable_thinking: bool = False):
        self.model = model
        self.openai_client = openai_client
        self.enable_thinking = enable_thinking
    
    def update_memory(self, conversation: list, date_str: str, user_email: str = None):
        """更新长期记忆，融合新的对话内容"""
        # 加载旧记忆
        old_memory = load_memory(user_email)
        
        # 将对话转换为文本格式
        conversation_text = "\n".join([
            f"用户: {msg['content']}" if msg['role'] == 'user' else f"Buddy: {msg['content']}"
            for msg in conversation
        ])

        # 构建记忆融合提示
        memory_prompt = f"""请根据用户与Buddy的对话记录，更新用户的长期记忆。

【关键概念 - 严格区分】
- 对话发生日期：{date_str}（这是用户实际对话的日期，所有事件必须按此日期记录）
- 当前处理时间：现在（这只是AI处理记忆的时间，不影响事件日期）

【重要】本次对话记录的发生日期是 {date_str}。无论今天是哪一天，对话中的事件都发生在 {date_str}。

旧记忆：
{old_memory}

新对话记录（发生日期：{date_str}）：
{conversation_text}

【各模块更新规则 - 严格遵守】

### 1. 用户核心画像（高优先级更新）
- 职业、关系、喜好、当前状态等基础信息
- 规则：用新的事实覆盖旧的，以对话发生日期{date_str}判断新旧（如换工作、搬家、状态变化）
- 注意：保持简洁，去除重复描述

### 2. 长期目标（动态调整）
- 用户的远期规划、愿景
- 规则：新增目标直接追加；已有目标若完成则标注(已完成)，若变更则更新描述
- 注意：区分"长期目标"与"未完结事项"

### 3. 未完结事项（任务追踪）
- 进行中的任务、待办事项
- 规则：
  - 新任务：直接追加
  - 已完成事项：移至"重要事件"记录完成节点（标注完成日期{date_str}），或标注(已完成)
  - 已取消事项：删除或标注(已取消)
- 注意：保持清单简洁，避免无限增长

### 4. 价值观（谨慎更新）
- 核心信念、人生哲学
- 规则：价值观相对稳定，仅当用户明确表达新的核心观念时才追加
- 注意：不要频繁修改，保持核心稳定性

### 5. 重要事件（最高优先级保护）
- 按时间顺序记录的关键事件、里程碑
- 【铁律1】已有事件严禁修改、删除、重写、简化或省略
- 【铁律2】旧记忆中的重要事件必须原封不动照抄，不得用"保持原有记录"、"下略"等任何方式替代
- 【铁律3】所有新事件必须标注日期 {date_str}，这是事件发生的真实日期
- 规则：
  1. 完整复制旧记忆中的重要事件列表（一字不改）
  2. 在列表顶部追加 {date_str} 的新事件
  3. 按日期倒序排列（新日期在前，旧日期在后）
- 注意：确保时间线准确，这是用户生命历程的客观记录

【通用要求】
1. 去重：去除重复的信息
2. 提炼：提取重要信息，保持记忆简洁
3. 格式：保持原有的Markdown格式和层级结构
4. 时效判断基准：对话发生日期 {date_str} 是判断新旧信息的唯一标准"""
        
        memory_messages = [
            {"role": "system", "content": memory_prompt}
        ]
        
        # 调用API (OpenAI 格式)
        # 构建 extra_body 参数（明确传递 enable_thinking 控制深度思考）
        extra_body = {"enable_thinking": self.enable_thinking}
        
        try:
            memory_response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=memory_messages,
                extra_body=extra_body
            )
            new_memory = memory_response.choices[0].message.content
            # 写入新记忆
            update_memory_file(new_memory, user_email)
            print(f"Memory updated successfully for user: {user_email}")
        except Exception as e:
            print(f"Failed to update memory for user {user_email}: {str(e)}")