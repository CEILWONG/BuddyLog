from src.utils.file_utils import load_memory, update_memory_file
import dashscope
from dashscope import Generation


class MemoryService:
    """记忆服务类"""
    
    def __init__(self, model: str):
        self.model = model
    
    def update_memory(self, conversation: list, date_str: str):
        """更新长期记忆，融合新的对话内容"""
        # 加载旧记忆
        old_memory = load_memory()
        
        # 将对话转换为文本格式
        conversation_text = "\n".join([
            f"用户: {msg['content']}" if msg['role'] == 'user' else f"Buddy: {msg['content']}"
            for msg in conversation
        ])

        # 构建记忆融合提示
        memory_prompt = f"""请根据用户与Buddy的对话记录，更新用户的长期记忆。
【重要】本次对话的日期是 {date_str}，请根据日期判断信息的时效性。

旧记忆：
{old_memory}

新对话记录（{date_str}）：
{conversation_text}

【各模块更新规则 - 严格遵守】

### 1. 用户核心画像（高优先级更新）
- 职业、关系、喜好、当前状态等基础信息
- 规则：用新的事实覆盖旧的，以日期判断新旧（如换工作、搬家、状态变化）
- 注意：保持简洁，去除重复描述

### 2. 长期目标（动态调整）
- 用户的远期规划、愿景
- 规则：新增目标直接追加；已有目标若完成则标注(已完成)，若变更则更新描述
- 注意：区分"长期目标"与"未完结事项"

### 3. 未完结事项（任务追踪）
- 进行中的任务、待办事项
- 规则：
  - 新任务：直接追加
  - 已完成事项：移至"重要事件"记录完成节点，或标注(已完成)
  - 已取消事项：删除或标注(已取消)
- 注意：保持清单简洁，避免无限增长

### 4. 价值观（谨慎更新）
- 核心信念、人生哲学
- 规则：价值观相对稳定，仅当用户明确表达新的核心观念时才追加
- 注意：不要频繁修改，保持核心稳定性

### 5. 重要事件（最高优先级保护）
- 按时间顺序记录的关键事件、里程碑
- 【铁律】已有事件严禁修改、删除或重写
- 规则：仅追加当天({date_str})新发生的事件
- 注意：确保时间线准确，这是用户生命历程的客观记录

【通用要求】
1. 去重：去除重复的信息
2. 提炼：提取重要信息，保持记忆简洁
3. 格式：保持原有的Markdown格式和层级结构
4. 时效：对话日期{date_str}是最新信息，以此判断新旧"""
        
        memory_messages = [
            {"role": "system", "content": memory_prompt}
        ]
        
        memory_response = Generation.call(
            api_key=dashscope.api_key,
            model=self.model,
            messages=memory_messages,
            result_format="message",
            enable_search=False,
            enable_thinking=False
        )
        
        if memory_response.status_code == 200:
            new_memory = memory_response.output.choices[0].message.content
            # 写入新记忆
            update_memory_file(new_memory)
            print("Memory updated successfully")
        else:
            print(f"Failed to update memory: {memory_response.message}")