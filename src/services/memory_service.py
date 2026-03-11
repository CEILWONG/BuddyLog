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
【重要】本次对话的日期是 {date_str}，请根据日期判断信息的时效性，如有变化，用新的信息覆盖旧的事实。

旧记忆：
{old_memory}

新对话记录（{date_str}）：
{conversation_text}

更新要求：
1. 去重：去除重复的信息
2. 更新：用新的事实覆盖旧的事实（如换工作、搬家等），以日期判断新旧
3. 提炼：提取重要信息，保持记忆简洁
4. 保留：保留未完结的事项
5. 格式：保持原有的Markdown格式
6. 时效：注意对话日期{date_str}，这是最新信息"""
        
        memory_messages = [
            {"role": "system", "content": memory_prompt}
        ]
        
        memory_response = Generation.call(
            api_key=dashscope.api_key,
            model=self.model,
            messages=memory_messages,
            result_format="message",
            enable_search=False
        )
        
        if memory_response.status_code == 200:
            new_memory = memory_response.output.choices[0].message.content
            # 写入新记忆
            update_memory_file(new_memory)
            print("Memory updated successfully")
        else:
            print(f"Failed to update memory: {memory_response.message}")