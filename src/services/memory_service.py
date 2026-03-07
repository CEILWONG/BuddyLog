from src.utils.file_utils import load_memory, update_memory_file
import dashscope
from dashscope import Generation


class MemoryService:
    """记忆服务类"""
    
    def __init__(self, model: str):
        self.model = model
    
    def update_memory(self, new_diary_content: str):
        """更新长期记忆，融合新日记内容"""
        # 加载旧记忆
        old_memory = load_memory()
        
        # 构建记忆融合提示
        memory_prompt = f"""
        请根据新的日记内容，更新用户的长期记忆。
        
        旧记忆：
        {old_memory}
        
        新日记：
        {new_diary_content}
        
        更新要求：
        1. 去重：去除重复的信息
        2. 更新：用新的事实覆盖旧的事实（如换工作、搬家等）
        3. 提炼：提取重要信息，保持记忆简洁
        4. 保留：保留未完结的事项
        5. 格式：保持原有的Markdown格式
        """
        
        memory_messages = [
            {"role": "system", "content": memory_prompt}
        ]
        
        memory_response = Generation.call(
            api_key=dashscope.api_key,
            model=self.model,
            messages=memory_messages,
            result_format="message"
        )
        
        if memory_response.status_code == 200:
            new_memory = memory_response.output.choices[0].message.content
            # 写入新记忆
            update_memory_file(new_memory)
            print("Memory updated successfully")
        else:
            print(f"Failed to update memory: {memory_response.message}")