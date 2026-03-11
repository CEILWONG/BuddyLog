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
        
        # 只提取用户发言（Buddy的回应不进入记忆）
        user_messages = [
            f"[{msg.get('time', '--:--:--')}] {msg['content']}"
            for msg in conversation if msg['role'] == 'user'
        ]
        conversation_text = "\n".join(user_messages) if user_messages else "今日无记录"
        
        # 构建记忆融合提示
        memory_prompt = f"""你是一位专业的用户画像分析师，负责维护用户的长期记忆档案。

【任务】
根据用户今日的发言，更新长期记忆。只关注用户本人的信息、状态、目标和事件。

【旧记忆档案】
{old_memory if old_memory else '（暂无记忆档案）'}

【用户今日发言】（日期：{date_str}）
{conversation_text}

【更新原则】
1. 只记录关于**用户**的事实，不记录AI的回应
2. 新信息覆盖旧信息（以日期{date_str}判断新旧）
3. 重要事件必须标注日期格式：YYYY-MM-DD
4. 保持简洁，去除冗余描述，合并相似信息
5. 未完结事项保留，已完成的标记为已完成

【输出格式】
必须按以下结构输出（保持Markdown格式）：

# 长期记忆

## 用户核心画像
[基本信息、性格特点、喜好习惯等]

## 长期目标
[用户明确表达的目标和愿望]

## 未完结事项
[进行中的任务、待办事项]

## 价值观
[用户的核心信念和原则]

## 重要事件
[按时间倒序排列，格式：YYYY-MM-DD：事件描述]

【重要】本次对话日期是{date_str}，请以此日期为准判断信息时效性。"""
        
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