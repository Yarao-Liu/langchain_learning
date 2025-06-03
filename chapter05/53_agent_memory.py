"""
LangChain 现代记忆模块完整示例（详细注释版）

本示例展示了如何在LangChain中实现各种类型的记忆功能，包括：
1. 基础聊天记忆 - 简单的消息历史记录
2. 带记忆的对话链 - 在对话中保持上下文
3. 缓冲记忆 - 保存所有对话历史
4. 窗口记忆 - 只保留最近的N轮对话
5. 摘要记忆 - 使用LLM生成对话摘要
6. 多会话记忆管理 - 管理多个用户的独立会话
7. 持久化记忆 - 将记忆保存到文件

注意：本示例使用现代LangChain API，避免了已弃用的langchain.memory模块

作者：AI助手
日期：2024年
版本：1.0
"""

import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from typing import List
import json
from datetime import datetime

# ============================================================================
# 环境配置和模型初始化
# ============================================================================

# 加载环境变量文件(.env)，包含API密钥等敏感信息
# 确保在项目根目录有.env文件，内容如：OPENAI_API_KEY=your_api_key_here
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 从环境变量获取OpenAI API密钥

# 检查API密钥是否存在
if not api_key:
    raise ValueError("未找到OPENAI_API_KEY环境变量，请在.env文件中设置")

# 初始化ChatOpenAI模型
# 这里使用的是兼容OpenAI API的第三方服务(SiliconFlow)
llm = ChatOpenAI(
    api_key=api_key,                              # API密钥
    base_url="https://api.siliconflow.cn/v1/",    # 第三方API服务地址
    model="Qwen/Qwen2.5-7B-Instruct",            # 使用的模型名称
    temperature=0.7                               # 控制输出的随机性，0-1之间，越高越随机
)

print("=" * 60)
print("LangChain 记忆模块完整示例（详细注释版）")
print("=" * 60)

# ============================================================================
# 1. 基础聊天记忆 - ChatMessageHistory
# ============================================================================
print("\n1. 基础聊天记忆 - ChatMessageHistory")
print("-" * 40)

"""
ChatMessageHistory 是LangChain中最基础的记忆组件
- 功能：简单地存储和检索聊天消息历史
- 特点：内存存储，程序结束后数据丢失
- 适用场景：简单的对话历史记录，不需要复杂的记忆管理
- 存储结构：消息列表，每个消息包含类型（用户/AI）和内容
"""

# 创建聊天记忆历史实例
# ChatMessageHistory是一个简单的内存存储，用于保存对话消息
chat_history = ChatMessageHistory()

# 添加用户和AI的历史消息
# add_user_message(): 添加用户消息，会创建HumanMessage对象
# add_ai_message(): 添加AI消息，会创建AIMessage对象
chat_history.add_user_message("你好，我叫张三")
chat_history.add_ai_message("你好张三！很高兴认识你。")
chat_history.add_user_message("我喜欢编程")
chat_history.add_ai_message("编程是一个很有趣的技能！你主要使用什么编程语言？")

# 遍历并显示所有历史消息
# chat_history.messages 返回所有消息的列表
print("聊天历史:")
for message in chat_history.messages:
    # 使用isinstance()检查消息类型，并相应地显示
    if isinstance(message, HumanMessage):
        print(f"用户: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")

# ============================================================================
# 2. 带记忆的对话链 - RunnableWithMessageHistory
# ============================================================================
print("\n\n2. 带记忆的对话链 - RunnableWithMessageHistory")
print("-" * 40)

"""
RunnableWithMessageHistory 是LangChain中用于创建有记忆的对话链的核心组件
- 功能：将记忆功能集成到LangChain的处理链中
- 特点：自动管理对话历史，支持多会话
- 适用场景：需要上下文感知的对话系统
- 工作原理：在每次调用时自动加载历史消息，并在处理后保存新消息
"""

# 创建提示模板，包含历史消息占位符
# ChatPromptTemplate.from_messages() 创建一个结构化的提示模板
prompt = ChatPromptTemplate.from_messages([
    # 系统消息：定义AI的角色和行为
    ("system", "你是一个友好的AI助手。请根据对话历史来回答用户的问题。"),
    # MessagesPlaceholder：为历史消息预留位置，变量名为"chat_history"
    MessagesPlaceholder(variable_name="chat_history"),
    # 人类消息：当前用户输入，使用{input}占位符
    ("human", "{input}")
])

# 创建基础处理链
# 使用管道操作符(|)连接：提示模板 -> LLM -> 输出解析器
# 这是LangChain的链式调用模式，数据从左到右流动
chain = prompt | llm | StrOutputParser()

# 创建会话存储字典，用于保存不同用户的对话历史
# 键：session_id（会话ID），值：ChatMessageHistory对象
# 这允许系统同时处理多个用户的独立对话
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    获取指定会话的历史记录
    
    这是一个工厂函数，用于为RunnableWithMessageHistory提供会话历史
    
    Args:
        session_id (str): 会话唯一标识符，通常是用户ID或会话ID
        
    Returns:
        ChatMessageHistory: 该会话的历史记录对象
        
    Note:
        如果会话不存在，会自动创建新的ChatMessageHistory实例
    """
    # 如果会话不存在，创建新的ChatMessageHistory
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 创建带记忆功能的处理链
# RunnableWithMessageHistory 包装基础链，添加记忆功能
chain_with_history = RunnableWithMessageHistory(
    chain,                              # 基础处理链
    get_session_history,                # 获取会话历史的函数
    input_messages_key="input",         # 输入消息的键名
    history_messages_key="chat_history", # 历史消息的键名
)

# 演示多轮对话，展示记忆功能
session_id = "user_123"  # 定义会话ID，用于标识特定用户的对话
print(f"会话ID: {session_id}")

# 第一轮对话：用户自我介绍
# invoke() 方法调用带记忆的链，传入输入和配置
response1 = chain_with_history.invoke(
    {"input": "我叫李四，我是一名软件工程师"},  # 用户输入
    config={"configurable": {"session_id": session_id}}  # 配置会话ID
)
print(f"用户: 我叫李四，我是一名软件工程师")
print(f"AI: {response1}")

# 第二轮对话：测试AI是否记住了用户姓名
response2 = chain_with_history.invoke(
    {"input": "你还记得我的名字吗？"},
    config={"configurable": {"session_id": session_id}}
)
print(f"用户: 你还记得我的名字吗？")
print(f"AI: {response2}")

# 第三轮对话：测试AI是否记住了用户职业
response3 = chain_with_history.invoke(
    {"input": "我的职业是什么？"},
    config={"configurable": {"session_id": session_id}}
)
print(f"用户: 我的职业是什么？")
print(f"AI: {response3}")

# ============================================================================
# 3. 现代缓冲记忆 - 使用ChatMessageHistory
# ============================================================================
print("\n\n3. 现代缓冲记忆 - 使用ChatMessageHistory")
print("-" * 40)

"""
缓冲记忆(Buffer Memory)是最简单的记忆类型
- 功能：保存所有的对话历史，不做任何处理
- 优点：完整保留所有信息，实现简单
- 缺点：随着对话增长，内存使用量线性增加
- 适用场景：短期对话，或者内存充足的环境
- 内存复杂度：O(n)，其中n是消息数量
"""

class BufferMemory:
    """
    现代缓冲记忆实现
    
    这个类封装了ChatMessageHistory，提供了简单的接口来管理对话历史
    所有的对话都会被完整保存，直到手动清除
    """
    
    def __init__(self):
        """初始化缓冲记忆"""
        self.chat_history = ChatMessageHistory()

    def add_conversation(self, user_msg: str, ai_msg: str):
        """
        添加一轮完整对话
        
        Args:
            user_msg (str): 用户消息
            ai_msg (str): AI回复消息
        """
        self.chat_history.add_user_message(user_msg)
        self.chat_history.add_ai_message(ai_msg)

    def get_messages(self) -> List[BaseMessage]:
        """
        获取所有消息
        
        Returns:
            List[BaseMessage]: 所有消息的列表，按时间顺序排列
        """
        return self.chat_history.messages

    def clear(self):
        """清空所有记忆"""
        self.chat_history.clear()
        
    def get_message_count(self) -> int:
        """
        获取消息总数
        
        Returns:
            int: 消息总数
        """
        return len(self.chat_history.messages)

# 创建缓冲记忆实例
buffer_memory = BufferMemory()

# 添加对话历史到缓冲记忆
# 这些对话会被完整保存，不会被删除或修改
buffer_memory.add_conversation("我今天学习了Python", "太好了！Python是一门很实用的编程语言。")
buffer_memory.add_conversation("我想学习机器学习", "机器学习很有趣！建议从scikit-learn开始。")

# 获取并显示缓冲记忆中的所有内容
print("缓冲记忆内容:")
print(f"总消息数: {buffer_memory.get_message_count()}")
for message in buffer_memory.get_messages():
    if isinstance(message, HumanMessage):
        print(f"用户: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")

# ============================================================================
# 4. 现代窗口记忆 - 只保留最近的对话
# ============================================================================
print("\n\n4. 现代窗口记忆 - 只保留最近的对话")
print("-" * 40)

"""
窗口记忆(Window Memory)只保留最近的N轮对话
- 功能：维护固定大小的对话窗口，自动删除旧对话
- 优点：内存使用量固定，不会无限增长
- 缺点：会丢失早期的对话信息
- 适用场景：长期对话，但只需要关注最近的上下文
- 内存复杂度：O(k)，其中k是窗口大小
"""

class WindowMemory:
    """
    现代窗口记忆实现

    维护一个固定大小的对话窗口，当新对话加入时，自动删除最旧的对话
    这种方式确保内存使用量保持恒定，适合长期运行的对话系统
    """

    def __init__(self, k: int = 2):
        """
        初始化窗口记忆

        Args:
            k (int): 保留的对话轮数（每轮包含用户消息和AI回复）
                    默认为2，即保留最近2轮对话（4条消息）
        """
        self.k = k  # 保留最近k轮对话
        self.chat_history = ChatMessageHistory()

    def add_conversation(self, user_msg: str, ai_msg: str):
        """
        添加对话并维护窗口大小

        Args:
            user_msg (str): 用户消息
            ai_msg (str): AI回复消息

        Note:
            如果添加后超过窗口大小，会自动删除最早的对话
        """
        # 添加新的用户消息和AI回复
        self.chat_history.add_user_message(user_msg)
        self.chat_history.add_ai_message(ai_msg)

        # 维护窗口大小：如果超过k轮对话（k*2条消息），删除最早的消息
        # 每次删除一对消息（用户消息+AI回复）
        while len(self.chat_history.messages) > self.k * 2:
            self.chat_history.messages.pop(0)  # 删除最早的用户消息
            self.chat_history.messages.pop(0)  # 删除最早的AI回复

    def get_messages(self) -> List[BaseMessage]:
        """
        获取窗口内的所有消息

        Returns:
            List[BaseMessage]: 窗口内的消息列表，最多包含k*2条消息
        """
        return self.chat_history.messages

    def get_window_info(self) -> dict:
        """
        获取窗口信息

        Returns:
            dict: 包含窗口大小、当前消息数等信息
        """
        return {
            "window_size": self.k,
            "current_messages": len(self.chat_history.messages),
            "current_conversations": len(self.chat_history.messages) // 2
        }

# 创建窗口记忆（只保留最近2轮对话）
window_memory = WindowMemory(k=2)

# 添加多轮对话，观察窗口记忆的行为
conversations = [
    ("我叫王五", "你好王五！"),
    ("我住在北京", "北京是个很棒的城市！"),
    ("我喜欢旅游", "旅游能开阔视野，很不错！"),
    ("我想去上海", "上海也是个很有魅力的城市！"),
    ("你还记得我的名字吗？", "让我看看...")
]

print("逐步添加对话到窗口记忆:")
for i, (user_msg, ai_msg) in enumerate(conversations, 1):
    window_memory.add_conversation(user_msg, ai_msg)
    info = window_memory.get_window_info()
    print(f"第{i}轮对话后 - 当前对话数: {info['current_conversations']}/{info['window_size']}")

# 获取窗口记忆内容（只显示最近k=2轮对话）
print("\n窗口记忆内容（最近2轮对话）:")
for message in window_memory.get_messages():
    if isinstance(message, HumanMessage):
        print(f"用户: {message.content}")
    elif isinstance(message, AIMessage):
        print(f"AI: {message.content}")

# ============================================================================
# 5. 现代摘要记忆 - 使用LLM生成对话摘要
# ============================================================================
print("\n\n5. 现代摘要记忆 - 使用LLM生成对话摘要")
print("-" * 40)

"""
摘要记忆(Summary Memory)使用LLM来压缩对话历史
- 功能：当对话过长时，使用LLM生成摘要来压缩历史信息
- 优点：保留重要信息的同时减少内存使用
- 缺点：可能丢失一些细节信息，依赖LLM的摘要质量
- 适用场景：长期对话，需要保留历史信息但控制内存使用
- 内存复杂度：O(1)，摘要大小相对固定
"""

class SummaryMemory:
    """
    现代摘要记忆实现

    当对话历史超过指定长度时，使用LLM生成摘要来压缩旧的对话
    保留最近的对话和历史摘要，实现内存使用的平衡
    """

    def __init__(self, llm, max_messages: int = 10):
        """
        初始化摘要记忆

        Args:
            llm: 用于生成摘要的语言模型
            max_messages (int): 触发摘要的最大消息数，默认10条
        """
        self.llm = llm
        self.max_messages = max_messages
        self.chat_history = ChatMessageHistory()
        self.summary = ""  # 存储历史对话的摘要

    def add_conversation(self, user_msg: str, ai_msg: str):
        """
        添加对话，超过限制时生成摘要

        Args:
            user_msg (str): 用户消息
            ai_msg (str): AI回复消息
        """
        self.chat_history.add_user_message(user_msg)
        self.chat_history.add_ai_message(ai_msg)

        # 如果消息数量超过限制，生成摘要
        if len(self.chat_history.messages) > self.max_messages:
            self._create_summary()

    def _create_summary(self):
        """
        创建对话摘要

        将较旧的对话转换为摘要，只保留最近的对话
        这是一个私有方法，由add_conversation自动调用
        """
        # 保留最近2轮对话（4条消息）
        recent_messages = self.chat_history.messages[-4:]
        old_messages = self.chat_history.messages[:-4]

        if old_messages:
            # 构建要摘要的文本
            messages_text = ""
            for msg in old_messages:
                if isinstance(msg, HumanMessage):
                    messages_text += f"用户: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    messages_text += f"AI: {msg.content}\n"

            # 创建摘要提示
            summary_prompt = f"""
            请将以下对话总结成简洁的摘要，保留关键信息：

            {messages_text}

            摘要：
            """

            try:
                # 使用LLM生成摘要
                response = self.llm.invoke(summary_prompt)
                new_summary = response.content if hasattr(response, 'content') else str(response)

                # 更新摘要：如果已有摘要，则合并；否则创建新摘要
                if self.summary:
                    self.summary = f"{self.summary}\n{new_summary}"
                else:
                    self.summary = new_summary

                # 清除旧消息，只保留最近的对话
                self.chat_history.messages = recent_messages
                print(f"生成了新的对话摘要")
            except Exception as e:
                print(f"生成摘要时出错: {e}")

    def get_context(self) -> str:
        """
        获取完整的上下文（摘要+最近对话）

        Returns:
            str: 包含历史摘要和最近对话的完整上下文
        """
        context = ""
        if self.summary:
            context += f"对话摘要: {self.summary}\n\n"

        context += "最近对话:\n"
        for msg in self.chat_history.messages:
            if isinstance(msg, HumanMessage):
                context += f"用户: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context += f"AI: {msg.content}\n"

        return context

    def get_memory_stats(self) -> dict:
        """
        获取记忆统计信息

        Returns:
            dict: 包含摘要长度、当前消息数等统计信息
        """
        return {
            "has_summary": bool(self.summary),
            "summary_length": len(self.summary) if self.summary else 0,
            "current_messages": len(self.chat_history.messages),
            "max_messages": self.max_messages
        }

# 创建摘要记忆
summary_memory = SummaryMemory(llm, max_messages=6)

# 添加长对话历史，触发摘要生成
long_conversations = [
    ("我是一名数据科学家", "很高兴认识你！数据科学是个很有前景的领域。"),
    ("我在一家互联网公司工作", "互联网行业发展很快，一定很有挑战性。"),
    ("我们公司主要做电商业务", "电商是个竞争激烈的行业，需要不断创新。"),
    ("我负责用户行为分析", "用户行为分析对业务决策很重要。"),
    ("我们使用Python和SQL进行数据分析", "这是数据分析的经典组合工具。"),
    ("最近在学习深度学习", "深度学习在很多领域都有应用，值得深入学习。")
]

print("添加长对话历史...")
for user_msg, ai_msg in long_conversations:
    summary_memory.add_conversation(user_msg, ai_msg)
    stats = summary_memory.get_memory_stats()
    print(f"当前消息数: {stats['current_messages']}, 有摘要: {stats['has_summary']}")

# 获取摘要记忆内容
print("\n摘要记忆内容:")
print(summary_memory.get_context())

# ============================================================================
# 6. 现代多会话记忆管理
# ============================================================================
print("\n\n6. 现代多会话记忆管理")
print("-" * 40)

"""
多会话记忆管理允许系统同时处理多个用户的独立对话
- 功能：为不同用户维护独立的记忆空间
- 特点：会话隔离，支持不同的记忆类型
- 适用场景：多用户聊天系统，客服系统
- 实现方式：使用会话ID作为键，存储不同的记忆实例
"""

# 创建多个会话的记忆存储
# 这是一个全局存储，用于管理所有用户的记忆
session_store = {}

def create_session_memory(session_id: str, memory_type: str = "buffer"):
    """
    为指定会话创建或获取记忆实例

    Args:
        session_id (str): 会话唯一标识符
        memory_type (str): 记忆类型，可选值：buffer, window, summary

    Returns:
        记忆实例：根据memory_type返回相应的记忆对象

    Note:
        如果会话已存在，直接返回现有的记忆实例
        如果会话不存在，根据memory_type创建新的记忆实例
    """
    if session_id not in session_store:
        if memory_type == "buffer":
            session_store[session_id] = BufferMemory()
        elif memory_type == "window":
            session_store[session_id] = WindowMemory(k=3)  # 保留3轮对话
        elif memory_type == "summary":
            session_store[session_id] = SummaryMemory(llm, max_messages=4)
        else:
            raise ValueError(f"不支持的记忆类型: {memory_type}")
    return session_store[session_id]

# 模拟不同用户的会话
# 每个元组包含：(用户ID, 记忆类型, 对话列表)
users = [
    ("user_001", "buffer", [("我是张三", "你好张三！"), ("我喜欢编程", "编程很有趣！")]),
    ("user_002", "window", [("我是李四", "你好李四！"), ("我是老师", "教师是个伟大的职业！")]),
    ("user_003", "summary", [("我在学习AI", "AI是未来的趋势！"), ("我想做研究", "研究工作很有意义！")])
]

print("为不同用户创建独立的记忆空间:")
for user_id, mem_type, conversations in users:
    print(f"\n用户 {user_id} (使用{mem_type}记忆):")

    # 为用户创建或获取记忆实例
    memory = create_session_memory(user_id, mem_type)

    # 添加用户的对话历史
    for user_msg, ai_msg in conversations:
        memory.add_conversation(user_msg, ai_msg)

    # 显示记忆内容
    if mem_type == "summary":
        print(f"  上下文: {memory.get_context()}")
    else:
        print("  对话历史:")
        for msg in memory.get_messages():
            if isinstance(msg, HumanMessage):
                print(f"    用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"    AI: {msg.content}")

print(f"\n当前管理的会话数: {len(session_store)}")

# ============================================================================
# 7. 持久化记忆示例
# ============================================================================
print("\n\n7. 持久化记忆示例")
print("-" * 40)

"""
持久化记忆将对话历史保存到文件系统
- 功能：将记忆数据保存到磁盘，程序重启后可恢复
- 特点：数据持久化，支持自动保存和加载
- 适用场景：需要长期保存用户对话历史的应用
- 存储格式：JSON格式，便于读取和编辑
"""

class PersistentMemory:
    """
    可持久化的记忆类

    将对话历史保存到JSON文件，支持自动加载和保存
    适合需要在程序重启后保持对话历史的场景
    """

    def __init__(self, file_path: str):
        """
        初始化持久化记忆

        Args:
            file_path (str): 保存记忆数据的文件路径
        """
        self.file_path = file_path
        self.memory = ChatMessageHistory()
        self.load_memory()  # 初始化时自动加载已有数据

    def save_memory(self):
        """
        保存记忆到文件

        将当前的对话历史序列化为JSON格式并保存到文件
        """
        data = []
        for msg in self.memory.messages:
            if isinstance(msg, HumanMessage):
                data.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                data.append({"type": "ai", "content": msg.content})

        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"记忆已保存到 {self.file_path}")
        except Exception as e:
            print(f"保存记忆时出错: {e}")

    def load_memory(self):
        """
        从文件加载记忆

        读取JSON文件并恢复对话历史
        如果文件不存在，会创建新的空记忆
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 重建对话历史
            for item in data:
                if item["type"] == "human":
                    self.memory.add_user_message(item["content"])
                elif item["type"] == "ai":
                    self.memory.add_ai_message(item["content"])
            print(f"记忆已从 {self.file_path} 加载，共{len(data)}条消息")
        except FileNotFoundError:
            print(f"记忆文件 {self.file_path} 不存在，创建新的记忆")
        except Exception as e:
            print(f"加载记忆时出错: {e}")

    def add_conversation(self, user_msg: str, ai_msg: str):
        """
        添加对话并自动保存

        Args:
            user_msg (str): 用户消息
            ai_msg (str): AI回复消息
        """
        self.memory.add_user_message(user_msg)
        self.memory.add_ai_message(ai_msg)
        self.save_memory()  # 每次添加对话后自动保存

    def get_messages(self):
        """
        获取所有消息

        Returns:
            List[BaseMessage]: 所有消息的列表
        """
        return self.memory.messages

    def clear_memory(self):
        """
        清空记忆并删除文件
        """
        self.memory.clear()
        try:
            os.remove(self.file_path)
            print(f"已清空记忆并删除文件 {self.file_path}")
        except FileNotFoundError:
            print("文件不存在，无需删除")
        except Exception as e:
            print(f"删除文件时出错: {e}")

# 创建持久化记忆实例
persistent_memory = PersistentMemory("data/memory_data.json")

# 添加对话（会自动保存到文件）
print("添加对话到持久化记忆:")
persistent_memory.add_conversation("我喜欢看电影", "电影是很好的娱乐方式！你喜欢什么类型的电影？")
persistent_memory.add_conversation("我喜欢科幻电影", "科幻电影很有想象力，能带我们探索未来世界。")

# 显示记忆内容
print("\n持久化记忆内容:")
for msg in persistent_memory.get_messages():
    if isinstance(msg, HumanMessage):
        print(f"用户: {msg.content}")
    elif isinstance(msg, AIMessage):
        print(f"AI: {msg.content}")

# ============================================================================
# 总结和最佳实践
# ============================================================================
print("\n" + "=" * 60)
print("现代记忆模块示例演示完成！")
print("=" * 60)

print("\n📚 记忆类型总结:")
print("1. 基础记忆 (ChatMessageHistory) - 简单存储，适合基础应用")
print("2. 缓冲记忆 (BufferMemory) - 完整保存，适合短期对话")
print("3. 窗口记忆 (WindowMemory) - 固定大小，适合长期对话")
print("4. 摘要记忆 (SummaryMemory) - 智能压缩，适合复杂对话")
print("5. 多会话管理 - 用户隔离，适合多用户系统")
print("6. 持久化记忆 - 数据保存，适合长期应用")
