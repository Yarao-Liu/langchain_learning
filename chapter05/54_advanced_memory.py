"""
LangChain 现代高级记忆模块示例

本示例展示了LangChain中更高级的记忆管理技术，包括：
1. 摘要缓冲记忆 - 结合摘要和缓冲的优点，智能管理token使用
2. 带时间戳记忆 - 为每条消息添加时间戳，便于追踪对话时间线
3. 持久化记忆 - 将对话历史保存到文件系统，支持程序重启后恢复
4. 智能记忆筛选 - 基于消息重要性进行智能筛选和管理

这些高级技术适用于：
- 长期运行的对话系统
- 需要精确控制内存使用的应用
- 要求数据持久化的生产环境
- 需要智能内容管理的复杂对话场景

技术特点：
- 避免使用已弃用的langchain.memory模块
- 使用现代LangChain API
- 提供完整的错误处理
- 支持自定义扩展

作者：AI助手
日期：2024年
版本：1.0
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

# ============================================================================
# 环境配置和模型初始化
# ============================================================================

# 加载环境变量文件(.env)，包含API密钥等敏感信息
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 检查API密钥是否存在
if not api_key:
    raise ValueError("未找到OPENAI_API_KEY环境变量，请在.env文件中设置")

# 初始化ChatOpenAI模型
# 使用兼容OpenAI API的第三方服务
llm = ChatOpenAI(
    api_key=api_key,                              # API密钥
    base_url="https://api.siliconflow.cn/v1/",    # 第三方API服务地址
    model="Qwen/Qwen2.5-7B-Instruct",            # 使用的模型名称
    temperature=0.7                               # 控制输出随机性
)

print("=" * 60)
print("LangChain 高级记忆模块示例（详细注释版）")
print("=" * 60)

# ============================================================================
# 1. 现代摘要缓冲记忆 - 自定义实现
# ============================================================================
print("\n1. 现代摘要缓冲记忆 - 自定义实现")
print("-" * 50)

"""
摘要缓冲记忆(Summary Buffer Memory)结合了摘要记忆和缓冲记忆的优点
- 功能：当token使用量超过限制时，自动生成摘要压缩旧对话
- 优点：既保留了重要信息，又控制了内存使用
- 适用场景：长期对话，需要平衡信息保留和内存控制
- 工作原理：监控token使用量，超限时将旧对话压缩为摘要
"""

class SummaryBufferMemory:
    """
    现代摘要缓冲记忆实现
    
    这个类实现了智能的记忆管理策略：
    1. 监控当前记忆的token使用量
    2. 当超过限制时，将旧对话生成摘要
    3. 保留最近的对话和历史摘要
    4. 提供完整的上下文信息
    """

    def __init__(self, llm, max_token_limit: int = 100):
        """
        初始化摘要缓冲记忆
        
        Args:
            llm: 用于生成摘要的语言模型
            max_token_limit (int): token使用量的上限，超过时触发摘要生成
        """
        self.llm = llm                              # 语言模型实例
        self.max_token_limit = max_token_limit      # token限制
        self.chat_history = ChatMessageHistory()   # 当前对话历史
        self.summary = ""                           # 历史对话摘要

    def _estimate_tokens(self, text: str) -> int:
        """
        简单估算文本的token数量
        
        Args:
            text (str): 要估算的文本
            
        Returns:
            int: 估算的token数量
            
        Note:
            这是一个简化的估算方法，实际应用中可以使用更精确的tokenizer
            一般来说，1个token约等于4个字符（对于英文）
        """
        return len(text) // 4

    def _get_total_tokens(self) -> int:
        """
        获取当前记忆的总token数
        
        Returns:
            int: 当前记忆使用的总token数（包括摘要和当前对话）
        """
        total = 0
        
        # 计算当前对话的token数
        for msg in self.chat_history.messages:
            total += self._estimate_tokens(msg.content)
        
        # 计算摘要的token数
        if self.summary:
            total += self._estimate_tokens(self.summary)
            
        return total

    def add_conversation(self, user_msg: str, ai_msg: str):
        """
        添加对话，超过token限制时自动生成摘要
        
        Args:
            user_msg (str): 用户消息
            ai_msg (str): AI回复消息
        """
        # 添加新的对话到历史记录
        self.chat_history.add_user_message(user_msg)
        self.chat_history.add_ai_message(ai_msg)

        # 检查是否超过token限制，如果超过则生成摘要
        if self._get_total_tokens() > self.max_token_limit:
            self._create_summary()

    def _create_summary(self):
        """
        创建摘要并保留最近的对话
        
        这是记忆管理的核心方法：
        1. 将较旧的对话转换为摘要
        2. 保留最近的对话以维持上下文连续性
        3. 更新内部状态
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

            # 创建摘要提示词
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

                # 只保留最近的消息
                self.chat_history.messages = recent_messages
                print(f"生成了新的摘要，当前token数: {self._get_total_tokens()}")
            except Exception as e:
                print(f"生成摘要时出错: {e}")

    def get_context(self) -> str:
        """
        获取完整的上下文信息
        
        Returns:
            str: 包含历史摘要和最近对话的完整上下文
        """
        context = ""
        
        # 添加历史摘要（如果存在）
        if self.summary:
            context += f"对话摘要: {self.summary}\n\n"

        # 添加最近的对话
        context += "最近对话:\n"
        for msg in self.chat_history.messages:
            if isinstance(msg, HumanMessage):
                context += f"用户: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context += f"AI: {msg.content}\n"

        return context
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Returns:
            dict: 包含各种统计信息的字典
        """
        return {
            "total_tokens": self._get_total_tokens(),
            "max_token_limit": self.max_token_limit,
            "has_summary": bool(self.summary),
            "summary_length": len(self.summary) if self.summary else 0,
            "current_messages": len(self.chat_history.messages),
            "utilization": self._get_total_tokens() / self.max_token_limit
        }

# 创建摘要缓冲记忆实例
summary_buffer_memory = SummaryBufferMemory(llm, max_token_limit=100)

# 准备测试对话数据
# 这些对话模拟了一个软件工程师的自我介绍和工作讨论
long_conversation = [
    ("我是一名软件工程师，在北京工作", "很高兴认识你！软件工程师是个很有前景的职业。"),
    ("我主要使用Python和Java开发", "这两种语言都很流行，Python特别适合数据处理。"),
    ("我们公司是做金融科技的", "金融科技是个快速发展的领域，技术要求很高。"),
    ("我负责后端API开发", "后端开发是系统的核心，责任重大。"),
    ("最近在学习微服务架构", "微服务架构能提高系统的可扩展性和维护性。"),
    ("我们使用Docker和Kubernetes", "容器化技术确实能简化部署和管理。"),
    ("你还记得我的工作地点吗？", "让我回忆一下...")
]

print("添加对话历史...")
for i, (user_msg, ai_msg) in enumerate(long_conversation, 1):
    summary_buffer_memory.add_conversation(user_msg, ai_msg)
    stats = summary_buffer_memory.get_memory_stats()
    print(f"第{i}轮对话后 - Token使用率: {stats['utilization']:.2f}")

# 获取并显示记忆内容
print("\n摘要缓冲记忆内容:")
print(summary_buffer_memory.get_context())

# 显示详细统计信息
stats = summary_buffer_memory.get_memory_stats()
print(f"\n记忆统计信息:")
print(f"总Token数: {stats['total_tokens']}/{stats['max_token_limit']}")
print(f"使用率: {stats['utilization']:.2%}")
print(f"有摘要: {stats['has_summary']}")
print(f"当前消息数: {stats['current_messages']}")
