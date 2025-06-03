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
print("LangChain 高级记忆模块示例")
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

# ============================================================================
# 2. 带时间戳记忆 - 为每条消息添加时间戳
# ============================================================================
print("\n\n2. 带时间戳记忆 - 为每条消息添加时间戳")
print("-" * 50)

"""
带时间戳记忆(Timestamped Memory)为每条消息添加时间戳
- 功能：记录每条消息的确切时间，便于追踪对话时间线
- 优点：可以基于时间进行消息筛选和分析
- 适用场景：需要时间追踪的对话系统，客服系统，会议记录
- 工作原理：在消息中嵌入时间戳信息，支持时间范围查询
"""

class TimestampedMemory:
    """
    带时间戳的记忆管理类

    这个类为每条消息添加时间戳，支持：
    1. 自动为新消息添加时间戳
    2. 基于时间范围查询消息
    3. 按时间排序和筛选
    4. 时间统计分析
    """

    def __init__(self):
        """初始化带时间戳的记忆"""
        self.messages: List[Dict[str, Any]] = []  # 存储带时间戳的消息

    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None):
        """
        添加带时间戳的消息

        Args:
            role (str): 消息角色 ('user' 或 'ai')
            content (str): 消息内容
            timestamp (datetime, optional): 自定义时间戳，默认使用当前时间
        """
        if timestamp is None:
            timestamp = datetime.now()

        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "formatted_time": timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }

        self.messages.append(message)
        print(f"[{message['formatted_time']}] {role}: {content}")

    def get_messages_in_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        获取指定时间范围内的消息

        Args:
            start_time (datetime): 开始时间
            end_time (datetime): 结束时间

        Returns:
            List[Dict]: 时间范围内的消息列表
        """
        return [
            msg for msg in self.messages
            if start_time <= msg["timestamp"] <= end_time
        ]

    def get_recent_messages(self, minutes: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近N分钟的消息

        Args:
            minutes (int): 时间范围（分钟）

        Returns:
            List[Dict]: 最近的消息列表
        """
        from datetime import timedelta
        now = datetime.now()
        start_time = now - timedelta(minutes=minutes)
        return self.get_messages_in_range(start_time, now)

    def get_conversation_timeline(self) -> str:
        """
        获取对话时间线的格式化字符串

        Returns:
            str: 格式化的对话时间线
        """
        if not self.messages:
            return "暂无对话记录"

        timeline = "对话时间线:\n"
        timeline += "=" * 50 + "\n"

        for msg in self.messages:
            timeline += f"[{msg['formatted_time']}] {msg['role'].upper()}: {msg['content']}\n"

        return timeline

    def get_time_statistics(self) -> Dict[str, Any]:
        """
        获取时间统计信息

        Returns:
            dict: 包含时间统计的字典
        """
        if not self.messages:
            return {"total_messages": 0}

        timestamps = [msg["timestamp"] for msg in self.messages]

        return {
            "total_messages": len(self.messages),
            "first_message": min(timestamps).strftime("%Y-%m-%d %H:%M:%S"),
            "last_message": max(timestamps).strftime("%Y-%m-%d %H:%M:%S"),
            "conversation_duration": str(max(timestamps) - min(timestamps)),
            "user_messages": len([m for m in self.messages if m["role"] == "user"]),
            "ai_messages": len([m for m in self.messages if m["role"] == "ai"])
        }

# 创建带时间戳记忆实例并测试
timestamped_memory = TimestampedMemory()

print("添加带时间戳的对话...")
import time

# 模拟不同时间的对话
timestamped_memory.add_message("user", "你好，我想了解Python编程")
time.sleep(1)  # 模拟时间间隔
timestamped_memory.add_message("ai", "你好！我很乐意帮助你学习Python编程")
time.sleep(1)
timestamped_memory.add_message("user", "Python有哪些主要特点？")
time.sleep(1)
timestamped_memory.add_message("ai", "Python具有简洁易读、功能强大、生态丰富等特点")

# 显示对话时间线
print("\n" + timestamped_memory.get_conversation_timeline())

# 显示时间统计
stats = timestamped_memory.get_time_statistics()
print("时间统计信息:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# 获取最近消息
recent = timestamped_memory.get_recent_messages(minutes=1)
print(f"\n最近1分钟的消息数量: {len(recent)}")

# ============================================================================
# 3. 持久化记忆 - 将对话历史保存到文件系统
# ============================================================================
print("\n\n3. 持久化记忆 - 将对话历史保存到文件系统")
print("-" * 50)

"""
持久化记忆(Persistent Memory)将对话历史保存到文件系统
- 功能：支持程序重启后恢复对话历史
- 优点：数据不会因程序关闭而丢失，支持长期存储
- 适用场景：生产环境，需要数据持久化的应用
- 工作原理：定期将内存中的对话保存到JSON文件，启动时自动加载
"""

class PersistentMemory:
    """
    持久化记忆管理类

    这个类提供数据持久化功能：
    1. 自动保存对话到文件
    2. 程序启动时自动加载历史数据
    3. 支持多种存储格式
    4. 提供数据备份和恢复功能
    """

    def __init__(self, file_path: str = "data/memory_data.json"):
        """
        初始化持久化记忆

        Args:
            file_path (str): 数据文件路径
        """
        self.file_path = file_path
        self.messages: List[Dict[str, Any]] = []

        # 确保数据目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 加载已有数据
        self.load_from_file()

    def add_message(self, role: str, content: str, auto_save: bool = True):
        """
        添加消息并可选择自动保存

        Args:
            role (str): 消息角色
            content (str): 消息内容
            auto_save (bool): 是否自动保存到文件
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "id": len(self.messages) + 1
        }

        self.messages.append(message)
        print(f"添加消息: {role} - {content}")

        if auto_save:
            self.save_to_file()

    def save_to_file(self):
        """
        保存对话历史到文件
        """
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "messages": self.messages,
                    "metadata": {
                        "total_messages": len(self.messages),
                        "last_updated": datetime.now().isoformat(),
                        "version": "1.0"
                    }
                }, f, ensure_ascii=False, indent=2)
            print(f"数据已保存到: {self.file_path}")
        except Exception as e:
            print(f"保存数据时出错: {e}")

    def load_from_file(self):
        """
        从文件加载对话历史
        """
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 兼容不同的数据格式
                    if isinstance(data, dict):
                        self.messages = data.get("messages", [])
                    elif isinstance(data, list):
                        self.messages = data
                    else:
                        self.messages = []
                    print(f"从文件加载了 {len(self.messages)} 条消息")
            else:
                print("数据文件不存在，创建新的记忆")
        except Exception as e:
            print(f"加载数据时出错: {e}")
            self.messages = []

    def clear_memory(self, backup: bool = True):
        """
        清空记忆，可选择备份

        Args:
            backup (bool): 是否在清空前创建备份
        """
        if backup and self.messages:
            backup_path = f"{self.file_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump({"messages": self.messages}, f, ensure_ascii=False, indent=2)
                print(f"备份已创建: {backup_path}")
            except Exception as e:
                print(f"创建备份时出错: {e}")

        self.messages = []
        self.save_to_file()
        print("记忆已清空")

    def get_conversation_history(self, limit: Optional[int] = None) -> str:
        """
        获取对话历史的格式化字符串

        Args:
            limit (int, optional): 限制返回的消息数量

        Returns:
            str: 格式化的对话历史
        """
        messages_to_show = self.messages[-limit:] if limit else self.messages

        if not messages_to_show:
            return "暂无对话历史"

        history = "对话历史:\n"
        history += "=" * 50 + "\n"

        for msg in messages_to_show:
            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
            history += f"[{timestamp}] {msg['role'].upper()}: {msg['content']}\n"

        return history

    def get_storage_info(self) -> Dict[str, Any]:
        """
        获取存储信息

        Returns:
            dict: 存储相关信息
        """
        file_size = 0
        if os.path.exists(self.file_path):
            file_size = os.path.getsize(self.file_path)

        return {
            "file_path": self.file_path,
            "file_exists": os.path.exists(self.file_path),
            "file_size_bytes": file_size,
            "file_size_kb": round(file_size / 1024, 2),
            "total_messages": len(self.messages),
            "last_message_time": self.messages[-1]["timestamp"] if self.messages else None
        }

# 创建持久化记忆实例并测试
persistent_memory = PersistentMemory()

print("添加新的对话到持久化记忆...")
persistent_memory.add_message("user", "什么是机器学习？")
persistent_memory.add_message("ai", "机器学习是人工智能的一个分支，让计算机能够从数据中学习")
persistent_memory.add_message("user", "有哪些常见的机器学习算法？")
persistent_memory.add_message("ai", "常见的包括线性回归、决策树、随机森林、神经网络等")

# 显示对话历史
print("\n" + persistent_memory.get_conversation_history())

# 显示存储信息
storage_info = persistent_memory.get_storage_info()
print("\n存储信息:")
for key, value in storage_info.items():
    print(f"  {key}: {value}")

print(f"\n数据文件位置: {persistent_memory.file_path}")
print("注意: 程序重启后，这些对话将自动恢复")

# ============================================================================
# 4. 智能记忆筛选 - 基于消息重要性进行智能筛选和管理
# ============================================================================
print("\n\n4. 智能记忆筛选 - 基于消息重要性进行智能筛选和管理")
print("-" * 50)

"""
智能记忆筛选(Smart Memory Filtering)基于消息重要性进行智能筛选
- 功能：自动识别和保留重要消息，过滤掉不重要的内容
- 优点：提高记忆质量，减少噪音，保留关键信息
- 适用场景：复杂对话场景，需要精确内容管理的应用
- 工作原理：使用关键词匹配、长度分析、语义重要性等多种策略
"""

class SmartMemoryFilter:
    """
    智能记忆筛选管理类

    这个类提供智能筛选功能：
    1. 基于关键词的重要性评分
    2. 基于消息长度的筛选
    3. 基于对话上下文的重要性判断
    4. 自定义筛选规则
    """

    def __init__(self, llm=None):
        """
        初始化智能记忆筛选器

        Args:
            llm: 可选的语言模型，用于语义分析
        """
        self.llm = llm
        self.messages: List[Dict[str, Any]] = []

        # 重要关键词列表（可以根据应用场景自定义）
        self.important_keywords = [
            "重要", "关键", "核心", "主要", "必须", "需要", "问题", "解决",
            "错误", "bug", "修复", "优化", "改进", "建议", "方案", "计划",
            "项目", "任务", "目标", "结果", "数据", "分析", "总结", "结论"
        ]

        # 不重要的短语模式
        self.unimportant_patterns = [
            "你好", "谢谢", "不客气", "再见", "好的", "嗯", "哦", "啊",
            "是的", "没问题", "可以", "明白了", "知道了"
        ]

    def calculate_importance_score(self, content: str) -> float:
        """
        计算消息的重要性评分

        Args:
            content (str): 消息内容

        Returns:
            float: 重要性评分 (0-1之间)
        """
        score = 0.0
        content_lower = content.lower()

        # 1. 基于长度的评分（较长的消息通常更重要）
        length_score = min(len(content) / 100, 0.3)  # 最多0.3分
        score += length_score

        # 2. 基于重要关键词的评分
        keyword_score = 0
        for keyword in self.important_keywords:
            if keyword in content_lower:
                keyword_score += 0.1
        keyword_score = min(keyword_score, 0.4)  # 最多0.4分
        score += keyword_score

        # 3. 检查是否包含不重要的模式
        unimportant_penalty = 0
        for pattern in self.unimportant_patterns:
            if pattern in content_lower and len(content) < 20:
                unimportant_penalty = 0.3
                break
        score -= unimportant_penalty

        # 4. 基于问号和感叹号的评分（问题和强调通常重要）
        if '?' in content or '？' in content:
            score += 0.1
        if '!' in content or '！' in content:
            score += 0.1

        # 5. 基于数字和特殊符号的评分（可能包含重要数据）
        import re
        if re.search(r'\d+', content):
            score += 0.1

        return min(max(score, 0.0), 1.0)  # 确保分数在0-1之间

    def add_message(self, role: str, content: str):
        """
        添加消息并计算重要性评分

        Args:
            role (str): 消息角色
            content (str): 消息内容
        """
        importance_score = self.calculate_importance_score(content)

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "importance_score": importance_score,
            "id": len(self.messages) + 1
        }

        self.messages.append(message)
        print(f"添加消息 [重要性: {importance_score:.2f}] {role}: {content}")

    def get_important_messages(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        获取重要性评分超过阈值的消息

        Args:
            threshold (float): 重要性阈值

        Returns:
            List[Dict]: 重要消息列表
        """
        return [
            msg for msg in self.messages
            if msg["importance_score"] >= threshold
        ]

    def get_top_important_messages(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        获取最重要的N条消息

        Args:
            count (int): 返回的消息数量

        Returns:
            List[Dict]: 按重要性排序的消息列表
        """
        sorted_messages = sorted(
            self.messages,
            key=lambda x: x["importance_score"],
            reverse=True
        )
        return sorted_messages[:count]

    def filter_and_compress(self, max_messages: int = 10, min_importance: float = 0.3) -> List[Dict[str, Any]]:
        """
        智能筛选和压缩消息

        Args:
            max_messages (int): 最大保留消息数
            min_importance (float): 最小重要性阈值

        Returns:
            List[Dict]: 筛选后的消息列表
        """
        # 首先按重要性筛选
        important_messages = [
            msg for msg in self.messages
            if msg["importance_score"] >= min_importance
        ]

        # 如果重要消息太多，选择最重要的
        if len(important_messages) > max_messages:
            important_messages = sorted(
                important_messages,
                key=lambda x: x["importance_score"],
                reverse=True
            )[:max_messages]

        # 如果重要消息不够，补充一些较新的消息
        elif len(important_messages) < max_messages:
            remaining_count = max_messages - len(important_messages)
            recent_messages = [
                msg for msg in self.messages[-remaining_count:]
                if msg not in important_messages
            ]
            important_messages.extend(recent_messages)

        return sorted(important_messages, key=lambda x: x["id"])

    def get_memory_analysis(self) -> Dict[str, Any]:
        """
        获取记忆分析报告

        Returns:
            dict: 包含各种分析数据的字典
        """
        if not self.messages:
            return {"total_messages": 0}

        scores = [msg["importance_score"] for msg in self.messages]

        return {
            "total_messages": len(self.messages),
            "average_importance": sum(scores) / len(scores),
            "max_importance": max(scores),
            "min_importance": min(scores),
            "high_importance_count": len([s for s in scores if s >= 0.7]),
            "medium_importance_count": len([s for s in scores if 0.3 <= s < 0.7]),
            "low_importance_count": len([s for s in scores if s < 0.3]),
            "user_messages": len([m for m in self.messages if m["role"] == "user"]),
            "ai_messages": len([m for m in self.messages if m["role"] == "ai"])
        }

    def display_filtered_conversation(self, threshold: float = 0.5):
        """
        显示筛选后的对话

        Args:
            threshold (float): 重要性阈值
        """
        important_messages = self.get_important_messages(threshold)

        if not important_messages:
            print("没有找到重要性超过阈值的消息")
            return

        print(f"筛选后的重要对话 (阈值: {threshold}):")
        print("=" * 50)

        for msg in important_messages:
            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
            score = msg["importance_score"]
            print(f"[{timestamp}] [{score:.2f}] {msg['role'].upper()}: {msg['content']}")

# 创建智能记忆筛选实例并测试
smart_filter = SmartMemoryFilter(llm)

print("添加各种重要性的对话...")

# 添加不同重要性的测试消息
test_conversations = [
    ("user", "你好"),  # 低重要性
    ("ai", "你好！有什么可以帮助你的吗？"),  # 低重要性
    ("user", "我遇到了一个重要的bug，系统无法正常启动"),  # 高重要性
    ("ai", "这确实是个严重问题。请详细描述一下错误信息"),  # 高重要性
    ("user", "错误代码是500，数据库连接失败"),  # 高重要性
    ("ai", "明白了"),  # 低重要性
    ("user", "你能帮我分析一下解决方案吗？"),  # 中等重要性
    ("ai", "当然可以。建议检查数据库配置文件和网络连接状态"),  # 高重要性
    ("user", "谢谢"),  # 低重要性
    ("ai", "不客气，还有其他问题吗？")  # 低重要性
]

for role, content in test_conversations:
    smart_filter.add_message(role, content)

# 显示记忆分析
print("\n记忆分析报告:")
analysis = smart_filter.get_memory_analysis()
for key, value in analysis.items():
    print(f"  {key}: {value}")

# 显示筛选后的重要对话
print("\n")
smart_filter.display_filtered_conversation(threshold=0.5)

# 获取最重要的消息
print("\n最重要的3条消息:")
top_messages = smart_filter.get_top_important_messages(3)
for i, msg in enumerate(top_messages, 1):
    print(f"{i}. [{msg['importance_score']:.2f}] {msg['role']}: {msg['content']}")

# 智能筛选和压缩
print("\n智能筛选和压缩结果:")
filtered = smart_filter.filter_and_compress(max_messages=5, min_importance=0.3)
for msg in filtered:
    print(f"[{msg['importance_score']:.2f}] {msg['role']}: {msg['content']}")
