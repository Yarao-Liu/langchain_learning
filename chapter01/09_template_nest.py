# 导入消息类型
# HumanMessage: 表示用户发送的消息
# AIMessage: 表示AI助手的回复
# SystemMessage: 表示系统级别的提示或指令
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# StrOutputParser: 用于将模型输出解析为字符串
from langchain_core.output_parsers import StrOutputParser
# 导入提示模板相关类
# ChatPromptTemplate: 用于创建聊天格式的提示模板
# HumanMessagePromptTemplate: 用于创建人类消息的模板
# MessagesPlaceholder: 用于在模板中插入消息列表
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
# OllamaLLM: 用于调用本地Ollama大语言模型
from langchain_ollama import OllamaLLM

# 创建系统提示模板
# 系统提示用于设置对话的整体上下文和风格
system_prompt = "愿{subject}与你同在"
system_message = SystemMessage(content=system_prompt)

# 创建人类消息模板
# 这个模板用于生成用户的消息
# {count} 是一个变量，将在运行时被替换为具体的数字
human_prompt = "用{count}字总结我们迄今为止的对话"
human_msg_template = HumanMessagePromptTemplate.from_template(human_prompt)

# 创建聊天提示模板
# 模板嵌套说明：
# 1. 最外层是 ChatPromptTemplate，它管理整个对话流程
# 2. 内部包含三个部分：
#    - system_message: 系统级别的提示
#    - MessagesPlaceholder: 用于插入历史对话
#    - human_msg_template: 用户消息模板
# 这种嵌套结构允许我们：
# - 保持系统提示的一致性
# - 动态插入历史对话
# - 灵活处理用户输入
chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="conversation"),
    human_msg_template
])

# 创建对话历史
# 这部分模拟了之前的对话内容
# 包括用户的问题和AI的回答
human_message = HumanMessage(content="学习编程最好的方法是什么?")
ai_message = AIMessage(
    content="1.选择编程语言：决定你想要学习的编程语言\n"
            "2.从基础开始：熟悉变量、数据类型和控制结构等基本编程概念\n"
            "3.练习、练习、再练习：学习编程的最好方法是通过实践"
)

# 格式化提示
# 将所有模板和变量组合成完整的消息列表
# 参数说明：
# - subject: 替换系统提示中的{subject}
# - conversation: 插入历史对话
# - count: 替换用户消息模板中的{count}
messages = chat_prompt.format_messages(
    subject="原力",
    conversation=[human_message, ai_message],
    count="10"
)
print("格式化后的消息：")
print(messages)

# 初始化Ollama大语言模型
# 使用Qwen 2.5 7B模型，这是一个支持中文的模型
llm = OllamaLLM(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",  # 可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 创建输出解析器
# 用于将模型的输出转换为字符串格式
output_parser = StrOutputParser()

# 创建处理链
# 处理链的组成：
# 1. chat_prompt: 生成格式化的提示
# 2. llm: 处理提示并生成回复
# 3. output_parser: 解析模型输出
chain = chat_prompt | llm | output_parser

# 执行处理链
# 提供所有必要的参数：
# - subject: 系统提示的主题
# - conversation: 历史对话内容
# - count: 用户消息中的字数限制
result = chain.invoke({
    "subject": "原力",
    "conversation": [human_message, ai_message],
    "count": "10"
})
print("\n模型输出：")
print(result)