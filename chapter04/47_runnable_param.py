"""
方程求解示例
这个脚本展示了如何使用LangChain创建一个简单的方程求解器。
主要功能：
1. 接收自然语言描述的数学方程
2. 将自然语言转换为代数符号
3. 求解方程并返回结果
4. 使用RunnablePassthrough处理输入参数
"""

from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_core.runnables import RunnablePassthrough  # 数据传递器
from langchain_ollama import OllamaLLM  # Ollama语言模型

# 创建输出解析器实例
# 用于将模型输出转换为字符串格式
output_parser = StrOutputParser()

# 初始化Ollama大语言模型
# 使用Qwen 2.5 3B模型，这是一个支持中文的轻量级模型
# 适合在本地运行，响应速度快
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 3B模型
    base_url="http://localhost:11434",  # Ollama服务地址
)

# 创建聊天提示模板
# 使用系统消息和用户消息的组合
# 系统消息定义了输出格式和任务要求
# 用户消息包含需要求解的方程描述
prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",  # 系统角色消息
            "content": "用代数符号写出下面的方程,然后求解。使用格式\n\n方程:...\n解决方案:..."  # 定义输出格式
        },
        {
            "role": "user",  # 用户角色消息
            "content": "{equation_statement}"  # 方程描述占位符
        }
    ]
)

# 创建处理链
# 1. RunnablePassthrough() 用于传递输入参数
# 2. prompt 将输入转换为提示
# 3. llm 使用语言模型生成回答
# 4. output_parser 将输出转换为字符串
chain = {"equation_statement": RunnablePassthrough()} | prompt | llm | output_parser

# 测试方程求解
# 输入一个用自然语言描述的方程
print(chain.invoke("x的3次方加7等于34"))