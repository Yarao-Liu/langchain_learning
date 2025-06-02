"""
自定义生成器示例
这个脚本展示了如何使用LangChain创建自定义的流式处理生成器。
主要功能：
1. 创建CSV格式的交通工具列表生成器
2. 实现流式处理和分块输出
3. 自定义列表分割处理
4. 展示流式处理链的使用
5. 支持异步处理
"""

from typing import Iterator, List, AsyncIterator  # 用于类型提示
import asyncio  # 用于异步操作

from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_ollama import OllamaLLM  # Ollama语言模型

# 创建输出解析器实例
# 用于将模型输出转换为字符串格式
output_parser = StrOutputParser()

# 初始化Ollama大语言模型
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 3B模型
    base_url="http://localhost:11434",  # Ollama服务地址
)

# 创建提示模板
# 要求模型以CSV格式返回交通工具列表
prompt = ChatPromptTemplate.from_template("响应以csv的格式返回中文列表，不要返回其他任何内容。请输出与{transport}类似的交通工具")

# 创建基本的处理链
# 将提示模板、语言模型和输出解析器组合在一起
str_chain = prompt | llm | output_parser

# 注释掉的测试代码
# print(str_chain.invoke({"transport": "飞机"}))  # 测试基本调用
# for chunk in str_chain.stream({"transport": "飞机"}):  # 测试流式输出
#     print(chunk, end="",flush=True)

def split_into_list(input: Iterator[str]) -> Iterator[List[str]]:
    """
    将输入的字符串流分割成列表
    
    参数:
        input: 输入的字符串迭代器
        
    返回:
        包含分割后字符串的列表迭代器
    """
    buffer = ""  # 用于存储未处理的字符串
    for chunk in input:
        buffer += chunk  # 将新的块添加到缓冲区
        while "," in buffer:  # 当缓冲区中包含逗号时
            comma_index = buffer.index(",")  # 找到第一个逗号的位置
            yield [buffer[:comma_index].strip()]  # 生成逗号前的部分（去除空白）
            buffer = buffer[comma_index + 1:]  # 更新缓冲区，移除已处理的部分
    yield [buffer.strip()]  # 生成最后剩余的部分

# 创建完整的处理链
# 将字符串处理链与列表分割函数组合
list_chain = str_chain | split_into_list

# 测试流式处理
# 使用stream方法获取流式输出
for chunk in list_chain.stream({"transport": "飞机"}):
    print(chunk, end="", flush=True)  # 实时打印每个块

async def asplit_into_list(input: AsyncIterator[str]) -> AsyncIterator[List[str]]:
    """
    异步版本：将输入的字符串流分割成列表
    
    参数:
        input: 输入的异步字符串迭代器
        
    返回:
        包含分割后字符串的异步列表迭代器
    """
    buffer = ""
    async for chunk in input:
        buffer += chunk
        while "," in buffer:
            comma_index = buffer.index(",")
            yield [buffer[:comma_index].strip()]
            buffer = buffer[comma_index + 1:]
    yield [buffer.strip()]

# 创建异步处理链
list_chain2 = str_chain | asplit_into_list

async def process_async():
    """
    异步处理函数
    用于处理异步流式输出
    """
    # 使用astream而不是stream来获取异步流
    async for chunk in list_chain2.astream({"transport": "火车"}):
        print(chunk, flush=True)

# 运行异步处理
if __name__ == "__main__":
    asyncio.run(process_async())