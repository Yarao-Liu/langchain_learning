# 导入必要的库
from typing import Iterable  # 用于类型提示，表示可迭代对象

# 导入LangChain相关组件
from langchain_core.messages import AIMessage, AIMessageChunk  # AI消息类型，用于处理模型输出
from langchain_core.runnables import RunnableGenerator  # 用于创建可运行的生成器
from langchain_ollama import OllamaLLM  # Ollama语言模型接口

# 初始化Ollama大语言模型
# 使用Qwen 2.5 3B模型，这是一个支持中文的轻量级模型
# base_url指定Ollama服务的地址，默认是本地服务
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 3B模型，适合轻量级任务
    base_url="http://localhost:11434",  # Ollama服务地址，可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 定义普通解析函数
# 这个函数将AI消息的内容转换为大小写互换的形式
def parse(ai_message: AIMessage) -> str:
    return ai_message.swapcase()  # 使用swapcase()方法互换大小写

# 创建处理链并测试普通解析
# 将语言模型和解析函数组合在一起
chain = llm | parse
msg = chain.invoke("Hello World")  # 测试普通调用
print(msg)  # 打印结果

# 测试流式输出
# 使用stream方法逐步输出结果
for chunk in chain.stream("Hello World"):
    print(chunk, end="")  # 不换行打印每个块

# 定义流式解析函数
# 这个函数处理流式输出的消息块
def streaming_parse(chunks: Iterable[AIMessageChunk]) -> Iterable[str]:
    for ch in chunks:
        yield ch.swapcase()  # 对每个块进行大小写互换

# 创建流式解析器
# 使用RunnableGenerator将生成器函数转换为可运行对象
streaming_parse = RunnableGenerator(streaming_parse)

# 创建新的处理链
# 使用流式解析器替代普通解析函数
chain = llm | streaming_parse

# 测试流式解析
msg = chain.invoke("Hello World")  # 测试流式调用
print(msg)  # 打印结果