"""
RunnablePassthrough示例
这个脚本展示了如何使用LangChain的RunnablePassthrough功能来处理和转换数据。
主要功能包括：
1. 数据传递和转换
2. 并行处理
3. 自定义函数处理
4. 链式调用示例
"""

from operator import itemgetter  # 用于从字典中获取特定键的值

from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda  # 并行运行器、数据传递器和Lambda函数包装器
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # Ollama的嵌入模型和语言模型

# 创建一个并行运行器示例
# RunnablePassthrough() 用于传递输入数据
# RunnablePassthrough.assign() 用于在传递过程中添加新的计算
# lambda 函数用于直接处理输入数据
runnable = RunnableParallel(
    passed=RunnablePassthrough(),  # 直接传递输入
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),  # 传递并计算新值
    modified=lambda x: x["num"] + 1  # 直接修改输入值
)
print(runnable.invoke({"num": 3}))  # 测试并行运行器

# 初始化Ollama嵌入模型
# 使用中文优化的BGE模型来生成文本的向量表示
embedding = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")

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

# 定义计算文本长度的函数
def length_function(text):
    """计算输入文本的长度"""
    return len(text)

# 定义计算两个文本长度之和的函数
def _multiple_length_function(text1, text2):
    """计算两个文本长度之和"""
    return len(text1) + len(text2)

# 定义处理字典输入的包装函数
def multiple_length_function(_dict):
    """从字典中提取两个文本并计算它们的长度之和"""
    return _multiple_length_function(_dict["text1"], _dict["text2"])

# 创建提示模板
# 用于生成简单的数学问题
prompt = ChatPromptTemplate.from_template("what is {a} + {b}")
chain1 = prompt | llm  # 创建简单的处理链

# 创建复杂的处理链
# 使用字典形式定义处理步骤
chain = (
        {
            # 计算第一个输入文本的长度
            "a": itemgetter("foo") | RunnableLambda(length_function),
            # 计算两个输入文本的长度之和
            "b": {"text1": itemgetter("foo"), "text2": itemgetter("bar")} | RunnableLambda(multiple_length_function)
        }
        | prompt  # 将处理结果传入提示模板
        | llm  # 使用语言模型生成回答
)

# 测试处理链
msg = chain.invoke({"foo": "bar", "bar": "gah"})
print(msg)
