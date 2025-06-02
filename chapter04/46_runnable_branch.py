"""
分类器示例
这个脚本展示了如何使用LangChain创建一个简单的文本分类器。
主要功能：
1. 将用户问题分类为"LangChain"、"OpenAI"或"其他"
2. 根据分类结果路由到不同的处理链
3. 使用不同的专家角色回答不同类型的问题
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_ollama import OllamaLLM

# 创建分类提示模板
prompt = PromptTemplate.from_template("""
你是一个专业的AI助手，需要将用户问题分类为以下三个类别之一：
- LangChain：与LangChain框架相关的问题
- OpenAI：与OpenAI API或服务相关的问题
- 其他：不属于上述两类的问题

请仔细分析下面的问题，并只返回一个分类结果（LangChain、OpenAI或其他）。

问题：{question}

分类结果：
""")

# 创建输出解析器实例
output_parser = StrOutputParser()

# 初始化Ollama大语言模型
# 使用Qwen 2.5 3B模型，这是一个支持中文的轻量级模型
# 适合在本地运行，响应速度快
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 3B模型
    base_url="http://localhost:11434",  # Ollama服务地址
)

# 创建分类链
chain = prompt | llm | output_parser

# 创建LangChain专家回答模板
langchainPrompt = PromptTemplate.from_template("""
您是langchain方面的专家。回答问题时始终以"正如老陈告诉我的那样"开头。
回答以下问题:
问题：{question}
回答：
""")
langchain_chain = langchainPrompt | llm

# 创建OpenAI专家回答模板
openaiPrompt = PromptTemplate.from_template("""
您是openai方面的专家。回答问题时始终以"正如山姆奥特曼告诉我的那样"开头。
回答以下问题:
问题：{question}
回答：
""")
openai_chain = openaiPrompt | llm

# 创建通用回答模板
generalPrompt = PromptTemplate.from_template("""
回答以下问题:
问题: {question}
回答：
""")
general_chain = generalPrompt | llm

# 定义路由函数
def router(info):
    """根据分类结果选择对应的处理链"""
    if "OpenAI" in info["topic"]:
        return openai_chain
    elif "LangChain" in info["topic"]:
        return langchain_chain
    else:
        return general_chain

# 创建完整的处理链
full_chain = {
    "topic": chain,
    "question": lambda x: x["question"]
} | RunnableLambda(router)

# 测试不同的问题
test_questions = [
    "how do I call OpenAI?",
    "how to use LangChain for text generation?",
    "what is the weather today?",
    "how to use OpenAI API with LangChain?",
    "what is the best way to implement RAG with LangChain?"
]

# 运行测试并打印结果
print("分类和回答测试结果：")
print("-" * 50)
for question in test_questions:
    print(f"问题: {question}")
    result = full_chain.invoke({"question": question})
    print(f"回答: {result}")
    print("-" * 50)
