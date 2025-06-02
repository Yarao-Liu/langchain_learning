"""
并行运行示例
这个脚本展示了如何使用LangChain的并行运行功能来同时处理多个任务。
主要功能包括：
1. 生成文章大纲
2. 生成写作建议
3. 并行处理多个任务
"""

# 导入必要的库
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # Ollama的嵌入模型和语言模型

# 初始化Ollama嵌入模型
# 使用中文优化的BGE模型来生成文本的向量表示
# 这个模型特别适合处理中文文本
embedding = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")

# 创建输出解析器实例
output_parser = StrOutputParser()

# 初始化Ollama大语言模型
# 使用Qwen 2.5 3B模型，这是一个支持中文的轻量级模型
# 适合在本地运行，响应速度快
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 3B模型
    base_url="http://localhost:11434",  # Ollama服务地址
)

# 定义大纲生成提示模板
outlinePromptTemplate = """
主题：{theme}
如果要根据主题写一篇文章，请列出文章的大纲。
"""

# 创建大纲生成提示
outlinePrompt = ChatPromptTemplate.from_template(outlinePromptTemplate)

# 定义写作建议提示模板
tipsTemplate = """
主题：{theme}
如果要根据主题写一篇文章，应该需要注意哪些方面，才能把这篇文章写好。
"""

# 创建写作建议提示
tipsPrompt = ChatPromptTemplate.from_template(tipsTemplate)

# 设置查询主题
query = "2025年中国经济走向与运行趋势"

# 创建大纲生成链
outlineChain = outlinePrompt | llm | output_parser

# 创建写作建议链
tipsChain = tipsPrompt | llm | output_parser

# 生成大纲
print("\n生成文章大纲：")
outline = outlineChain.invoke({"theme": query})
print(outline)

# 生成写作建议
print("\n生成写作建议：")
tips = tipsChain.invoke({"theme": query})
print(tips)

articlePromptTemplate = """
    主题：{theme}
    大纲：{outline}
    注意事项：{tips}
    请根据上面的主题、大纲和注意事项写出丰富的完整文章内容。
"""
articlePrompt = ChatPromptTemplate.from_template(articlePromptTemplate)
articleChain = articlePrompt | llm | output_parser
print("\n最终文章：")
msg  = articleChain.invoke({"theme": query, "outline": outline, "tips": tips})
print(msg)