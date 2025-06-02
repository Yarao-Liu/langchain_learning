"""
链式处理示例
这个脚本展示了如何使用LangChain的@chain装饰器创建处理链。
主要功能：
1. 使用@chain装饰器创建处理链
2. 实现故事生成和优化
3. 展示链式处理流程
"""

from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_core.runnables import chain  # 链式处理装饰器
from langchain_ollama import OllamaLLM  # Ollama语言模型

# 创建第一个提示模板：生成故事
prompt1 = ChatPromptTemplate.from_template("给我讲一个关于{topic}的故事")

# 创建第二个提示模板：优化故事
prompt2 = ChatPromptTemplate.from_template("{story}\n\n对上面的故事进行修改，让故事变得更加生动、口语化")

# 创建输出解析器实例
output_parser = StrOutputParser()

# 初始化Ollama大语言模型
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 3B模型
    base_url="http://localhost:11434",  # Ollama服务地址
)

# 使用@chain装饰器创建处理链
@chain
def custom_chain(text):
    """使用@chain装饰器创建的处理链"""
    # 生成原始故事
    prompt_value = prompt1.invoke({"topic": text})
    output = llm.invoke(prompt_value)
    story = output_parser.parse(output)
    
    # 优化故事
    chain2 = prompt2 | llm | output_parser
    return chain2.invoke({"story": story})

# 测试处理链
print("=== 使用@chain装饰器的处理链 ===")
# 使用invoke方法调用处理链
result = custom_chain.invoke("有志者事竟成")
print(result)