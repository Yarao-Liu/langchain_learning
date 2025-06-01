from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 创建完整的提示模板
# 使用 f-string 风格的变量占位符
full_template = """你是一个严格按照模板格式回答问题的AI助手。

{introduction}

{example}

{start}

请严格按照以上模板格式回答，保持相同的问答结构。"""

# 创建各个子模板
# 1. 介绍部分：定义角色
introduction_template = "假如你是{person}，请用{person}的语气和风格回答问题。"
introduction_prompt = PromptTemplate.from_template(introduction_template)

# 2. 示例部分：提供交互示例
example_template = """
下面是一个交互示例，请严格按照这个格式回答：
Q: {example_q}
A: {example_a}
"""
example_prompt = PromptTemplate.from_template(example_template)

# 3. 开始部分：用户输入
start_template = """
现在请回答以下问题：
Q: {input}
A: """
start_prompt = PromptTemplate.from_template(start_template)

# 创建最终的提示模板
final_prompt = PromptTemplate.from_template(full_template)

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
# 1. final_prompt: 生成格式化的提示
# 2. llm: 处理提示并生成回复
# 3. output_parser: 解析模型输出
chain = final_prompt | llm | output_parser

# 准备输入变量
input_variables = {
    "input": "您最喜欢的社交媒体网站是什么",
    "person": "Elon Musk",
    "example_q": "你最喜欢什么车?",
    "example_a": "Tesla"
}

# 首先格式化各个子模板
introduction = introduction_prompt.format(person=input_variables["person"])
example = example_prompt.format(
    example_q=input_variables["example_q"],
    example_a=input_variables["example_a"]
)
start = start_prompt.format(input=input_variables["input"])

# 打印完整的提示模板，用于调试
print("完整的提示模板：")
print(final_prompt.format(
    introduction=introduction,
    example=example,
    start=start
))

# 然后使用格式化后的子模板调用处理链
msg = chain.invoke({
    "introduction": introduction,
    "example": example,
    "start": start
})

print("\n模型输出：")
print(msg)

