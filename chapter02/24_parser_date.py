from langchain.output_parsers import DatetimeOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 初始化Ollama大语言模型
# 使用Qwen 2.5 7B模型，这是一个支持中文的模型
# base_url指定Ollama服务的地址，默认是本地服务
llm = OllamaLLM(
    model="qwen2.5:7b",  # 使用Qwen 2.5 7B模型
    base_url="http://localhost:11434",  # Ollama服务地址，可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 创建输出解析器
output_parser = DatetimeOutputParser()

# 获取解析器的格式说明
format_instructions = output_parser.get_format_instructions()

template = """
请回答以下问题，并严格按照格式要求输出：
{question}

{format_instructions}
"""

# 创建提示模板
prompt = PromptTemplate(
    template=template,
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions},
)

# 创建处理链
chain = prompt | llm | output_parser

# 调用链并打印结果
msg = chain.invoke({"question": "比特币是什么时候创立的？"})
print(msg)


