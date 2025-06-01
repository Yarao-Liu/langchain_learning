from langchain_core.output_parsers import CommaSeparatedListOutputParser
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
output_parser = CommaSeparatedListOutputParser()

# 定义格式说明
format_instructions = "您的响应应该是csv格式的逗号分隔值的列表。例如：内容1,内容2,内容3"

# 创建提示模板
prompt = PromptTemplate(
    template="请列举5个{format_instructions}。\n{subject}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)
# 创建处理链
chain = prompt | llm | output_parser

# 调用链并打印结果
msg = chain.invoke({"subject": "冰淇淋口味"})
print(msg)