from enum import Enum

from langchain.output_parsers import EnumOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 初始化Ollama大语言模型
# 使用Qwen 2.5 7B模型，这是一个支持中文的模型
# base_url指定Ollama服务的地址，默认是本地服务
llm = OllamaLLM(
    model="qwen2.5:7b",  # 使用Qwen 2.5 7B模型
    base_url="http://localhost:11434",  # Ollama服务地址，可以修改为其他地址，如 "http://192.168.1.100:11434"
)


class Colour(Enum):
    RED= "红色"
    GREEN = "绿色"
    YELLOW = "黄色"
    WHITE = "白色"
parser = EnumOutputParser(enum=Colour)
prompt_template = PromptTemplate.from_template("""
    {person}的皮肤主要是什么颜色？
    {instructions}
""")
instructions = "响应的结果请选择以下选项之一:红色、绿色、黄色、白色.不要有其他内容"
prompt = prompt_template.partial(instructions=instructions)
chain = prompt | llm | parser
msg = chain.invoke({"person": "亚洲人"})

print(msg)