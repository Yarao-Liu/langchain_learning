import os

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI

# 加载环境变量
dotenv.load_dotenv()
python_repl = PythonREPL()
print(python_repl.run("print(2 + 2)"))
api_key = os.getenv("OPENAI_API_KEY")
# 创建 LLM
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1/",
    model="Qwen/Qwen2.5-72B-Instruct",
    temperature=0.1
)

promptFormat = """{query}
请根据上面的问题,生成python代码,计算出问题的答案:最后计算出来的结果用print函数打印出来.请直接返回python代码,不要返回其他任何内容.
"""
def parsePythonCode(code):
    code = code.replace("```python","").replace("```","").strip()
    return code
prompt = ChatPromptTemplate.from_template(promptFormat)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser | parsePythonCode | python_repl.run

result = chain.invoke({"query":"3箱苹果,每箱10个,每个苹果2元,一共多少钱?"})
print(result)
