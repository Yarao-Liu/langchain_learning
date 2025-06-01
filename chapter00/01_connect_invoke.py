import os

import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# 加载.env
dotenv.load_dotenv()
# 获取 环境变量
api_key = os.getenv("OPENAI_API_KEY")
# 初始化模型配置
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1/",
    model= "Qwen/Qwen2.5-7B-Instruct"
)
# 创建提示词模板
prompt = ChatPromptTemplate.from_template("请根据下面的主题做一首诗:{topic}")
# 输出格式化
output_parser = StrOutputParser()
# 链式调用
chain = prompt | llm | output_parser
# 提交提示词
result = chain.invoke({"topic": "康师傅绿茶"})
# 打印输出
print(result)
