import json
import os
import random

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("错误: 未找到 OPENAI_API_KEY 环境变量")
    print("请在 .env 文件中设置: OPENAI_API_KEY=your_api_key_here")
    exit(1)

with open("data/joke.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(data)
random_data = random.choices(data, k=3)

print(random_data)

demoList = []

for item in random_data:
    item["input"] = "讲一个学习的段子"
    demoList.append(item)

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": """
                这个段子使用了以下技巧： {skill}
                你是一个写段子的能手，请按照上面的段子的技巧，写一个搞笑的段子。
                主题是：{input}
            """
        }
    ]
)
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1/",  # SiliconFlow API 地址
    model="Qwen/Qwen2.5-72B-Instruct",  # 使用通义千问模型
    temperature=0.3  # 设置创造性参数
)
output_parser = StrOutputParser()
chain = assistant_prompt | llm | output_parser

resList = chain.batch(demoList)
print(resList)