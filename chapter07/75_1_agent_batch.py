import json
import os

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 加载环境变量
dotenv.load_dotenv()

# 检查 API 密钥配置
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("错误: 未找到 OPENAI_API_KEY 环境变量")
    print("请在 .env 文件中设置: OPENAI_API_KEY=your_api_key_here")
    exit(1)

# 创建大语言模型实例
# 使用 SiliconFlow 提供的 API 接口，兼容 OpenAI 格式
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1/",  # SiliconFlow API 地址
    model="Qwen/Qwen2.5-72B-Instruct",  # 使用通义千问模型
    temperature=0.3  # 设置创造性参数
)
output_parser = StrOutputParser()

list = [
    "减肥哪有那么容易？我的每块肉都有它的脾气！",
    "我消极的对待减肥，能不能取消我胖子的资格啊！",
    "减肥简直是世界上最反人类的事情，不吃饭饿得想打人，可吃完饭又想打自己。",
    "当一两个人说我胖的时候，我不以为然，后来越来越多的人说我胖，这个时候我终于意识到了事情的严重性，这个世界上的骗子真是越来越多了。",
    "其实我小时候挺瘦的，后来上学了，一句“谁知盘中餐，粒粒皆辛苦”让我变成了如今这副模样。",
    "从来都不用化妆品，我保持年轻的秘诀就是，谎报年龄。",
    "妈妈说不能交不三不四的朋友，所以我的朋友都很二。",
    "做坏事早晚都会被发现，深思熟虑之后，我都改中午做。",
    "你想一夜暴富吗？你想一夜资产过亿吗？不如和我在一起，我们一起想。",
    "别看我平时对你总是漠不关心的样子，其实背底下说了你好多坏话。",
    "没钱的日子来找我，我来告诉你一个馒头，怎么分两天吃？",
    "我每晚都会对自己说：熬夜会死，事实证明我真的不怕死",
    "如果你有喜欢的女生，就送她一支口红吧，至少她亲别人的时候，你还有参与感。",
    "你瘦的时候在我心里，后来胖了，卡在里面出不来了"
]
assistant_prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": "{input} 你是一个讲笑话的高手，请分析上面的段子使用了什么技巧"
        }
    ]
)
chain = ({"input": RunnablePassthrough()} | assistant_prompt | llm | output_parser)

result = chain.batch(list)
print(result)

jsonList= []
for index in range(len(list)):
    jsonList.append({"input":list[index],"skill":result[index]})

print(jsonList)

with open("data/joke.json", "w", encoding="utf-8") as f:
    json.dump(jsonList, f,ensure_ascii=False, indent=4)
