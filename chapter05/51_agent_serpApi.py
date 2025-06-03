# serpApi
import os

import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
# 加载.env
dotenv.load_dotenv()
# 获取 环境变量
api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")

# 检查是否有SerpAPI密钥
if not serpapi_key:
    print("错误：未找到SERPAPI_API_KEY环境变量")
    print("请按照以下步骤获取并配置SerpAPI密钥：")
    print("1. 访问 https://serpapi.com/ 注册账户")
    print("2. 获取API密钥")
    print("3. 在.env文件中添加：SERPAPI_API_KEY=你的密钥")
    print("4. 或者直接在代码中传递密钥：SerpAPIWrapper(serpapi_api_key='你的密钥')")
    exit(1)

# 初始化模型配置
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1/",
    model= "Qwen/Qwen2.5-7B-Instruct"
)

# 初始化SerpAPI搜索工具
search = SerpAPIWrapper(serpapi_api_key=serpapi_key)

print(search.run("刘亦菲最近有什么活动?"))

