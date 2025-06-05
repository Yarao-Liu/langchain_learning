import os

import dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI

# 加载环境变量
dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
# 创建 LLM
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1/",
    model="Qwen/Qwen2.5-72B-Instruct",
    temperature=0.1
)

search = DuckDuckGoSearchRun()
print(search.run("蔡徐坤"))

search = DuckDuckGoSearchResults()
print(search.run("刘亦菲"))
# 配置搜索
wrapper = DuckDuckGoSearchAPIWrapper(region="de-de",max_results=2,source="news")
search = DuckDuckGoSearchResults(api_wrapper=wrapper)
print(search.run("刘亦菲"))