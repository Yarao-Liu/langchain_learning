# 加载环境变量
import os

import dotenv
import requests
from langchain.agents import AgentOutputParser
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.utils.json import parse_json_markdown
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import MessageGraph

dotenv.load_dotenv()

# 检查 API 密钥
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("错误: 未找到 OPENAI_API_KEY 环境变量")
    print("请在 .env 文件中设置: OPENAI_API_KEY=your_api_key_here")
    exit(1)

# 创建 LLM
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1/",
    model="Qwen/Qwen2.5-72B-Instruct",
    temperature=0.3
)

@tool
def searxng_search(query: str) -> str:
    """使用SearxNG搜索互联网"""
    SEARXNG_URL = "http://localhost:6688"
    params = {}
    params["q"]=query
    params["format"]= "json"
    params["engines"]="bing"
    response = requests.get(url=SEARXNG_URL, params=params)

    if response.status_code == 200:
        res = response.json()
        resList = []
        for item in res["results"]:
            resList.append({
                "title":item["title"],
                "content":item["content"],
                "url":item["url"]
            })
            if  len(resList)>= 3:
                break
        return resList
    else:
        response.raise_for_status()
print(searxng_search.invoke("成龙电影"))

promptTemplate = """
尽可能的帮助用户回答任何问题。
你可以使用以下工具来帮忙解决问题，如何已经知道了答案，也可以直接回答；
searxng_search: searxng_search(query) -> 输入搜索内容，使用searxng搜索互联网。
回复我时，请以以下两种格式之一输出回复：
------------------------------------------------
选项1：如果你希望人类使用工具，请使用此选项。
采用以下JSON模式格式化的回复内容，回复的格式里不要有注释内容：
'''json
{{
    "reason":string,
    "action": "searxng_search",
    "action_input": string
}}
''''

选项2：如果你已经知道了答案或者已经通过使用工具找到了答案，想直接对人类作出反应，请使用此选项。
采用以下JSON模式格式化的回复内容，回复的格式里不要有注释内容：
'''json
{{
    "action": "Final Answer",
    "answer": string
}}
''''
用户输入：
----------------------------------------------------
这是用户的输入（请记住通过单个选项，以JS0N模式格式化的回复内容，回复的格式里不要有注释内容，不要回复其他内容):
{input}
"""

prompt = ChatPromptTemplate.from_messages([
    {
        "role":"system",
        "content":"你是非常强大的助手，你可以使用各种工具来完成人类交给的问题和任务。"
    },
    {
        "role":"user",
        "content":promptTemplate
    }
])

class JSONAgentOutputParser(AgentOutputParser):
    def parse(self, text: str):
        try:
            response = parse_json_markdown(text)
            if isinstance(response,list):
                print("Got multiple action resoponses:%s",response)
                response= response[0]
            if response["action"] == "Final Answer":
                return AgentFinish({"output":response["answer"]})
            else:
                return AgentAction(
                    tool=response["action"],
                    tool_input=response.get("action_input",{}),
                    log=text
                )
        except Exception as e:
            raise OutputParserException(f"Failed to parse agent output: {text}") from e
    @property
    def _type(self):
        return  "json-agent"

output_parser = StrOutputParser()
chain1 = prompt | llm
msg = chain1.invoke({"input":"小米su7的发布时间"})
print(msg)

msg = chain1.invoke({"input":"请用中文讲个笑话"})
print(msg)

promptTemplate = """
{observation}
请根据浏览器的响应回答下面的问题:
{input}
"""
prompt2 = ChatPromptTemplate.from_messages([
    {
        "role":"system",
        "content":"你是一个非常强大的助手，可以使用各种工具来完成人类交给的问题和任务。"
    },
    {
        "role":"user",
        "content":promptTemplate
    }
])
chain2 = prompt2 | llm
# print(chain2.invoke({"input":"小米su7的发布时间","observation":searxng_search.invoke("小米su7的发布时间")}))
graph = MessageGraph()
graph.add_node("chain",chain1)
graph.add_edge("chain",END)
graph.set_entry_point("chain")

runnable1 = graph.compile()

print(runnable1.invoke(HumanMessage(content="小米su7的发布时间")))
# from IPython.display import Image
# Image(runnable1.get_graph().draw_png())