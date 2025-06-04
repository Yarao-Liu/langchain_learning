# 方法1: 禁用LangSmith相关警告
import os
import warnings

import arxiv
import dotenv
from langchain_community.document_loaders.arxiv import ArxivLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")
# 禁用LangSmith追踪
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 加载环境变量
dotenv.load_dotenv()

print("🔍 搜索演示")
print("=" * 50)

# 初始化LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.siliconflow.cn/v1/",
    model="Qwen/Qwen2.5-72B-Instruct",
    temperature=0.1
)

# arxiv = ArxivAPIWrapper()
# docs = arxiv.run("1605.08386")
# print(docs)


search = arxiv.Search(
    query="gpt4",
    max_results=5,
    sort_by=arxiv.SortCriterion.Relevance,
)

client = arxiv.Client()

results= client.results(search)
papers =[]
for item in  results:
    print(item)
    papers.append(item)
print(papers)

htmlUrls = []
for item in papers:
    url= item.entry_id.replace("abs","html")
    htmlUrls.append(url)
print(htmlUrls)
docs = ArxivLoader(query="2309.12732v1",load_max_docs=5).load()

print(docs)

prompt = ChatPromptTemplate.from_template("{article} \n\n 请使用中文详细讲解上面这篇文章。并将核心内容的要点提炼出来")

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

print(chain.invoke({"article":docs[0].page_content}))