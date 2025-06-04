# 简化版 Agent + LLM + 搜索示例
import os
import warnings

import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub

# 禁用LangSmith追踪
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# 方法1: 禁用LangSmith相关警告
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")
# 加载环境变量
dotenv.load_dotenv()

print("🔍 简化版智能搜索Agent")
print("=" * 40)

# 1. 初始化LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.siliconflow.cn/v1/",
    model="Qwen/Qwen2.5-72B-Instruct",
    temperature=0.1
)

# 2. 创建搜索工具
search_tool = Tool(
    name="search",
    description="搜索互联网获取最新信息",
    func=SearxSearchWrapper(
        searx_host="http://localhost:6688",
        k=5
    ).run
)

# 3. 创建Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, [search_tool], prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)

# 4. 测试搜索
test_questions = [
    "刘亦菲最近有什么新作品？",
    "Python 3.12有什么新特性？",
    "ChatGPT最新版本的功能"
]

for question in test_questions:
    print(f"\n问题: {question}")
    print("-" * 30)
    
    try:
        response = agent_executor.invoke({"input": question})
        print("回答:", response["output"])
    except Exception as e:
        print(f"错误: {e}")

print("\n✅ 测试完成！")

# 5. 交互模式
print("\n🎯 交互模式（输入'quit'退出）:")
while True:
    question = input("\n请输入问题: ").strip()
    if question.lower() in ['quit', 'exit', '退出']:
        break
    
    if question:
        try:
            response = agent_executor.invoke({"input": question})
            print("回答:", response["output"])
        except Exception as e:
            print(f"错误: {e}")

print("👋 再见！")
