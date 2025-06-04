# 搜索方式对比：直接搜索 vs Agent搜索
import os
import warnings

import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper, SerpAPIWrapper
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub
# 方法1: 禁用LangSmith相关警告
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")
# 禁用LangSmith追踪
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 加载环境变量
dotenv.load_dotenv()

print("🔍 搜索方式对比演示")
print("=" * 50)

# 初始化LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.siliconflow.cn/v1/",
    model="Qwen/Qwen2.5-72B-Instruct",
    temperature=0.1
)

# ========================================================================
# 方式1: 直接使用搜索工具
# ========================================================================

print("\n1️⃣ 方式1: 直接搜索")
print("-" * 30)

try:
    # SearxNG直接搜索
    searx_search = SearxSearchWrapper(
        searx_host="http://localhost:6688",
        k=3
    )
    
    query = "Python LangChain教程"
    print(f"搜索查询: {query}")
    
    result = searx_search.run(query)
    print("直接搜索结果:")
    print(result[:200] + "..." if len(result) > 200 else result)
    
except Exception as e:
    print(f"SearxNG搜索失败: {e}")

# 如果有SerpAPI密钥，也可以测试
serpapi_key = os.getenv("SERPAPI_API_KEY")
if serpapi_key:
    try:
        serp_search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
        result = serp_search.run(query)
        print("\nSerpAPI搜索结果:")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"SerpAPI搜索失败: {e}")

# ========================================================================
# 方式2: Agent智能搜索
# ========================================================================

print("\n\n2️⃣ 方式2: Agent智能搜索")
print("-" * 30)

try:
    # 创建搜索工具
    search_tool = Tool(
        name="search",
        description="搜索互联网获取最新信息和资料",
        func=SearxSearchWrapper(
            searx_host="http://localhost:6688",
            k=5
        ).run
    )
    
    # 创建Agent
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_functions_agent(llm, [search_tool], prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[search_tool],
        verbose=False  # 简化输出
    )
    
    # 使用Agent搜索
    print(f"Agent查询: {query}")
    response = agent_executor.invoke({"input": f"请帮我搜索关于{query}的信息，并总结要点"})
    
    print("Agent智能回答:")
    print(response["output"])
    
except Exception as e:
    print(f"Agent搜索失败: {e}")

# ========================================================================
# 方式3: 多工具Agent
# ========================================================================

print("\n\n3️⃣ 方式3: 多工具Agent")
print("-" * 30)

try:
    # 创建多个专门的搜索工具
    general_tool = Tool(
        name="general_search",
        description="搜索一般信息和新闻",
        func=SearxSearchWrapper(
            searx_host="http://localhost:6688",
            engines=["google", "bing"],
            k=3
        ).run
    )
    
    tech_tool = Tool(
        name="tech_search", 
        description="搜索技术和编程相关信息",
        func=SearxSearchWrapper(
            searx_host="http://localhost:6688",
            engines=["github", "stackoverflow"],
            k=3
        ).run
    )
    prompt = hub.pull("hwchase17/openai-functions-agent")
    # 创建多工具Agent
    multi_agent = create_openai_functions_agent(llm, [general_tool, tech_tool], prompt)
    multi_executor = AgentExecutor(
        agent=multi_agent,
        tools=[general_tool, tech_tool],
        verbose=False
    )
    
    # 测试多工具搜索
    tech_query = "如何使用LangChain创建Agent"
    print(f"多工具查询: {tech_query}")
    
    response = multi_executor.invoke({
        "input": f"请帮我搜索{tech_query}的详细教程和代码示例"
    })
    
    print("多工具Agent回答:")
    print(response["output"])
    
except Exception as e:
    print(f"多工具Agent失败: {e}")

# ========================================================================
# 对比总结
# ========================================================================

print("\n\n📊 搜索方式对比总结")
print("=" * 50)

print("""
🔍 直接搜索:
✅ 优点: 速度快，直接获取原始结果
❌ 缺点: 需要人工筛选和理解结果

🤖 Agent智能搜索:
✅ 优点: 自动分析和总结，回答更智能
✅ 优点: 可以进行多轮推理
❌ 缺点: 速度较慢，消耗更多token

🛠️ 多工具Agent:
✅ 优点: 根据问题类型选择最合适的搜索引擎
✅ 优点: 搜索结果更精准
❌ 缺点: 配置复杂，需要更多资源

💡 使用建议:
- 简单查询: 使用直接搜索
- 复杂分析: 使用Agent智能搜索  
- 专业领域: 使用多工具Agent
""")
