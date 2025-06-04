# LangChain Agent + LLM + SearxNG 智能搜索系统
import os
import warnings

import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub

# 禁用LangSmith追踪（避免API密钥警告）
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# 方法1: 禁用LangSmith相关警告
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")
# 加载环境变量
dotenv.load_dotenv()

# 获取环境变量
api_key = os.getenv("OPENAI_API_KEY")
searx_host = os.getenv("SEARXNG_HOST", "http://localhost:6688")

# 检查必要的配置
if not api_key:
    print("错误：未找到OPENAI_API_KEY环境变量")
    print("请在.env文件中添加：OPENAI_API_KEY=你的密钥")
    exit(1)

print("🤖 LangChain Agent + LLM + SearxNG 智能搜索系统")
print("=" * 60)

# ========================================================================
# 1. 初始化LLM
# ========================================================================

print("1. 初始化语言模型...")
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1/",
    model="Qwen/Qwen2.5-72B-Instruct",
    temperature=0.1  # 降低温度以获得更准确的回答
)
print("✓ 语言模型初始化完成")

# ========================================================================
# 2. 创建多种搜索工具
# ========================================================================

print("\n2. 创建搜索工具...")

try:
    # 通用搜索工具
    general_search = Tool(
        name="general_search",
        description=(
            "搜索互联网获取一般信息。"
            "当需要查找实时信息、新闻、常识性问题、产品信息等时使用。"
            "输入应该是搜索查询字符串。"
        ),
        func=SearxSearchWrapper(
            searx_host=searx_host,
            k=5
        ).run
    )
    
    # 技术搜索工具
    tech_search = Tool(
        name="tech_search", 
        description=(
            "搜索技术相关信息，包括编程、软件开发、技术文档等。"
            "当需要查找代码示例、技术教程、API文档、编程问题解决方案时使用。"
            "输入应该是技术相关的搜索查询。"
        ),
        func=SearxSearchWrapper(
            searx_host=searx_host,
            engines=["github", "stackoverflow"],
            k=5
        ).run
    )
    
    # 学术搜索工具
    academic_search = Tool(
        name="academic_search",
        description=(
            "搜索学术论文、研究资料和科学文献。"
            "当需要查找学术研究、论文、科学数据、研究报告时使用。"
            "输入应该是学术相关的搜索查询。"
        ),
        func=SearxSearchWrapper(
            searx_host=searx_host,
            engines=["arxiv", "google scholar"],
            k=3
        ).run
    )
    
    tools = [general_search, tech_search, academic_search]
    print(f"✓ 成功创建 {len(tools)} 个搜索工具")
    
except Exception as e:
    print(f"✗ 搜索工具创建失败: {e}")
    print("请确保SearxNG服务正在运行")
    
    # 创建模拟工具以便演示
    mock_tool = Tool(
        name="mock_search",
        description="模拟搜索工具（SearxNG不可用时使用）",
        func=lambda x: f"模拟搜索结果：关于'{x}'的信息"
    )
    tools = [mock_tool]

# ========================================================================
# 3. 创建Agent
# ========================================================================

print("\n3. 创建智能搜索Agent...")

try:
    # 获取Agent提示模板
    prompt = hub.pull("hwchase17/openai-functions-agent")
    
    # 创建Agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # 创建Agent执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 显示详细执行过程
        max_iterations=3,  # 最大迭代次数
        return_intermediate_steps=True  # 返回中间步骤
    )
    
    print("✓ Agent创建成功")
    
except Exception as e:
    print(f"✗ Agent创建失败: {e}")
    exit(1)

# ========================================================================
# 4. 测试搜索功能
# ========================================================================

print("\n4. 开始测试智能搜索...")

# 测试查询列表
test_queries = [
    "刘亦菲最近有什么新电影？",
    "Python LangChain框架的最新版本有什么新特性？", 
    "机器学习在医疗诊断中的最新研究进展",
    "如何使用Docker部署SearxNG搜索引擎？"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*60}")
    print(f"测试 {i}: {query}")
    print(f"{'='*60}")
    
    try:
        # 执行搜索
        response = agent_executor.invoke({"input": query})
        
        print("\n🤖 Agent回答:")
        print("-" * 40)
        print(response["output"])
        
        # 显示中间步骤（可选）
        if response.get("intermediate_steps"):
            print("\n🔍 搜索过程:")
            print("-" * 40)
            for step in response["intermediate_steps"]:
                action = step[0]
                result = step[1]
                print(f"工具: {action.tool}")
                print(f"查询: {action.tool_input}")
                print(f"结果: {result[:100]}...")
                print()
        
    except Exception as e:
        print(f"✗ 搜索失败: {e}")
    
    print()

# ========================================================================
# 6. 使用说明
# ========================================================================

print("\n" + "=" * 60)
print("使用说明")
print("=" * 60)

print("""
🔧 环境配置:
1. 安装依赖: pip install langchain langchain-openai langchain-community
2. 配置.env文件:
   OPENAI_API_KEY=你的OpenAI密钥
   SEARXNG_HOST=http://localhost:8888

🚀 SearxNG部署:
1. Docker部署: docker run -d -p 8888:8080 searxng/searxng
2. 验证部署: curl "http://localhost:8888/search?q=test&format=json"

🤖 Agent工作原理:
1. 接收用户问题
2. 分析问题类型（一般信息、技术问题、学术研究）
3. 选择合适的搜索工具
4. 执行搜索并获取结果
5. 使用LLM分析和总结搜索结果
6. 返回智能化的回答

🛠️ 可用工具:
- general_search: 一般信息搜索
- tech_search: 技术问题搜索（GitHub、StackOverflow）
- academic_search: 学术研究搜索（arXiv、Google Scholar）
""")
