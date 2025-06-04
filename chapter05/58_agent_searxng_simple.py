# SearxSearchWrapper 简单使用示例
import os
import dotenv
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.tools import Tool

# 加载环境变量
dotenv.load_dotenv()

# 禁用LangSmith追踪（避免API密钥警告）
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ========================================================================
# 方法1: 使用本地SearxNG实例（推荐）
# ========================================================================

print("方法1: 使用本地SearxNG实例")
print("-" * 40)

# 本地SearxNG服务地址
searx_host = "http://localhost:6688"

try:
    # 创建搜索包装器
    search = SearxSearchWrapper(searx_host=searx_host)
    
    # 执行搜索
    result = search.run("Python LangChain教程")
    print("搜索结果:")
    print(result)
    
except Exception as e:
    print(f"本地搜索失败: {e}")
    print("请确保SearxNG服务正在运行")
# ========================================================================
# 方法2: 指定搜索引擎
# ========================================================================

print("\n方法2: 指定搜索引擎")
print("-" * 40)

try:
    # 只搜索GitHub
    github_search = SearxSearchWrapper(
        searx_host=searx_host,
        engines=["github"]
    )
    
    result = github_search.run("langchain python")
    print("GitHub搜索结果:")
    print(result[:200] + "..." if len(result) > 200 else result)
    
except Exception as e:
    print(f"GitHub搜索失败: {e}")

# ========================================================================
# 方法3: 创建LangChain工具
# ========================================================================

print("\n方法3: 创建LangChain工具")
print("-" * 40)

try:
    # 创建搜索工具
    search_tool = Tool(
        name="searx_search",
        description="搜索互联网获取信息",
        func=SearxSearchWrapper(searx_host=searx_host).run
    )
    
    # 使用工具
    result = search_tool.invoke("机器学习算法")
    print("工具搜索结果:")
    print(result[:200] + "..." if len(result) > 200 else result)
    
except Exception as e:
    print(f"工具搜索失败: {e}")

# ========================================================================
# SearxNG部署说明
# ========================================================================

print("\n" + "=" * 50)
print("SearxNG部署说明")
print("=" * 50)

print("""
1. 使用Docker快速部署:
   docker run -d -p 6688:8080 searxng/searxng

2. 配置JSON输出格式:
   需要在settings.yml中添加:
   search:
     formats:
       - html
       - json

3. 测试API是否工作:
   curl -X GET "http://localhost:6688/search?q=test&format=json"")
"""
)
