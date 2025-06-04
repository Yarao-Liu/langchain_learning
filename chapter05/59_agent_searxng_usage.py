# SearxSearchWrapper 使用示例 - 无警告版本
# 解决LangSmith API密钥警告问题

import os
import warnings
import dotenv
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.tools import Tool

# 方法1: 禁用LangSmith相关警告
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")

# 方法2: 设置环境变量禁用LangSmith追踪
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""

# 加载环境变量
dotenv.load_dotenv()

print("SearxSearchWrapper 使用示例 - 无警告版本")
print("=" * 50)

# ========================================================================
# 基础使用示例
# ========================================================================

print("\n1. 基础搜索示例")
print("-" * 30)

# 配置SearxNG服务器地址
searx_host = "http://localhost:6688"  # 本地部署
# searx_host = "https://searx.be"     # 公共实例（可能有限制）

try:
    # 创建搜索包装器
    search = SearxSearchWrapper(
        searx_host=searx_host,
        k=3  # 返回3个结果
    )
    
    # 执行搜索
    query = "Python编程教程"
    print(f"搜索查询: {query}")
    
    result = search.run(query)
    print("搜索结果:")
    print(result[:300] + "..." if len(result) > 300 else result)
    
except Exception as e:
    print(f"搜索失败: {e}")
    print("请确保SearxNG服务正在运行")

# ========================================================================
# 指定搜索引擎
# ========================================================================

print("\n2. 指定搜索引擎示例")
print("-" * 30)

try:
    # GitHub专用搜索
    github_search = SearxSearchWrapper(
        searx_host=searx_host,
        engines=["github"],
        k=2
    )
    
    query = "langchain python examples"
    print(f"GitHub搜索: {query}")
    
    result = github_search.run(query)
    print("GitHub搜索结果:")
    print(result[:300] + "..." if len(result) > 300 else result)
    
except Exception as e:
    print(f"GitHub搜索失败: {e}")

# ========================================================================
# 创建工具
# ========================================================================

print("\n3. 创建LangChain工具")
print("-" * 30)

try:
    # 创建通用搜索工具
    search_tool = Tool(
        name="web_search",
        description="搜索互联网获取信息",
        func=SearxSearchWrapper(searx_host=searx_host, k=3).run
    )
    
    # 使用工具
    query = "人工智能最新发展"
    print(f"工具搜索: {query}")
    
    result = search_tool.invoke(query)
    print("工具搜索结果:")
    print(result[:300] + "..." if len(result) > 300 else result)
    
except Exception as e:
    print(f"工具搜索失败: {e}")

# ========================================================================
# 多引擎搜索
# ========================================================================

print("\n4. 多引擎搜索示例")
print("-" * 30)

try:
    # 多引擎搜索
    multi_search = SearxSearchWrapper(
        searx_host=searx_host,
        engines=["google", "bing", "duckduckgo"],
        k=5
    )
    
    query = "机器学习算法"
    print(f"多引擎搜索: {query}")
    
    result = multi_search.run(query)
    print("多引擎搜索结果:")
    print(result[:300] + "..." if len(result) > 300 else result)
    
except Exception as e:
    print(f"多引擎搜索失败: {e}")

# ========================================================================
# SearxNG部署快速指南
# ========================================================================

print("\n" + "=" * 50)
print("SearxNG快速部署指南")
print("=" * 50)

print("""
1. 使用Docker部署（推荐）:
   docker run -d -p 8888:8080 searxng/searxng

2. 验证部署:
   浏览器访问: http://localhost:8888
   API测试: curl "http://localhost:8888/search?q=test&format=json"

3. 配置JSON格式输出:
   编辑settings.yml文件，添加:
   search:
     formats:
       - html
       - json

4. 常用公共实例:
   - https://searx.be
   - https://searx.info
   - https://search.sapti.me
   注意: 公共实例可能不支持API或有使用限制

5. 环境变量配置:
   SEARXNG_HOST=http://localhost:8888
""")

print("\n程序执行完成，无警告信息！")
