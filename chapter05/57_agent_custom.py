"""
多功能自定义智能体示例（简化版）
================================

本文件演示如何创建多功能智能体，包括：
1. 文本长度计算工具
2. 互联网搜索工具
3. 天气查询工具

使用Tool类包装函数，确保工具能被正确调用。
参考56_agent_custom.py的简化风格，保持代码简洁易懂。
"""

import os
import requests
import dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ==================== 环境配置 ====================

# 加载环境变量
dotenv.load_dotenv()

# 获取API密钥
api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

# 验证必要的环境变量
if not api_key:
    print("❌ 错误：未找到OPENAI_API_KEY环境变量")
    exit(1)

if not serpapi_key:
    print("❌ 错误：未找到SERPAPI_API_KEY环境变量")
    exit(1)

# ==================== 模型初始化 ====================

# 初始化ChatOpenAI模型
# 使用更强的模型以获得更好的函数调用支持
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1/",
    model="Qwen/Qwen2.5-72B-Instruct",  # 使用更强的72B模型
    temperature=0  # 降低温度以获得更确定的输出
)

# ==================== 自定义工具定义 ====================

def get_word_length(word: str) -> str:
    """计算单词或文本的字符长度"""
    try:
        length = len(word.strip())
        return f"单词 '{word}' 的长度是 {length} 个字符"
    except Exception as e:
        return f"计算单词长度时出错: {str(e)}"

def search_internet(query: str) -> str:
    """在互联网上搜索信息"""
    try:
        search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
        result = search.run(query)
        return result
    except Exception as e:
        return f"搜索时出错: {str(e)}"

def get_weather(location: str) -> str:
    """获取指定城市的天气信息"""
    try:
        if weather_api_key:
            # 使用真实的天气API
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': location,
                'appid': weather_api_key,
                'units': 'metric',
                'lang': 'zh_cn'
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                weather_desc = data['weather'][0]['description']
                temp = data['main']['temp']
                return f"{location}的天气：{weather_desc}，温度：{temp}°C"
            else:
                return f"无法获取{location}的天气信息"
        else:
            # 模拟天气数据用于测试
            return f"{location}的模拟天气：晴天，温度：22°C（请配置WEATHER_API_KEY获取真实数据）"
    except Exception as e:
        return f"获取天气信息时出错: {str(e)}"

# ==================== 工具配置 ====================

# 工具列表（使用Tool类包装函数）
tools = [
    Tool(
        name="get_word_length",
        description="计算单词或文本的字符长度。输入参数：word（字符串）",
        func=get_word_length
    ),
    Tool(
        name="search_internet",
        description="在互联网上搜索信息。输入参数：query（搜索查询字符串）",
        func=search_internet
    ),
    Tool(
        name="get_weather",
        description="获取指定城市的天气信息。输入参数：location（城市名称）",
        func=get_weather
    )
]

# ==================== 提示词模板配置 ====================

# 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能助手，可以使用以下工具来帮助用户：

可用工具：
- get_word_length: 计算单词或文本的字符长度
- search_internet: 在互联网上搜索信息
- get_weather: 获取指定城市的天气信息

使用规则：
1. 当用户询问单词或文本长度时，必须使用 get_word_length 工具
2. 当用户需要搜索信息时，必须使用 search_internet 工具
3. 当用户询问天气时，必须使用 get_weather 工具
4. 不要猜测答案，必须调用相应的工具获取准确信息
5. 调用工具后，基于工具返回的结果给出完整回答"""),

    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ==================== 智能体创建 ====================

# 创建智能体
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# ==================== 测试代码 ====================

# 主程序入口
if __name__ == "__main__":
    print("🚀 多功能智能体测试开始...")
    print("=" * 50)

    # 首先测试工具是否正确定义
    print("\n🔧 工具定义检查:")
    for tool in tools:
        print(f"- 工具名称: {tool.name}")
        print(f"- 工具描述: {tool.description}")

    # 直接测试工具函数
    print("\n🧪 直接测试工具函数:")
    try:
        result1 = get_word_length("LangChain")
        print(f"get_word_length('LangChain'): {result1}")

        result3 = get_weather("北京")
        print(f"get_weather('北京'): {result3}")
    except Exception as e:
        print(f"直接测试失败: {e}")

    # 测试智能体
    print("\n🤖 智能体测试:")

    # 简单测试 - 只测试一个工具确保正常工作
    test_question = "请计算单词 'Hello' 的长度"
    print(f"\n📝 测试问题: {test_question}")
    print("-" * 40)

    try:
        response = agent_executor.invoke({"input": test_question})
        output = response['output']
        print(f"🤖 智能体回答: {output}")

        # 检查是否包含正确的结果
        if "5" in output and ("Hello" in output or "hello" in output):
            print("✅ 工具调用成功！智能体正确计算了单词长度")
        else:
            print("⚠️  智能体可能没有正确调用工具")

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

    print("-" * 40)

    # 如果第一个测试成功，继续测试其他工具
    print("\n📋 完整测试（三个工具）:")
    test_questions = [
        ("计算单词 'LangChain' 的长度", "9"),           # 测试文本长度工具
        ("搜索人工智能新闻", "搜索"),                    # 测试搜索工具
        ("查询北京天气", "天气")                         # 测试天气工具
    ]

    for i, (question, expected_keyword) in enumerate(test_questions, 1):
        print(f"\n📝 测试 {i}: {question}")
        print("-" * 30)

        try:
            response = agent_executor.invoke({"input": question})
            output = response['output']
            print(f"🤖 回答: {output}")

            if expected_keyword in output:
                print("✅ 测试通过！")
            else:
                print("⚠️  结果可能不准确")

        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")

        print("-" * 30)

    print("\n🎉 所有测试完成！")