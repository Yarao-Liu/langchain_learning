"""
自定义智能体示例
================

本文件演示如何创建一个自定义的LangChain智能体，包括：
1. 定义自定义工具函数
2. 创建智能体和执行器
3. 配置提示词模板
4. 测试智能体功能

主要功能：
- 创建一个可以计算单词长度的智能体
- 使用OpenAI Functions Agent架构
- 集成自定义工具
"""

import os
import dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ==================== 环境配置 ====================

# 加载.env文件中的环境变量
# 这样可以安全地存储API密钥等敏感信息
dotenv.load_dotenv()

# 从环境变量中获取API密钥
# OPENAI_API_KEY: OpenAI API的访问密钥
# SERPAPI_API_KEY: SerpAPI的访问密钥（虽然本例中未使用，但保留用于扩展）
api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")

# 验证必要的环境变量是否存在
# 如果缺少关键的API密钥，程序将退出并显示错误信息
if not api_key:
    print("❌ 错误：未找到OPENAI_API_KEY环境变量")
    print("请在.env文件中设置OPENAI_API_KEY")
    exit(1)

if not serpapi_key:
    print("❌ 错误：未找到SERPAPI_API_KEY环境变量")
    print("请在.env文件中设置SERPAPI_API_KEY")
    exit(1)

# ==================== 模型初始化 ====================

# 初始化ChatOpenAI模型
# 使用SiliconFlow作为代理服务，提供更稳定的API访问
# Qwen2.5-7B-Instruct是一个高性能的中文大语言模型
llm = ChatOpenAI(
    api_key=api_key,                                    # API密钥
    base_url="https://api.siliconflow.cn/v1/",         # 使用SiliconFlow代理服务
    model="Qwen/Qwen2.5-7B-Instruct"                   # 选择Qwen2.5模型
)

# ==================== 自定义工具定义 ====================

def get_word_length(word: str) -> str:
    """
    计算单词或文本的字符长度

    这是一个自定义工具函数，用于演示如何创建智能体工具。

    参数:
        word (str): 需要计算长度的单词或文本

    返回:
        str: 包含长度信息的格式化字符串

    异常处理:
        如果计算过程中出现错误，返回错误信息
    """
    try:
        # 去除首尾空格后计算长度
        length = len(word.strip())
        return f"单词 '{word}' 的长度是 {length} 个字符"
    except Exception as e:
        return f"计算单词长度时出错: {str(e)}"

# ==================== 工具配置 ====================

# 创建工具列表
# Tool类用于将Python函数包装成LangChain可以使用的工具
tools = [
    Tool(
        name="get_word_length",                         # 工具名称，智能体会根据此名称调用工具
        description="获取单词或文本的字符长度。输入参数：word（字符串）",  # 工具描述，帮助智能体理解工具用途
        func=get_word_length                            # 实际执行的函数
    )
]

# ==================== 提示词模板配置 ====================

# 创建聊天提示词模板
# ChatPromptTemplate用于定义智能体的行为和响应方式
prompt = ChatPromptTemplate.from_messages([
    # 系统消息：定义智能体的角色和行为规范
    ("system", """你是一个智能助手，可以计算单词或文本的长度。

当用户询问单词长度时，使用 get_word_length 工具来计算。
请友好、准确地回答用户的问题。"""),

    # 人类消息：用户输入的占位符
    ("human", "{input}"),

    # 智能体工作区：存储智能体的思考过程和工具调用历史
    # 这是OpenAI Functions Agent必需的组件
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ==================== 智能体创建 ====================

# 创建OpenAI Functions智能体
# 这种类型的智能体专门设计用于调用函数/工具
agent = create_openai_functions_agent(
    llm=llm,        # 使用的语言模型
    tools=tools,    # 可用的工具列表
    prompt=prompt   # 提示词模板
)

# ==================== 执行器配置 ====================

# 创建智能体执行器
# AgentExecutor负责管理智能体的执行流程，包括工具调用和错误处理
agent_executor = AgentExecutor(
    agent=agent,                    # 要执行的智能体
    tools=tools,                    # 可用的工具列表
    verbose=True,                   # 启用详细输出，显示执行过程
    handle_parsing_errors=True      # 自动处理解析错误，提高稳定性
)

# ==================== 测试代码 ====================

# 主程序入口
# 只有直接运行此文件时才会执行测试代码
if __name__ == "__main__":
    print("🚀 开始测试自定义智能体...")
    print("=" * 50)

    # 定义测试问题
    # 这个问题将触发智能体使用get_word_length工具
    test_question = "请帮我计算单词 'hello' 的长度"
    print(f"\n📝 测试问题: {test_question}")

    try:
        # 调用智能体执行器处理用户输入
        # invoke方法会：
        # 1. 解析用户输入
        # 2. 决定是否需要使用工具
        # 3. 调用相应的工具
        # 4. 生成最终回答
        response = agent_executor.invoke({"input": test_question})

        # 显示智能体的回答
        print(f"\n🤖 智能体回答: {response['output']}")
        print("\n✅ 测试成功！")

    except Exception as e:
        # 捕获并显示任何执行过程中的错误
        print(f"\n❌ 测试失败: {str(e)}")
        print("请检查：")
        print("1. 环境变量是否正确设置")
        print("2. 网络连接是否正常")
        print("3. API密钥是否有效")
