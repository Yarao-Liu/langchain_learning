"""
LangChain XML Agent 详细注释版示例

本示例展示了如何创建和使用XML格式的智能代理，包括：
1. 环境配置和错误处理 - 安全地加载API密钥和配置参数
2. 搜索工具集成 - 集成SerpAPI实现网络搜索功能
3. XML代理创建和使用 - 使用LangChain的XML代理框架
4. 完整的对话交互示例 - 提供实际可用的对话界面

XML代理工作原理：
- XML代理使用特定的XML标签格式与工具交互
- 代理通过<tool>和<tool_input>标签调用工具
- 工具返回结果后，代理生成<final_answer>标签包含最终答案
- AgentExecutor负责管理整个执行流程和错误处理

技术架构：
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   用户输入   │ -> │  XML代理     │ -> │  工具调用    │
└─────────────┘    └──────────────┘    └─────────────┘
                           │                    │
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   最终答案   │ <- │  答案生成     │ <- │  搜索结果    │
└─────────────┘    └──────────────┘    └─────────────┘

XML代理格式示例：
<tool>search</tool>
<tool_input>今天北京天气</tool_input>
<observation>天气数据...</observation>
<final_answer>今天北京天气晴朗...</final_answer>

优化特点：
- 使用经典的XML代理格式，结构清晰易读
- 完善的错误处理和日志记录
- 模块化的代码结构，便于维护和扩展
- 详细的文档和注释，便于学习理解
- 实际使用示例和交互式模式

依赖库说明：
- langchain: 核心框架，提供代理和工具抽象
- langchain_openai: OpenAI模型集成
- langchain_community: 社区工具集成（如SerpAPI）
- dotenv: 环境变量管理
- logging: 日志记录

作者：AI助手
日期：2024年
版本：2.0 (XML代理详细注释版)
"""

# ============================================================================
# 导入必要的库和模块
# ============================================================================

import os          # 操作系统接口，用于环境变量操作
import sys         # 系统相关参数和函数，用于程序退出
import logging     # 日志记录模块，用于调试和监控
import json        # JSON处理模块，用于格式化输出
from typing import Dict, Any, List  # 类型提示，提高代码可读性和IDE支持

# LangChain相关导入
import dotenv                                    # 环境变量加载器
from langchain import hub                        # LangChain Hub，用于获取预定义的提示词模板
from langchain.agents import AgentExecutor       # 代理执行器，管理代理的执行流程
from langchain.agents.xml.base import create_xml_agent  # XML代理创建函数
from langchain_core.tools import Tool            # 工具基类，用于创建自定义工具
from langchain_openai import ChatOpenAI          # OpenAI聊天模型集成
from langchain_community.utilities import SerpAPIWrapper  # SerpAPI搜索工具包装器

# ============================================================================
# 日志配置和全局设置
# ============================================================================

# 配置日志系统
# 日志级别设置为INFO，记录程序运行的关键信息
# 日志格式包含时间戳、日志级别和消息内容，便于调试和监控
logging.basicConfig(
    level=logging.INFO,                                    # 设置日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s'    # 设置日志格式
)
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# ============================================================================
# 环境配置管理
# ============================================================================

def load_environment():
    """
    加载和验证环境变量配置

    这个函数负责：
    1. 从.env文件加载环境变量
    2. 获取必需的API密钥和配置参数
    3. 验证配置的完整性和有效性
    4. 提供默认值和错误处理

    Returns:
        dict: 包含所有配置参数的字典

    Raises:
        ValueError: 当必需的API密钥缺失时
        Exception: 当环境配置加载失败时

    环境变量说明：
    - OPENAI_API_KEY: OpenAI API密钥（必需）
    - SERPAPI_API_KEY: SerpAPI搜索服务密钥（必需）
    - OPENAI_BASE_URL: API服务地址（可选，默认使用SiliconFlow）
    - OPENAI_MODEL: 使用的模型名称（可选，默认Qwen2.5-7B）
    - OPENAI_TEMPERATURE: 模型温度参数（可选，默认0.7）
    """
    try:
        # 加载.env文件中的环境变量
        # dotenv.load_dotenv()会查找当前目录及父目录中的.env文件
        dotenv.load_dotenv()
        logger.info("环境变量文件加载成功")

        # 构建配置字典，包含所有必需和可选的配置参数
        config = {
            # 必需的API密钥
            'openai_api_key': os.getenv("OPENAI_API_KEY"),      # OpenAI兼容API密钥
            'serpapi_key': os.getenv("SERPAPI_API_KEY"),        # SerpAPI搜索服务密钥

            # 可选的配置参数（提供默认值）
            'base_url': os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1/"),  # API服务地址
            'model': os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct"),             # 模型名称
            'temperature': float(os.getenv("OPENAI_TEMPERATURE", "0.7"))                # 温度参数（控制输出随机性）
        }

        # ========================================================================
        # API密钥验证
        # ========================================================================

        # 验证OpenAI API密钥
        if not config['openai_api_key']:
            error_msg = "未找到OPENAI_API_KEY环境变量"
            logger.error(error_msg)
            raise ValueError(f"{error_msg}，请在.env文件中设置")

        # 验证SerpAPI密钥
        if not config['serpapi_key']:
            # 提供详细的SerpAPI配置指南
            error_msg = """
错误：未找到SERPAPI_API_KEY环境变量

SerpAPI是一个提供Google搜索结果的API服务，用于获取实时搜索信息。

请按照以下步骤获取并配置SerpAPI密钥：

1. 注册账户：
   - 访问 https://serpapi.com/
   - 点击"Sign Up"创建免费账户

2. 获取API密钥：
   - 登录后进入Dashboard
   - 在"API Key"部分复制你的密钥

3. 配置密钥（选择其中一种方式）：
   - 方式1：在.env文件中添加：SERPAPI_API_KEY=你的密钥
   - 方式2：设置环境变量：export SERPAPI_API_KEY=你的密钥
   - 方式3：在系统环境变量中设置

4. 注意事项：
   - SerpAPI提供每月100次免费搜索额度
   - 免费额度足够测试和学习使用
   - 超出额度后需要付费订阅

示例.env文件内容：
OPENAI_API_KEY=your_openai_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here
            """
            print(error_msg)
            logger.error("SerpAPI密钥未配置")
            sys.exit(1)  # 退出程序，因为没有搜索功能无法正常工作

        logger.info("API密钥验证通过")
        logger.info(f"使用模型: {config['model']}")
        logger.info(f"API服务地址: {config['base_url']}")

        return config

    except Exception as e:
        logger.error(f"环境配置加载失败: {e}")
        raise  # 重新抛出异常，让调用者处理

# ============================================================================
# 代理和工具创建
# ============================================================================

def create_agent_executor(config):
    """
    创建XML代理执行器和相关工具

    这个函数是整个系统的核心，负责：
    1. 初始化语言模型（LLM）
    2. 创建搜索工具
    3. 设置XML代理
    4. 配置代理执行器

    Args:
        config (dict): 包含API密钥和配置参数的字典

    Returns:
        tuple: (agent_executor, tools) 代理执行器和工具列表

    Raises:
        Exception: 当任何组件初始化失败时

    组件说明：
    - ChatOpenAI: 语言模型，负责理解和生成文本
    - SerpAPIWrapper: 搜索工具包装器，提供网络搜索能力
    - Tool: 工具抽象，将搜索功能包装为代理可用的工具
    - create_xml_agent: 创建使用XML格式的代理
    - AgentExecutor: 代理执行器，管理代理的运行和错误处理
    """
    try:
        # ========================================================================
        # 1. 初始化语言模型
        # ========================================================================

        # 创建ChatOpenAI实例，这是代理的"大脑"
        # 使用OpenAI兼容的API接口，支持多种模型提供商
        llm = ChatOpenAI(
            api_key=config['openai_api_key'],    # API密钥，用于身份验证
            base_url=config['base_url'],         # API服务地址，支持第三方服务
            model=config['model'],               # 模型名称，如Qwen2.5-7B-Instruct
            temperature=config['temperature']    # 温度参数，控制输出的随机性和创造性
        )
        logger.info(f"语言模型初始化成功: {config['model']}")
        logger.info(f"模型温度设置: {config['temperature']}")

        # ========================================================================
        # 2. 创建搜索工具
        # ========================================================================

        # 初始化SerpAPI搜索包装器
        # SerpAPI提供Google搜索结果的API接口，让代理能够获取实时信息
        search_wrapper = SerpAPIWrapper(serpapi_api_key=config['serpapi_key'])

        # 将搜索包装器转换为LangChain工具
        # Tool类提供了标准化的工具接口，包含名称、描述和执行函数
        search_tool = Tool(
            name="search",  # 工具名称，代理会通过这个名称调用工具
            description=(   # 工具描述，告诉代理何时以及如何使用这个工具
                "搜索互联网获取最新信息。"
                "当需要查找实时信息、新闻、数据或回答需要最新知识的问题时使用。"
                "输入应该是搜索查询字符串。"
            ),
            func=search_wrapper.run  # 实际执行搜索的函数
        )

        # 添加一个计算器工具作为示例
        def calculator(expression: str) -> str:
            """
            简单的计算器工具

            Args:
                expression (str): 数学表达式

            Returns:
                str: 计算结果
            """
            try:
                # 安全地评估数学表达式
                # 只允许基本的数学运算
                allowed_chars = set('0123456789+-*/.() ')
                if not all(c in allowed_chars for c in expression):
                    return "错误：表达式包含不允许的字符"

                result = eval(expression)
                return f"计算结果: {result}"
            except Exception as e:
                return f"计算错误: {str(e)}"

        calculator_tool = Tool(
            name="calculator",
            description="执行基本的数学计算。输入应该是数学表达式，如 '2+3*4' 或 '(10-5)/2'。",
            func=calculator
        )

        # 将工具添加到工具列表
        # XML代理可以使用多个工具，展示其灵活性
        tools = [search_tool, calculator_tool]
        logger.info(f"工具初始化成功，共加载 {len(tools)} 个工具")
        logger.info(f"可用工具: {[tool.name for tool in tools]}")

        # ========================================================================
        # 3. 创建XML代理
        # ========================================================================

        # 从LangChain Hub获取XML代理的提示词模板
        # "hwchase17/xml-agent-convo"是一个预定义的XML代理模板
        # 这个模板定义了代理如何使用XML格式与工具交互
        prompt = hub.pull("hwchase17/xml-agent-convo")
        logger.info("XML代理提示词模板加载成功")

        # 创建XML代理
        # XML代理使用特定的XML标签格式来调用工具和生成回答
        # 格式示例：<tool>search</tool><tool_input>查询内容</tool_input>
        agent = create_xml_agent(
            llm=llm,        # 语言模型
            tools=tools,    # 可用工具列表
            prompt=prompt   # 提示词模板
        )
        logger.info("XML代理创建成功")

        # ========================================================================
        # 4. 创建代理执行器
        # ========================================================================

        # AgentExecutor是代理的运行环境，负责：
        # - 管理代理的执行流程
        # - 处理工具调用
        # - 错误处理和重试
        # - 日志记录和调试
        agent_executor = AgentExecutor(
            agent=agent,                    # XML代理实例
            tools=tools,                    # 工具列表
            verbose=True,                   # 启用详细输出，显示代理的思考过程
            handle_parsing_errors=True,     # 自动处理解析错误，提高容错性
            max_iterations=5,               # 最大迭代次数，防止无限循环
            max_execution_time=60           # 最大执行时间（秒），防止超时
        )
        logger.info("XML代理执行器创建成功")
        logger.info("代理配置: verbose=True, handle_parsing_errors=True")

        return agent_executor, tools

    except Exception as e:
        logger.error(f"代理执行器创建失败: {e}")
        logger.error(f"错误详情: {type(e).__name__}: {str(e)}")
        raise  # 重新抛出异常，让调用者处理

# ============================================================================
# 代理查询和交互
# ============================================================================

def query_agent(agent_executor, question: str, verbose: bool = True) -> Dict[str, Any]:
    """
    向XML代理提问并获取回答

    这个函数是用户与代理交互的主要接口，负责：
    1. 接收用户问题
    2. 调用代理执行器处理问题
    3. 解析和格式化代理的回答
    4. 处理可能出现的错误

    Args:
        agent_executor: 代理执行器实例
        question (str): 用户提出的问题
        verbose (bool): 是否显示详细的交互过程，默认True

    Returns:
        Dict[str, Any]: 包含以下键的字典：
            - output (str): 代理的回答或错误信息
            - success (bool): 是否成功处理问题
            - error (str, optional): 错误信息（仅在失败时存在）

    代理执行流程：
    1. 代理接收问题
    2. 分析问题是否需要使用工具
    3. 如果需要，调用相应工具（如搜索或计算器）
    4. 基于工具结果生成最终答案
    5. 返回格式化的回答

    XML代理的工作示例：
    用户问题: "今天北京的天气怎么样？"
    代理思考: 需要获取实时天气信息
    工具调用: <tool>search</tool><tool_input>今天北京天气</tool_input>
    搜索结果: <observation>天气数据...</observation>
    最终回答: <final_answer>今天北京天气晴朗...</final_answer>

    用户问题: "计算 15 * 8 + 32"
    代理思考: 需要进行数学计算
    工具调用: <tool>calculator</tool><tool_input>15*8+32</tool_input>
    计算结果: <observation>152</observation>
    最终回答: <final_answer>计算结果是152</final_answer>
    """
    try:
        logger.info(f"开始处理问题: {question}")
        logger.info("调用代理执行器...")

        # ========================================================================
        # 执行代理查询
        # ========================================================================

        # 调用代理执行器处理问题
        # agent_executor.invoke()是异步执行的，会：
        # 1. 将问题传递给XML代理
        # 2. 代理分析问题并决定是否使用工具
        # 3. 如果需要，调用搜索工具或计算器工具获取信息
        # 4. 基于所有信息生成最终回答
        result = agent_executor.invoke({"input": question})

        # ========================================================================
        # 处理代理返回结果
        # ========================================================================

        # 从结果中提取输出文本
        # AgentExecutor返回的结果是一个字典，通常包含'output'键
        # 如果没有'output'键，则将整个结果转换为字符串
        output = result.get('output', str(result))

        # 记录成功信息
        logger.info("问题处理完成")
        logger.info(f"回答长度: {len(output)} 字符")

        # ========================================================================
        # 显示交互结果（如果启用详细模式）
        # ========================================================================

        if verbose:
            print(f"\n问题: {question}")
            print(f"回答: {output}")
            print("-" * 50)

        # 返回成功结果
        from datetime import datetime
        return {
            "output": output,
            "success": True,
            "question": question,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        # ========================================================================
        # 错误处理
        # ========================================================================

        # 构建用户友好的错误信息
        error_msg = f"抱歉，处理您的问题时出现错误: {str(e)}"

        # 记录详细的错误信息用于调试
        logger.error(f"查询处理失败: {e}")
        logger.error(f"错误类型: {type(e).__name__}")
        logger.error(f"问题内容: {question}")

        # 显示错误信息（如果启用详细模式）
        if verbose:
            print(f"\n问题: {question}")
            print(f"错误: {error_msg}")
            print("-" * 50)

        # 返回错误结果
        return {
            "output": error_msg,
            "error": str(e),
            "success": False,
            "question": question
        }

# ============================================================================
# 主程序和应用入口
# ============================================================================

def main():
    """
    主程序入口函数

    这是整个应用程序的入口点，负责：
    1. 初始化所有组件
    2. 运行示例查询
    3. 提供交互式对话界面
    4. 处理程序异常和退出

    程序执行流程：
    1. 显示欢迎信息
    2. 加载环境配置
    3. 创建代理执行器
    4. 显示系统配置信息
    5. 运行预设示例
    6. 可选进入交互模式
    7. 优雅处理退出

    异常处理：
    - KeyboardInterrupt: 用户按Ctrl+C中断程序
    - 其他异常: 记录错误并退出程序
    """

    # ========================================================================
    # 程序启动和欢迎信息
    # ========================================================================

    print("=" * 60)
    print("LangChain XML Agent 详细注释版示例")
    print("=" * 60)
    print("本程序演示如何使用LangChain创建XML格式的智能代理")
    print("代理具备网络搜索和数学计算能力，可以回答各种问题")
    print("=" * 60)

    try:
        # ========================================================================
        # 1. 系统初始化
        # ========================================================================

        logger.info("开始初始化系统...")

        # 加载环境配置
        # 这一步会验证API密钥并设置所有必要的配置参数
        config = load_environment()
        logger.info("环境配置加载完成")

        # 创建代理执行器和工具
        # 这是系统的核心组件，包含语言模型、搜索工具和XML代理
        agent_executor, tools = create_agent_executor(config)
        logger.info("代理系统初始化完成")

        # ========================================================================
        # 2. 显示系统配置信息
        # ========================================================================

        print(f"\n🤖 代理配置信息:")
        print(f"   模型: {config['model']}")
        print(f"   API服务: {config['base_url']}")
        print(f"   温度参数: {config['temperature']}")
        print(f"   工具数量: {len(tools)}")
        print(f"   可用工具: {', '.join([tool.name for tool in tools])}")

        # ========================================================================
        # 3. 运行示例查询
        # ========================================================================

        # 预设的示例问题，展示XML代理的不同能力
        example_questions = [
            ("今天北京的天气怎么样？", "搜索工具", "实时信息查询"),
            ("计算 25 * 4 + 18 - 7", "计算器工具", "数学计算"),
            ("最新的人工智能发展趋势是什么？", "搜索工具", "知识性问题")
        ]

        print(f"\n🔍 开始示例查询...")
        print("=" * 60)
        print("以下示例展示XML代理如何处理不同类型的问题：")

        # 逐个处理示例问题
        for i, (question, expected_tool, question_type) in enumerate(example_questions, 1):
            print(f"\n📝 示例 {i}:")
            print(f"问题: {question}")
            print(f"问题类型: {question_type}")
            print(f"预期使用工具: {expected_tool}")

            # 调用代理处理问题
            result = query_agent(agent_executor, question)

            # 显示处理结果统计
            if result['success']:
                print(f"✅ 处理成功，回答长度: {len(result['output'])} 字符")
            else:
                print(f"❌ 处理失败: {result.get('error', '未知错误')}")

            # 添加延时避免API速率限制
            # 这对于免费API账户特别重要
            import time
            print("⏳ 等待2秒避免API限制...")
            time.sleep(2)

        print("\n" + "=" * 60)
        print("✅ 示例查询完成！")

        # ========================================================================
        # 4. 交互式模式
        # ========================================================================

        # 询问用户是否进入交互模式
        print(f"\n💬 交互模式说明:")
        print(f"   - 可以与代理进行自由对话")
        print(f"   - 代理会根据需要自动选择工具")
        print(f"   - 输入 'quit'、'exit' 或 '退出' 结束对话")

        interactive_mode = input("\n是否进入交互模式？(y/n): ").lower().strip()

        if interactive_mode == 'y':
            logger.info("用户选择进入交互模式")
            print("\n🚀 进入交互模式")
            print("-" * 40)
            print("提示：您可以问任何问题，代理会智能判断使用哪个工具")
            print("示例问题：")
            print("  搜索类：今天上海的股市表现如何？")
            print("  计算类：计算 (123 + 456) * 2 / 3")
            print("  知识类：解释一下什么是量子计算")
            print("  混合类：如果我有1000元，按年利率5%计算，10年后是多少？")
            print("-" * 40)

            # 交互式对话循环
            conversation_count = 0
            while True:
                try:
                    # 获取用户输入
                    user_input = input(f"\n[第{conversation_count + 1}轮] 请输入您的问题: ").strip()

                    # 检查退出命令
                    if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                        print("\n👋 感谢使用！再见！")
                        logger.info("用户主动退出交互模式")
                        break

                    # 检查输入有效性
                    if not user_input:
                        print("⚠️  请输入有效的问题")
                        continue

                    # 处理用户问题
                    conversation_count += 1
                    logger.info(f"处理第{conversation_count}轮对话")

                    result = query_agent(agent_executor, user_input)

                    # 显示简单的统计信息
                    if result['success']:
                        print(f"📊 回答统计: {len(result['output'])} 字符")

                except KeyboardInterrupt:
                    print("\n\n⚠️  检测到中断信号，退出交互模式")
                    break
                except Exception as e:
                    print(f"\n❌ 交互过程中出现错误: {e}")
                    logger.error(f"交互模式错误: {e}")
                    continue

            print(f"\n📈 本次会话统计: 共进行了 {conversation_count} 轮对话")
        else:
            print("\n👍 跳过交互模式")
            logger.info("用户选择跳过交互模式")

    # ========================================================================
    # 异常处理和程序退出
    # ========================================================================

    except KeyboardInterrupt:
        # 用户按Ctrl+C中断程序
        print("\n\n⚠️  程序被用户中断 (Ctrl+C)")
        logger.info("程序被用户中断")
        print("👋 感谢使用！")

    except Exception as e:
        # 其他未预期的异常
        print(f"\n❌ 程序执行出错: {e}")
        print(f"错误类型: {type(e).__name__}")
        logger.error(f"程序执行出错: {e}")
        logger.error(f"错误类型: {type(e).__name__}")

        # 提供调试建议
        print(f"\n🔧 调试建议:")
        print(f"   1. 检查.env文件中的API密钥是否正确")
        print(f"   2. 确认网络连接正常")
        print(f"   3. 查看日志文件获取详细错误信息")

        sys.exit(1)  # 以错误状态退出

    finally:
        # 程序结束时的清理工作
        logger.info("程序执行结束")
        print(f"\n📝 程序执行完毕")

# ============================================================================
# 程序入口点
# ============================================================================

if __name__ == "__main__":
    """
    Python程序的标准入口点

    当直接运行此脚本时（而不是作为模块导入），会执行main()函数
    这是Python程序的最佳实践，确保代码可以既作为脚本运行，也可以作为模块导入
    """
    main()
