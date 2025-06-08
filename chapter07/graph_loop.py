"""
LangGraph 智能Agent工具调用系统
=====================================

这是一个完整的LangGraph工作流示例，实现了智能Agent的工具调用循环：

核心功能：
1. 🚀 start_node: 接收用户输入，LLM进行智能决策
2. 🤔 isUseTool: 判断是否需要调用工具还是直接回答
3. 🔧 tool_node: 执行具体的工具调用（如网络搜索）
4. 🎯 final_answer_node: 生成最终回答并结束流程

工作流程图：
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ 用户输入    │ -> │ start_node   │ -> │ isUseTool判断   │
└─────────────┘    │ (LLM决策)    │    │                 │
                   └──────────────┘    └─────────────────┘
                          ↑                      │
                          │                      ▼
                   ┌──────────────┐         ┌─────────────┐
                   │ tool_node    │ <------ │ 需要工具？  │
                   │ (执行工具)   │         │             │
                   └──────────────┘         └─────────────┘
                                                   │
                                                   ▼
                                            ┌─────────────────┐
                                            │ final_answer    │
                                            │ (最终回答)      │
                                            └─────────────────┘
                                                   │
                                                   ▼
                                               ┌─────────┐
                                               │   END   │
                                               └─────────┘

技术特点：
- 🔄 支持多轮工具调用循环
- 📝 完整的状态管理和历史记录
- 🛡️ 健壮的错误处理机制
- 🔍 详细的执行日志和调试信息

作者：AI助手
日期：2024年
版本：v1.0
"""

# ================================
# 1. 导入必要的库和模块
# ================================

import os
import dotenv
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, render_text_description
from langchain_core.utils.json import parse_json_markdown
from langchain_openai import ChatOpenAI
from langgraph.constants import END

# ================================
# 2. 环境配置和LLM初始化
# ================================

# 加载环境变量文件(.env)
# 确保在项目根目录有 .env 文件，包含 OPENAI_API_KEY=your_key_here
dotenv.load_dotenv()

# 检查 API 密钥配置
# 从环境变量中获取OpenAI API密钥
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ 错误: 未找到 OPENAI_API_KEY 环境变量")
    print("📝 请在 .env 文件中设置: OPENAI_API_KEY=your_api_key_here")
    print("💡 或者在系统环境变量中设置该值")
    exit(1)

# 创建大语言模型实例
# 使用 SiliconFlow 提供的 API 接口，完全兼容 OpenAI 格式
llm = ChatOpenAI(
    api_key=api_key,                                    # API密钥
    base_url="https://api.siliconflow.cn/v1/",         # SiliconFlow API地址
    model="Qwen/Qwen2.5-72B-Instruct",                 # 通义千问大模型
    temperature=0.3                                     # 控制回复的随机性(0-1)
)

# 字符串输出解析器（备用，主要使用JSON解析器）
output_parser = StrOutputParser()

# ================================
# 3. 提示词模板设计
# ================================

# 主要的提示词模板
# 这个模板定义了Agent的行为模式和输出格式
promptTemplate = """
尽可能的帮助用户回答任何问题。
你可以使用以下工具来帮忙解决问题：
{tools}

用户问题: {input}

请严格按照以下两种JSON格式之一回复，不要添加任何其他内容：

选项1：如果需要使用工具搜索信息，请使用此格式：
```json
{{
    "reason": "需要搜索的原因",
    "action": "searxng_search",
    "action_input": "搜索关键词"
}}
```

选项2：如果已经知道答案或不需要搜索，请使用此格式：
```json
{{
    "action": "Final Answer",
    "answer": "你的回答内容"
}}
```

重要：只返回JSON格式，不要添加任何解释或其他文字。
"""

# 构建完整的聊天提示词模板
# 包含系统消息、用户消息和工具调用历史
prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": "你是非常强大的助手，你可以使用各种工具来完成人类交给的问题和任务。"
    },
    {
        "role": "user",
        "content": promptTemplate  # 主要的指令模板
    },
    # 工具调用历史的占位符，optional=True表示可以为空
    MessagesPlaceholder(variable_name="agent_scratchpad", optional=True)
])


# ================================
# 4. 工具定义 - 网络搜索功能
# ================================

@tool
def searxng_search(query: str) -> list:
    """
    使用SearXNG搜索引擎进行网络搜索

    参数:
        query (str): 搜索关键词或问题

    返回:
        list: 搜索结果列表，每个结果包含title、content、url字段

    注意:
        - 需要本地运行SearXNG服务在6688端口
        - 使用Bing搜索引擎
        - 限制返回前3个结果
    """
    # SearXNG服务的搜索端点URL
    SEARXNG_URL = "http://localhost:6688/search"  # 正确的搜索端点

    print(f"🔍 [工具-searxng_search] 开始搜索: {query}")

    # 构建搜索参数
    params = {
        "q": query,                 # 搜索查询
        "format": "json",           # 返回JSON格式
        "engines": "bing",          # 使用Bing搜索引擎
        "language": "zh-CN"         # 中文搜索
    }

    try:
        print(f"🌐 [工具-searxng_search] 请求URL: {SEARXNG_URL}")
        print(f"📋 [工具-searxng_search] 请求参数: {params}")

        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        print(f"📡 [工具-searxng_search] 响应状态码: {response.status_code}")

        if response.status_code == 200:
            res = response.json()
            print(f"📊 [工具-searxng_search] 原始响应: {res}")

            # 检查响应结构
            if "results" not in res:
                print(f"⚠️  [工具-searxng_search] 响应中没有 'results' 字段")
                return []

            if not res["results"]:
                print(f"⚠️  [工具-searxng_search] 搜索结果为空")
                return []

            result = []
            for item in res["results"]:
                # 添加字段存在性检查
                title = item.get("title", "无标题")
                content = item.get("content", "无内容描述")
                url = item.get("url", "无链接")

                result.append({
                    "title": title,
                    "content": content,
                    "url": url
                })

            print(f"✅ [工具-searxng_search] 处理后的搜索结果: {result}")
            return result[:3]
        else:
            print(f"❌ [工具-searxng_search] HTTP错误: {response.status_code}")
            print(f"📄 [工具-searxng_search] 响应内容: {response.text}")
            return []

    except requests.exceptions.ConnectionError:
        print(f"❌ [工具-searxng_search] 连接错误: 无法连接到 SearXNG 服务")
        return []
    except requests.exceptions.Timeout:
        print(f"❌ [工具-searxng_search] 超时错误: 请求超时")
        return []
    except Exception as e:
        print(f"❌ [工具-searxng_search] 未知错误: {e}")
        return []


tools = [searxng_search]
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=",".join([tool.name for tool in tools])
)
print(prompt)

def jsonParser(message):
    """安全的JSON解析器，带有错误处理"""
    try:
        print(f"🔍 [JSON解析] 原始LLM输出: {message.content}")

        # 检查内容是否为空
        if not message.content or message.content.strip() == "":
            print("⚠️  [JSON解析] LLM返回空内容")
            return {"action": "Final Answer", "answer": "抱歉，我无法获取相关信息。"}

        # 尝试解析JSON
        result = parse_json_markdown(message.content)
        print(f"✅ [JSON解析] 解析成功: {result}")
        return result

    except Exception as e:
        print(f"❌ [JSON解析] 解析失败: {e}")
        print(f"📄 [JSON解析] 原始内容: '{message.content}'")

        # 返回一个默认的JSON结构
        return {
            "action": "Final Answer",
            "answer": f"解析响应时出现错误，原始回复: {message.content}"
        }
# ================================
# LangGraph 工作流实现
# ================================

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

# 定义状态结构
class AgentState(TypedDict):
    input: str                                          # 用户输入
    agent_scratchpad: Annotated[List, operator.add]    # 工具调用历史
    output: str                                         # 最终输出
    llm_decision: dict                                  # LLM决策结果
    tool_result: str                                    # 工具执行结果

def start_node(state: AgentState):
    """
    起始节点：处理用户输入，生成LLM响应
    """
    print(f"🚀 [start_node] 处理用户输入: {state['input']}")

    # 构建agent_scratchpad的消息列表
    from langchain_core.messages import HumanMessage, AIMessage

    scratchpad_messages = []
    if state.get("agent_scratchpad"):
        for item in state["agent_scratchpad"]:
            # 将工具调用历史转换为消息格式
            scratchpad_messages.append(HumanMessage(content=f"工具调用记录: {item}"))

    print(f"📋 [start_node] 工具调用历史: {len(scratchpad_messages)} 条记录")

    # 调用LLM链
    result = chain.invoke({
        "input": state["input"],
        "agent_scratchpad": scratchpad_messages
    })

    print(f"💭 [start_node] LLM决策结果: {result}")

    # 将LLM的决策结果存储到状态中
    return {
        "llm_decision": result,
        "agent_scratchpad": state.get("agent_scratchpad", [])
    }

def isUseTool(state: AgentState):
    """
    判断节点：决定是否使用工具
    """
    llm_decision = state.get("llm_decision", {})
    action = llm_decision.get("action", "")

    print(f"🤔 [isUseTool] 判断是否使用工具...")
    print(f"📋 [isUseTool] 决策动作: {action}")

    if action == "Final Answer":
        print("✅ [isUseTool] 决策: 直接回复用户")
        return "final_answer"
    elif action in [tool.name for tool in tools]:
        print(f"🔧 [isUseTool] 决策: 使用工具 '{action}'")
        return "use_tool"
    else:
        print(f"⚠️  [isUseTool] 未知动作: {action}，默认直接回复")
        return "final_answer"

def tool_node(state: AgentState):
    """
    工具节点：执行工具调用
    """
    llm_decision = state.get("llm_decision", {})
    action = llm_decision.get("action", "")
    action_input = llm_decision.get("action_input", "")

    print(f"🔧 [tool_node] 执行工具: {action}")
    print(f"📥 [tool_node] 工具输入: {action_input}")

    # 查找并执行对应的工具
    tool_result = None
    for tool in tools:
        if tool.name == action:
            try:
                tool_result = tool.invoke(action_input)
                print(f"✅ [tool_node] 工具执行成功")
                print(f"📤 [tool_node] 工具输出: {tool_result}")
                break
            except Exception as e:
                print(f"❌ [tool_node] 工具执行失败: {e}")
                tool_result = f"工具执行失败: {e}"
                break

    if tool_result is None:
        print(f"❌ [tool_node] 找不到工具: {action}")
        tool_result = f"找不到工具: {action}"

    # 将工具调用记录添加到scratchpad
    tool_record = f"使用工具 {action}({action_input}) -> {tool_result}"

    return {
        "agent_scratchpad": [tool_record],
        "tool_result": tool_result
    }

def final_answer_node(state: AgentState):
    """
    最终回答节点：生成最终输出
    """
    llm_decision = state.get("llm_decision", {})

    if llm_decision.get("action") == "Final Answer":
        final_output = llm_decision.get("answer", "抱歉，我无法提供答案。")
    else:
        final_output = "处理完成，但没有找到最终答案。"

    print(f"🎯 [final_answer_node] 生成最终回答: {final_output}")

    return {
        "output": final_output
    }

# ================================
# 构建 LangGraph 工作流
# ================================

# 创建状态图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("start_node", start_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("final_answer_node", final_answer_node)

# 设置入口点
workflow.set_entry_point("start_node")

# 添加条件边：从start_node根据isUseTool的判断结果路由
workflow.add_conditional_edges(
    "start_node",
    isUseTool,
    {
        "use_tool": "tool_node",
        "final_answer": "final_answer_node"
    }
)

# 添加边：工具执行完成后回到start_node继续处理
workflow.add_edge("tool_node", "start_node")

# 添加边：最终回答后结束
workflow.add_edge("final_answer_node", END)

# 编译工作流
app = workflow.compile()

# 定义LLM处理链（在工作流编译后定义，避免循环引用）
chain = prompt | llm | jsonParser

# ================================
# 测试工作流
# ================================

print("=" * 80)
print("🚀 开始测试 LangGraph 工具调用工作流")
print("=" * 80)

# 测试用例1：需要搜索的问题
test_input = {
    "input": "刘亦菲最近有什么活动?",
    "agent_scratchpad": [],
    "output": "",
    "llm_decision": {},
    "tool_result": ""
}

print(f"\n📝 测试输入: {test_input['input']}")
print("-" * 60)

try:
    # 执行工作流
    for step in app.stream(test_input):
        print(f"\n📊 [执行步骤] {step}")
        print("-" * 40)

    # 获取最终结果
    final_result = app.invoke(test_input)

    print("\n" + "=" * 80)
    print("🎯 最终结果:")
    print("=" * 80)
    print(f"用户问题: {final_result.get('input', '')}")
    print(f"最终回答: {final_result.get('output', '')}")
    print(f"工具调用历史: {final_result.get('agent_scratchpad', [])}")

except Exception as e:
    print(f"❌ 工作流执行失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ 工作流测试完成")
print("=" * 80)
