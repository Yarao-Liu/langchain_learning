import operator
import os
from typing import TypedDict, Annotated, List

import dotenv
from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

# 加载环境变量
dotenv.load_dotenv()

# 检查 API 密钥配置
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("错误: 未找到 OPENAI_API_KEY 环境变量")
    print("请在 .env 文件中设置: OPENAI_API_KEY=your_api_key_here")
    exit(1)

# 创建大语言模型实例
# 使用 SiliconFlow 提供的 API 接口，兼容 OpenAI 格式
llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1/",  # SiliconFlow API 地址
    model="Qwen/Qwen2.5-72B-Instruct",  # 使用通义千问模型
    temperature=0.3  # 设置创造性参数
)
output_parser = StrOutputParser()


class AssistantState(TypedDict):
    conversation: Annotated[List, operator.add]


assistant_prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": "你是一个非常强大的助手。"
    },
    MessagesPlaceholder(variable_name="conversation")
])

main_builder = StateGraph(AssistantState)

# 创建assistant节点的处理函数
def assistant_node(state: AssistantState):
    print(f"🤖 [主图-assistant节点] 开始处理非笑话请求...")
    message = state["conversation"][-1]
    print(f"📥 [主图-assistant节点] 处理消息: {message}")

    # 创建通用助手提示词
    general_prompt = ChatPromptTemplate.from_messages([
        {
            "role": "system",
            "content": "你是一个有用的AI助手，能够帮助用户解决各种问题。请根据用户的需求提供帮助。"
        },
        {
            "role": "user",
            "content": "{input}"
        }
    ])

    # 执行LLM调用
    chain = general_prompt | llm
    response = chain.invoke({"input": message})

    print(f"💬 [主图-assistant节点] 生成回复: {response.content[:50]}...")
    return {"conversation": [response.content]}

main_builder.add_node("assistant", assistant_node)


def get_user_message(state: AssistantState):
    last_message = state["conversation"][-1]
    return {"messages": [last_message]}


prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": "你是一个讲故事的助手,用一个笑话来回应，这是有史以来最好的笑话"
    },
    MessagesPlaceholder(variable_name="messages")
])

critic_prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": """
            {message}
            -----------------
            请提出对这个笑话的改建建议，使其成为有史以来最好的笑话。
        """
    }
])


def update(out):
    print(f"📝 [子图-tell_joke节点] 生成笑话: {out.content[:50]}...")
    return {"messages": [("assistant", out.content)]}


def replace_role(out):
    print(f"🔧 [子图-critique节点] 生成改进建议: {out.content[:50]}...")
    return {"messages": [HumanMessage(out.content)]}


class SubGraphState(TypedDict):
    messages: Annotated[List, operator.add]


def critiqueFn(state: SubGraphState):
    print(f"🔍 [子图-critique节点] 提取消息进行分析...")
    last_message = state["messages"][-1]
    # 处理元组格式的消息 ("role", "content")
    if isinstance(last_message, tuple):
        content = last_message[1]
    else:
        content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    print(f"📋 [子图-critique节点] 提取的内容: {content[:50]}...")
    return {"message": content}


builder = StateGraph(SubGraphState)
builder.add_node("tell_joke", prompt | llm | update)
builder.add_node("critique", critiqueFn | critic_prompt | llm | replace_role)


def route(state):
    message_count = len(state["messages"])
    next_node = END if message_count >= 3 else "critique"
    print(f"🚦 [子图-路由判断] 消息数量: {message_count}, 下一步: {next_node}")
    return next_node


builder.add_conditional_edges("tell_joke", route)
builder.add_edge("critique", "tell_joke")
builder.set_entry_point("tell_joke")
joke_graph = builder.compile()

# for step in joke_graph.stream({"messages": [("user", "请讲个笑话")]}):
#     print(step)





def route(state: AssistantState):
    print(f"🎯 [主图-路由判断] 开始意图识别...")
    message = state["conversation"][-1]
    print(f"📥 [主图-路由判断] 用户输入: {message}")

    assistant_prompt = ChatPromptTemplate.from_messages([
        {
            "role": "system",
            "content": """
            你是一个意图识别助手，能够识别以下意图:
            1.讲故事
            2.讲笑话
            3.AI绘画
            4.学习知识
            5.其他
            例如:
            用户输入:给我说个故事吧。
            1
            用户输入：给我画个美女图片
            3
            ================================
            用户输入:
            {input}
            请识别用户意图，返回上面意图的数字符号，只返回数字，不要添加其他内容。
            """
        }
    ])
    chain = assistant_prompt | llm | output_parser
    result = chain.invoke({"input": message})
    result = result.strip()
    print(f"🔍 [主图-路由判断] 意图识别结果: {result}")

    if result == "2":
        print("➡️  [主图-路由判断] 路由到: joke_graph (笑话子图)")
        return "joke_graph"
    else:
        print("➡️  [主图-路由判断] 路由到: assistant (助手节点)")
        return "assistant"

# 状态转换函数：将主图状态转换为子图状态
def convert_to_subgraph_state(state: AssistantState):
    last_message = state["conversation"][-1]
    print(f"🔄 [状态转换] 主图 → 子图: {last_message}")
    return {"messages": [("user", last_message)]}

# 状态转换函数：将子图状态转换回主图状态
def convert_from_subgraph_state(state: SubGraphState):
    print(f"🔄 [状态转换] 子图 → 主图")
    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, tuple):
            content = last_message[1]
        elif hasattr(last_message, 'content'):
            content = last_message.content
        else:
            content = str(last_message)
        print(f"📤 [状态转换] 返回内容: {content[:50]}...")
        return {"conversation": [content]}
    return {"conversation": []}

# 创建包装后的子图节点
def joke_graph_node(state: AssistantState):
    print(f"🎭 [主图-joke_graph节点] 开始执行笑话子图...")
    # 转换状态格式
    subgraph_input = convert_to_subgraph_state(state)
    # 执行子图
    result = joke_graph.invoke(subgraph_input)
    # 转换回主图状态格式
    final_result = convert_from_subgraph_state(result)
    print(f"✅ [主图-joke_graph节点] 子图执行完成")
    return final_result

main_builder.add_node("joke_graph", joke_graph_node)
main_builder.set_conditional_entry_point(route)
main_builder.set_finish_point("assistant")
main_builder.set_finish_point("joke_graph")
graph = main_builder.compile()

print("=" * 80)
print("🚀 测试1: 非笑话意图 (应该路由到assistant节点)")
print("=" * 80)

for step in graph.stream({"conversation": ["帮我点个外卖"]}):
    print(f"\n📊 [执行步骤] {step}")
    print("-" * 60)

print("\n" + "=" * 80)
print("🚀 测试2: 笑话意图 (应该路由到joke_graph子图)")
print("=" * 80)

for step in graph.stream({"conversation": ["讲个笑话"]}):
    print(f"\n📊 [执行步骤] {step}")
    print("-" * 60)

print("\n" + "=" * 80)
print("✅ 所有测试执行完成")
print("=" * 80)