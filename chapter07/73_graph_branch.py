"""
LangGraph 分支合并图示例 - 辩论系统
=====================================

这个示例展示了如何使用 LangGraph 创建一个分支合并的工作流：
1. 从一个主题开始
2. 分支到两个不同的观点（支持者和反对者）
3. 合并两个观点，生成最终的辩论结论

工作流程：
source (空节点) -> branch1 (支持者) -> sink (合并节点)
                -> branch2 (反对者) ->

作者：AI助手
日期：2024年
"""

# 导入必要的库
import os
import dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import MessageGraph


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
    model="Qwen/Qwen2.5-72B-Instruct",         # 使用通义千问模型
    temperature=0.3                             # 设置创造性参数
)

# ================================
# 定义三个不同角色的提示词模板
# ================================

# 支持者角色：为任何主题提供支持论据
fan_prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": "现在你放弃一切想法，假装成为一个任何主题的狂热粉丝和追随者，应该尽一切能力吹捧主题的观点"
    },
    MessagesPlaceholder(variable_name="messages")  # 动态插入消息历史
])
# 创建支持者链：提示词模板 + LLM
proponent = fan_prompt | llm

# 反对者角色：为任何主题提供反对论据
detractor_prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": "现在你放弃一切想法，假装成为一个任何主题的狂热 detractor 和反对者，应该尽一切能力抨击主题的观点"
    },
    MessagesPlaceholder(variable_name="messages")  # 动态插入消息历史
])
# 创建反对者链：提示词模板 + LLM
detractor = detractor_prompt | llm

# 综合分析角色：分析双方观点并给出结论
synthesis_prompt = ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": "你是一个专业的辩论家，需要根据辩论双方的观点，给出一个公正的结论"
    },
    MessagesPlaceholder(variable_name="messages")  # 动态插入消息历史
])


# ================================
# 定义消息合并函数
# ================================

def merge_messages(messages):
    """
    合并来自不同分支的消息

    参数:
        messages: 消息列表，包含原始主题和各分支的回复

    返回:
        包含合并后消息的列表，用于最终的综合分析

    消息结构:
        messages[0]: 原始主题消息
        messages[1:]: 来自各个分支的论据消息
    """
    # 调试输出：打印最后一条消息
    print("最后一条消息:", messages[-1])

    # 调试输出：如果有足够的消息，打印倒数第二条
    if len(messages) >= 2:
        print("倒数第二条消息:", messages[-2])

    # 提取原始主题内容
    original = messages[0].content

    # 将所有分支的论据合并成一个字符串
    # enumerate(messages[1:], 1) 从索引1开始为后续消息编号
    arguments = "\n".join([
        f" Argument{i}: {msg.content}" for i, msg in enumerate(messages[1:], 1)
    ])

    # 返回格式化后的消息，供综合分析使用
    return [
        HumanMessage(content=f"""Topic: {original}
Arguments:
{arguments}

哪一个论点更有说服力？请给出公正的分析。""")
    ]


# 创建最终的综合分析链（目前未使用，但保留以备后用）
final = merge_messages | synthesis_prompt | llm

# ================================
# 构建 LangGraph 分支合并图
# ================================

# 创建消息图实例
builder = MessageGraph()

def dictify(messages: list):
    """
    将消息列表转换为字典格式

    这是为了适配 LangChain 的提示词模板格式要求
    提示词模板中的 MessagesPlaceholder 需要 "messages" 键

    参数:
        messages: 消息列表

    返回:
        包含 "messages" 键的字典
    """
    return {"messages": messages}

# ================================
# 添加图节点
# ================================

# source 节点：起始节点，返回空列表作为初始状态
# lambda x: [] 表示忽略输入，返回空的消息列表
builder.add_node("source", lambda x: [])

# branch1 节点：支持者分支
# dictify | proponent 表示先转换格式，再调用支持者链
builder.add_node("branch1", dictify | proponent)

# branch2 节点：反对者分支
# dictify | detractor 表示先转换格式，再调用反对者链
builder.add_node("branch2", dictify | detractor)

# sink 节点：汇聚节点，合并所有分支的结果
builder.add_node("sink", merge_messages)

# ================================
# 设置图的连接关系
# ================================

# 设置入口点：从 source 节点开始
builder.set_entry_point("source")

# 添加边：定义节点之间的连接关系
builder.add_edge("source", "branch1")    # source -> branch1 (支持者分支)
builder.add_edge("source", "branch2")    # source -> branch2 (反对者分支)
builder.add_edge("branch1", "sink")      # branch1 -> sink (汇聚)
builder.add_edge("branch2", "sink")      # branch2 -> sink (汇聚)

# 注意：LangGraph 会自动处理分支的并行执行和结果合并
# sink 节点会等待所有输入分支完成后再执行

# ================================
# 编译并运行图
# ================================

# 编译图：将图结构转换为可执行的工作流
graph = builder.compile()

# 执行图：输入一个主题，获得辩论结果
print("=" * 60)
print("LangGraph 分支合并辩论系统演示")
print("=" * 60)

try:
    # 调用图，传入辩论主题
    result = graph.invoke(HumanMessage(content="躺平是当代人的解药"))
    print("\n最终结果:")
    print(result)

except Exception as e:
    print(f"执行过程中出现错误: {e}")
    print("请检查 API 配置和网络连接")

print("\n" + "=" * 60)
print("详细步骤执行过程:")
print("=" * 60)

# 使用 stream 方法获取每个步骤的详细信息
input_message = HumanMessage(content="躺平是当代人的解药")
steps = []

print("\n逐步执行过程:")
for step in graph.stream(input_message):
    print(f"当前步骤: {step}")
    steps.append(step)

# 打印最后一个步骤的输出结果
print("\n" + "=" * 40)
print("最后一个步骤的详细输出:")
print("=" * 40)

if steps:
    last_step = steps[-1]
    print(f"最后步骤完整内容: {last_step}")

    # 遍历最后一个步骤中的所有节点输出
    for node_name, node_output in last_step.items():
        print(f"\n节点 '{node_name}' 的输出:")
        print("-" * 30)

        if isinstance(node_output, list):
            for i, item in enumerate(node_output):
                if hasattr(item, 'content'):
                    print(f"  消息 {i+1}: {item.content}")
                else:
                    print(f"  项目 {i+1}: {item}")
        else:
            if hasattr(node_output, 'content'):
                print(f"  内容: {node_output.content}")
            else:
                print(f"  输出: {node_output}")
else:
    print("没有捕获到执行步骤")

print("\n" + "=" * 60)
print("演示结束")
