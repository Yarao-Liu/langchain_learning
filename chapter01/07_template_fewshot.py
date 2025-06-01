# 导入必要的库
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate  # 聊天提示模板和少样本提示模板
from langchain_ollama import OllamaLLM  # Ollama语言模型接口

# 定义少样本学习的示例
# 这些示例将帮助模型理解任务的要求
examples = [
    {"input": "2+2", "output": "4"},  # 示例1：简单的加法
    {"input": "3+2", "output": "5"},  # 示例2：另一个加法示例
]

# 创建示例提示模板
# 这个模板定义了如何格式化每个示例
example_prompts = ChatPromptTemplate.from_messages(
    [
        {
            "role": "user",  # 用户角色
            "content": "{input}"  # 用户输入
        },
        {
            "role": "assistant",  # 助手角色
            "content": "{output}"  # 期望的输出
        }
    ]
)

# 创建少样本提示模板
# 将示例和示例提示模板组合在一起
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,  # 示例列表
    example_prompt=example_prompts,  # 示例提示模板
)

# 打印格式化后的少样本提示
# 用于调试和查看提示的格式
print(few_shot_prompt.format())

# 创建最终的提示模板
# 包含系统消息、少样本示例和用户输入
final_prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",  # 系统角色
            "content": "你是一位非常厉害的数学天才。"  # 系统提示，定义模型角色
        },
        few_shot_prompt,  # 插入少样本示例
        {
            "role": "user",  # 用户角色
            "content": "{input}"  # 用户输入
        }
    ]
)

# 打印格式化后的最终提示
# 用于调试和查看完整提示的格式
print(final_prompt.format(input="2+2"))

# 初始化Ollama大语言模型
# 使用Qwen 2.5 7B模型，这是一个支持中文的模型
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 7B模型
    base_url="http://localhost:11434",  # Ollama服务地址，可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 创建输出解析器
# 将模型输出解析为字符串
output_parser = StrOutputParser()

# 创建处理链
# 将提示模板、语言模型和输出解析器组合在一起
chain = final_prompt | llm | output_parser

# 调用处理链并打印结果
# 测试数学计算能力
msg = chain.invoke({"input": "2^3"})  # 计算2的3次方
print(msg)  # 打印结果