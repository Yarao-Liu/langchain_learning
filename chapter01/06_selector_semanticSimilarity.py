# 导入必要的库
# PromptTemplate: 用于创建单个示例的格式化模板
from langchain.prompts import PromptTemplate
# Chroma: 用于向量存储和检索的数据库
from langchain_community.vectorstores import Chroma
# SemanticSimilarityExampleSelector: 基于语义相似度选择示例的选择器
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
# StrOutputParser: 用于解析模型输出为字符串
from langchain_core.output_parsers import StrOutputParser
# FewShotPromptTemplate: 用于创建少样本学习提示模板
from langchain_core.prompts import FewShotPromptTemplate
# OllamaLLM: 用于调用本地Ollama大语言模型
# OllamaEmbeddings: 用于生成文本的向量表示
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# 创建一个多步骤推理的示例列表
# 每个示例包含问题和详细的推理过程
# 这些示例将用于少样本学习，帮助模型理解如何通过多步骤推理来回答问题
examples = [
    {
        "question": "乾隆和鸛操谁活得更久？",
        "answer": """
        这里是否需要跟进问题：是的。
        追问：乾隆去世时几岁？
        中间答案：乾隆去世时87岁。
        追问：曹操去世时几岁？
        中间答案：曹操去世时66岁：
        所以最终答案是：乾隆
        """,
    },
    {
        "question": "小米手机的创始人什么时候出生？",
        "answer": """
            这里是否需要跟进问题：是的。
            追问：小米手机的创始人是谁？
            中间答案：小米手机由雷军创立。
            跟进：雪军什么时候出生？
            中间答案：雪军出生于1969年12月16日。
            所以最终的答案是：1969年12月16日
        """
    },
    {
        "question": "乔治·华盛顿的外祖父是谁？",
        "answer": """
    这里是否需要跟进问题：是的。
    追问：乔治·华盛顿的母亲是谁？
    中间答案：乔治·华盛顿的母亲是玛丽·鲍尔·华盛顿
    追问：玛丽·鲍尔·华盛顿的父亲是谁？
    中间答案：玛丽·鲍尔·华盛顿的父亲是约瑟夫·鲍尔。
    所以最终答案是：约瑟夫·鲍尔
"""
    },
    {
        "question": "《大白鲨》和《皇家赌场》的导演是同一个国家的吗？",
        "answer": """
    这里是否需要跟进问题：是的。
    追问：《大白鲨》的导演是谁？
    中间答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
    追问：史蒂文·斯皮尔伯格来自哪里？
    中间答案：美国。
    追问：皇家赌嗜场的导演是谁？
    中间答案：《皇家赌场》的导演是马丁·坎贝尔。
    跟进：马丁·坎贝尔来自哪里？
    中间答案：新西兰。
    所以最终的答案是：不会
    """
    }
]

# 创建示例模板，定义如何格式化每个示例
# input_variables: 指定模板中需要填充的变量名
# template: 定义示例的展示格式，使用{变量名}作为占位符
examples_prompt = PromptTemplate(input_variables=["question","answer"], template="Question:{question}\nAnswer:{answer}")

# 测试示例模板的格式化效果
print(examples_prompt.format(**examples[0]))

# 初始化Ollama大语言模型
# model: 使用Qwen 2.5 7B模型，这是一个支持中文的模型
# base_url: Ollama服务的地址，默认本地运行
llm = OllamaLLM(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",  # 可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 初始化Ollama嵌入模型
# 使用中文优化的BGE模型来生成文本的向量表示
embeddings = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")

# 创建基于语义相似度的示例选择器
# examples: 示例列表，用于训练和选择
# embeddings: 用于将文本转换为向量表示的模型
# vectorstore_cls: 使用Chroma作为向量存储后端
# k: 每次选择1个最相关的示例
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=1
)

# 测试示例选择器
question = "李白和白居易谁活得更久?"
print(f"与输入最相似的示例:{example_selector.select_examples({'question':question})}")

# 创建少样本提示模板
# example_selector: 用于动态选择最相关的示例
# example_prompt: 定义单个示例的展示格式
# prefix: 提示词前缀，说明任务目标
# suffix: 提示词后缀，定义用户输入的格式
# input_variables: 需要用户提供的变量列表
last_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=examples_prompt,
    prefix="根据案例的方式回答问题",  # 任务说明
    suffix="Question:{input}",  # 用户输入格式
    input_variables=["input"]  # 用户需要提供的变量
)

# 创建输出解析器，将模型输出转换为字符串
output_parser = StrOutputParser()

# 创建处理链
# 1. 使用提示模板生成提示词
# 2. 将提示词发送给语言模型
# 3. 解析模型输出
chain = last_prompt | llm | output_parser

# 打印完整的提示模板
print(last_prompt)

# 执行处理链，获取结果
msg = chain.invoke({"input": "李白和白居易谁活得更久?"})
print(msg)