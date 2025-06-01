# 提示词案例选择器 - 用于根据输入长度动态选择最合适的示例
# FewShotPromptTemplate: 用于创建少样本学习提示模板
# PromptTemplate: 用于创建单个示例的格式化模板
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
# FAISS: 用于高效的向量存储和检索
from langchain_community.vectorstores import FAISS
# MaxMarginalRelevanceExampleSelector: 用于选择最相关的示例，同时保持多样性
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
# StrOutputParser: 用于解析模型输出为字符串
from langchain_core.output_parsers import StrOutputParser
# OllamaLLM: 用于调用本地Ollama大语言模型
# OllamaEmbeddings: 用于生成文本的向量表示
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# 创建一个反义词的任务示例列表
# 每个示例包含输入词和对应的反义词
# 这些示例将用于少样本学习，帮助模型理解任务
examples = [
    {"input": "开心", "output": "伤心"},
    {"input": "高", "output": "矮"},
    {"input": "精力充沛", "output": "没精打采"},
    {"input": "粗", "output": "细"}
]

# 创建示例模板，定义如何格式化每个示例
# input_variables: 指定模板中需要填充的变量名
# template: 定义示例的展示格式，使用{变量名}作为占位符
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input:{input}\nOutput:{output}",
)

# 初始化 Ollama 嵌入模型
# 使用中文优化的BGE模型来生成文本的向量表示
# 这个模型专门针对中文文本进行了优化
embeddings = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")

# 创建基于最大边际相关性的示例选择器
# examples: 示例列表，用于训练和选择
# embeddings: 用于将文本转换为向量表示的模型
# vectorstore_cls: 使用FAISS作为向量存储后端
# k: 每次选择2个最相关的示例
example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=FAISS,
    k=2
)

# 创建少样本提示模板
# example_prompt: 定义单个示例的展示格式
# example_selector: 用于动态选择最相关的示例
# prefix: 提示词前缀，说明任务目标
# suffix: 提示词后缀，定义用户输入的格式
# input_variables: 需要用户提供的变量列表
mmr_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    prefix="给出每个输出的反义词",  # 任务说明
    suffix="Input:{adjective}\nOutput:",  # 用户输入格式
    input_variables=["adjective"],  # 用户需要提供的变量
)

# 测试提示模板
# 使用"big"作为输入，查看生成的提示词
print(mmr_prompt.format(adjective="big"))

# 添加新的示例到选择器
# 这可以动态扩展示例库，提高模型的适应性
new_example = {"input": "胖", "output": "瘦"}
mmr_prompt.example_selector.add_example(new_example)
print(mmr_prompt.format(adjective="热情"))

# 创建输出解析器，将模型输出转换为字符串
output_parser = StrOutputParser()

# 初始化Ollama大语言模型
# model: 使用Qwen 2.5 7B模型
# base_url: Ollama服务的地址，默认本地运行
llm = OllamaLLM(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",  # 可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 创建处理链
# 1. 使用提示模板生成提示词
# 2. 将提示词发送给语言模型
# 3. 解析模型输出
chain = mmr_prompt | llm | output_parser

# 执行处理链，获取结果
msg = chain.invoke({"adjective": "热情"})
print(msg)