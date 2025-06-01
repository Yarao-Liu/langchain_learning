# 提示词案例选择器 - 用于根据输入长度动态选择最合适的示例
# FewShotPromptTemplate: 用于创建少样本学习提示模板
# PromptTemplate: 用于创建单个示例的格式化模板
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
# NGramOverlapExampleSelector: 基于n-gram重叠度选择示例的选择器
from langchain_community.example_selectors import NGramOverlapExampleSelector
# FAISS: 用于高效的向量存储和检索
from langchain_community.vectorstores import FAISS
# MaxMarginalRelevanceExampleSelector: 用于选择最相关的示例，同时保持多样性
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
# StrOutputParser: 用于解析模型输出为字符串
from langchain_core.output_parsers import StrOutputParser
# OllamaLLM: 用于调用本地Ollama大语言模型
# OllamaEmbeddings: 用于生成文本的向量表示
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# 创建一个英译中的任务示例列表
# 每个示例包含英文输入和对应的中文翻译
# 这些示例将用于少样本学习，帮助模型理解翻译任务
examples = [
    {"input": "See Spot run.", "output": "请参阅现场运行"},
    {"input": "My dog barks.", "output": "我的狗犬吠"},
    {"input": "cat can run", "output": "猫会跑"},
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

# 创建基于n-gram重叠度的示例选择器
# examples: 示例列表，用于训练和选择
# example_prompt: 示例的格式化模板
# threshold: 重叠度阈值，范围0-1，值越大要求重叠度越高
# 工作原理：
# 1. 将输入文本和示例文本分解为n-gram（n个词的组合）
# 2. 计算输入文本与每个示例的n-gram重叠度
# 3. 选择重叠度超过阈值的示例
# 4. 按重叠度从高到低排序
example_selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=1.0,  # 要求完全匹配
)

# 创建少样本提示模板
# example_prompt: 定义单个示例的展示格式
# example_selector: 用于动态选择最相关的示例
# prefix: 提示词前缀，说明任务目标
# suffix: 提示词后缀，定义用户输入的格式
# input_variables: 需要用户提供的变量列表
prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    prefix="为每个输入提供中文翻译",  # 任务说明
    suffix="Input:{sentence}\nOutput:",  # 用户输入格式
    input_variables=["sentence"],  # 用户需要提供的变量
)

# 测试提示模板
# 使用示例句子作为输入，查看生成的提示词
print(prompt.format(sentence="cat can run fast."))

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
chain = prompt | llm | output_parser

# 执行处理链，获取结果
msg = chain.invoke({"sentence": "cat can run fast."})
print(msg)