# 导入必要的库
from langchain_community.vectorstores import Chroma  # 向量数据库，用于存储和检索文本向量
from langchain_core.example_selectors import SemanticSimilarityExampleSelector  # 语义相似度示例选择器
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate  # 提示模板
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # Ollama的嵌入模型和语言模型

# 定义示例数据集
# 包含数学计算和中文问答的示例
examples = [
    {"input": "2+2", "output": "4"},  # 数学示例1
    {"input": "2+3", "output": "5"},  # 数学示例2
    {"input": "2+4", "output": "6"},  # 数学示例3
    {"input": "牛对月亮说了什么?", "output": "什么都没有"},  # 中文问答示例1
    {"input": "给我写一首关于月亮的五言诗",  # 中文问答示例2
     "output": "月儿挂枝头，清辉洒人间。银盘如明镜，照亮夜归人。思绪随风舞，共赏中秋圆。"}
]

# 初始化Ollama嵌入模型
# 使用中文优化的BGE模型来生成文本的向量表示
embeddings = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")

# 将示例转换为向量
# 将每个示例的输入和输出连接起来，生成文本向量
to_vectorized = ["".join(example.values()) for example in examples]

# 创建向量数据库
# 使用Chroma存储文本向量和元数据
vectorstore = Chroma.from_texts(
    to_vectorized,  # 要向量化的文本
    embeddings,  # 使用的嵌入模型
    metadatas=examples  # 存储原始示例作为元数据
)

# 创建语义相似度示例选择器
# 用于根据输入选择最相似的示例
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,  # 向量数据库
    k=1  # 选择最相似的1个示例
)

# 测试示例选择器
# 查看对特定输入的示例选择结果
print(example_selector.select_examples({"input": "对牛弹琴"}))

# 创建少样本提示模板
# 使用语义相似度选择器来选择相关示例
few_shot_prompt = FewShotChatMessagePromptTemplate(
    input_variables=["input"],  # 输入变量
    example_selector=example_selector,  # 示例选择器
    example_prompt=ChatPromptTemplate(  # 示例提示模板
        [
            {
                "role": "user",  # 用户角色
                "content": "{input}",  # 用户输入
            },
            {
                "role": "assistant",  # 助手角色
                "content": "{output}",  # 期望的输出
            }
        ]
    )
)

# 测试少样本提示模板
# 查看格式化后的提示
print(few_shot_prompt.format(input="2+3?"))

# 创建最终的提示模板
# 包含系统消息、少样本示例和用户输入
final_prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",  # 系统角色
            "content": "你是一位非常厉害的数学天才,根据案例的方式回答问题。",  # 系统提示
        },
        few_shot_prompt,  # 插入少样本示例
        {
            "role": "user",  # 用户角色
            "content": "{input}",  # 用户输入
        }
    ]
)

# 测试最终提示模板
# 查看完整提示的格式
print(final_prompt.format(input="2+3?"))

# 初始化Ollama大语言模型
# 使用Qwen 2.5 7B模型，这是一个支持中文的模型
llm = OllamaLLM(
    model="qwen2.5:7b",  # 使用Qwen 2.5 7B模型
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
print(chain.invoke({"input": "2^3?"}))