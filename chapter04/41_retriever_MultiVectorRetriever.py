"""
多向量检索器示例
这个脚本展示了如何使用LangChain的多向量检索器来处理和总结多个文档。
主要功能包括：
1. 文档加载和分割
2. 向量化存储
3. 文档总结生成
4. 多向量检索
5. 基于检索的问答
"""

# 导入必要的库
import uuid  # 用于生成唯一标识符，确保每个文档有唯一的ID

from langchain.retrievers import MultiVectorRetriever  # 多向量检索器，用于处理多个文档的向量检索
from langchain_community.document_loaders import TextLoader  # 文本加载器，用于加载文本文件
from langchain_chroma import Chroma  # Chroma向量数据库，用于存储和检索文档向量
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器，用于处理模型输出
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板，用于构建提示
from langchain_core.stores import InMemoryByteStore  # 内存字节存储，用于存储文档数据
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # Ollama的嵌入模型和语言模型
from langchain_text_splitters import CharacterTextSplitter  # 字符文本分割器，用于分割长文本

# 创建文本加载器列表
# 每个加载器对应一个文本文件，使用UTF-8编码以正确处理中文
loaders = [
    TextLoader("./text/ai_introduction.txt", encoding="utf-8"),  # 加载AI介绍文档
    TextLoader("./text/machine_learning.txt", encoding="utf-8"),  # 加载机器学习文档
]

# 加载所有文档
# 使用extend而不是append，因为load()返回的是文档列表
docs = []
for loader in loaders:
    docs.extend(loader.load())

# 创建输出解析器
# 用于将模型输出转换为字符串格式
output_parser = StrOutputParser()

# 初始化Ollama嵌入模型
# 使用中文优化的BGE模型来生成文本的向量表示
# 这个模型特别适合处理中文文本
embedding = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")

# 初始化Ollama大语言模型
# 使用Qwen 2.5 3B模型，这是一个支持中文的轻量级模型
# 适合在本地运行，响应速度快
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 3B模型
    base_url="http://localhost:11434",  # Ollama服务地址
)

# 创建向量存储
# 使用Chroma作为向量数据库，用于存储文档的向量表示
vectorstore = Chroma(
    collection_name="full_documents",  # 集合名称，用于组织存储的文档
    embedding_function=embedding  # 使用之前创建的嵌入模型
)

# 设置文档ID键名和存储
# 用于在存储中唯一标识每个文档
id_key = "doc_id"
store = InMemoryByteStore()  # 创建内存存储，用于临时存储文档数据

# 创建多向量检索器
# 用于处理多个文档的向量检索
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,  # 向量存储，用于存储和检索文档向量
    byte_store=store,  # 字节存储，用于存储文档数据
    id_key=id_key  # 文档ID键名，用于标识文档
)

# 为每个文档生成唯一ID
# 使用UUID确保ID的唯一性
doc_ids = [str(uuid.uuid4()) for _ in docs]
print("生成的文档ID：", doc_ids)

# 创建文本分割器
# 用于将长文本分割成较小的块，便于处理
child_text_spliter = CharacterTextSplitter(
    separator="\n\n",  # 使用双换行符作为分隔符
    chunk_size=100,    # 每个文本块的最大字符数
    chunk_overlap=10,  # 相邻文本块之间的重叠字符数，保持上下文连贯性
    length_function=len,  # 用于计算文本长度的函数
    is_separator_regex=False  # 分隔符是否为正则表达式
)

# 分割文档并添加元数据
# 将每个文档分割成小块，并为每个块添加元数据
sub_docs = []
for i, doc in enumerate(docs):
    _id = doc_ids[i]  # 获取当前文档的ID
    _sub_docs = child_text_spliter.split_documents([doc])  # 分割文档
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id  # 为每个分割后的文档添加ID
    sub_docs.extend(_sub_docs)  # 将分割后的文档添加到列表中

# 添加文档到向量存储
vectorstore.add_documents(sub_docs)

# 创建文档总结链
# 定义提示模板，用于生成文档总结
template = """
请总结以下文档的主要内容，保持简洁明了:

{doc}
"""

# 创建提示模板
# 使用模板创建提示，用于指导模型生成总结
prompt = ChatPromptTemplate.from_template(template)

# 创建处理链
# 定义函数来处理多个文档的总结
def process_docs(docs):
    """
    处理多个文档并生成总结
    
    Args:
        docs: 要处理的文档列表
        
    Returns:
        list: 包含每个文档总结的列表
    """
    summaries = []
    for doc in docs:
        # 对每个文档单独处理
        # 创建处理链：提示模板 -> 语言模型 -> 输出解析器
        summary = prompt | llm | output_parser
        # 处理文档并获取总结
        result = summary.invoke({"doc": doc.page_content})
        summaries.append(result)
    return summaries

# 创建问答链
# 定义问答提示模板
qa_template = """
基于以下文档内容回答问题。如果文档中没有相关信息，请明确说明。

文档内容：
{context}

问题：{question}

请给出详细的回答：
"""

# 创建问答提示模板
qa_prompt = ChatPromptTemplate.from_template(qa_template)

# 创建问答处理函数
def answer_question(question, k=3):
    """
    基于检索的文档回答问题
    
    Args:
        question: 用户问题
        k: 检索的文档数量
        
    Returns:
        str: 问题的答案
    """
    # 检索相关文档
    docs = retriever.invoke(question)
    
    # 合并文档内容
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 创建问答链
    qa_chain = qa_prompt | llm | output_parser
    
    # 生成答案
    answer = qa_chain.invoke({"context": context, "question": question})
    return answer

# 处理所有文档并打印结果
print("\n文档总结结果：")
summaries = process_docs(sub_docs)
for i, summary in enumerate(summaries, 1):
    print(f"\n文档 {i} 的总结：")
    print(summary)

# 测试问答功能
print("\n问答测试:")
test_questions = [
    "什么是机器学习？",
    "人工智能有哪些主要特点？",
    "机器学习的应用场景有哪些？"
]

for question in test_questions:
    print(f"\n问题:{question}")
    answer = answer_question(question)
    print(f"答案:{answer}")


