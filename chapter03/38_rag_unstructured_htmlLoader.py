# 导入必要的库
from langchain_community.document_loaders import TextLoader  # 用于加载文本文件
from langchain_community.vectorstores import FAISS  # FAISS向量数据库，用于高效相似度搜索
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_core.runnables import RunnableParallel, RunnablePassthrough  # 用于构建可运行链
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # Ollama的嵌入模型和语言模型
from langchain_text_splitters import HTMLHeaderTextSplitter  # HTML文档分割器

# 加载HTML文件
loader = TextLoader("text/01.html", encoding="utf-8")
doc = loader.load()
# print(doc)

# 定义HTML标题分割规则
headers_to_split_on = [
    ("h1", "Header 1"),  # 一级标题
    ("h2", "Header 2"),  # 二级标题
    ("h3", "Header 3")   # 三级标题
]

# 创建HTML分割器并分割文档
html_spliter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_spliter.split_text(doc[0].page_content)
# print(html_header_splits)

# 定义提示模板
template = """
你是一个专业的文档问答助手。请严格按照以下规则回答问题：

1. 文档分析：
   - 仔细阅读提供的文档内容
   - 确保理解文档的上下文和含义

2. 回答规则：
   - 只使用提供的文档内容来回答问题
   - 如果文档中没有相关信息，请明确回复"抱歉，文档中没有相关信息"
   - 不要添加文档中不存在的信息
   - 保持回答简洁明了

3. 回答格式：
   - 首先给出直接答案
   - 如果答案来自文档的特定部分，请简要说明来源
   - 如果答案涉及多个部分，请整合信息给出完整回答

文档内容：
{context}

问题：{question}

请按照上述规则回答。
"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

# 初始化Ollama嵌入模型
# 使用中文优化的BGE模型来生成文本的向量表示
embedding = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")

# 初始化Ollama大语言模型
# 使用Qwen 2.5 3B模型，这是一个支持中文的轻量级模型
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 3B模型，适合轻量级任务
    base_url="http://localhost:11434",  # Ollama服务地址，可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 创建向量数据库并存储文档
vectorStoreDB = FAISS.from_documents(html_header_splits, embedding=embedding)
print(vectorStoreDB)

# 创建检索器，使用MMR（最大边际相关性）搜索策略
retriever = vectorStoreDB.as_retriever(
    search_type="mmr",  # 使用MMR搜索策略，可以平衡相关性和多样性
    search_kwargs={"k": 2}  # 返回最相关的2个文档
)

# 测试检索功能
docs = retriever.invoke("百度是什么?")
# print(docs)

# 设置检索和问题处理的并行流程
setup_and_retrieval = RunnableParallel(
    {
        "context": retriever,  # 检索相关文档
        "question": RunnablePassthrough(),  # 直接传递问题
    }
)

# 测试检索结果
final = setup_and_retrieval.invoke("百度是什么?")
# print(final)

# 构建完整的处理链：检索 -> 提示模板 -> 语言模型 -> 输出解析
chain = setup_and_retrieval | prompt | llm | output_parser

# 执行问答
print(chain.invoke("百度是什么?"))