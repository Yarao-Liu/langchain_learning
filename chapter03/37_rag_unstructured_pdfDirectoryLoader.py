# 导入必要的库
from langchain_community.document_loaders import PyPDFLoader  # PDF文件加载器
from langchain_community.vectorstores import FAISS  # FAISS向量数据库，用于高效的相似度搜索
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain_core.runnables import RunnableParallel, RunnablePassthrough  # 用于构建可运行链的工具
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # Ollama的嵌入模型和语言模型

# 创建PDF加载器
# 加载指定路径的PDF文件
loader = PyPDFLoader("text/01.pdf")

# 加载并分割PDF文档
# 将PDF文档分割成多个页面
pages = loader.load_and_split()
# print(pages)  # 可以取消注释来查看分割后的页面内容

# 初始化Ollama嵌入模型
# 使用中文优化的BGE模型来生成文本的向量表示
embedding = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")

# 创建向量数据库
# 使用FAISS存储文档的向量表示
vectorStoreDB = FAISS.from_documents(pages, embedding=embedding)

# 创建检索器
# 使用相似度分数阈值搜索方式
# 只返回相似度分数大于阈值的文档
retriever = vectorStoreDB.as_retriever(
    search_type="similarity_score_threshold",  # 使用相似度分数阈值搜索
    search_kwargs={"score_threshold": 0.3}  # 设置相似度阈值，只返回相似度大于0.3的结果
)

# 初始化Ollama大语言模型
# 使用Qwen 2.5 3B模型，这是一个支持中文的轻量级模型
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 3B模型，适合轻量级任务
    base_url="http://localhost:11434",  # Ollama服务地址，可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 定义提示模板
# 用于构建问答系统的提示
template = """
根据以下内容:
{context}
回答问题:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 创建检索和问题处理的并行运行器
# 同时处理上下文检索和问题传递
setup_and_retrieval = RunnableParallel(
    {
        "context": retriever,  # 使用检索器获取相关上下文
        "question": RunnablePassthrough(),  # 直接传递问题
    }
)

# 执行检索
# 获取与问题相关的文档
res = setup_and_retrieval.invoke("PMI有哪些影响力?")
print(res)  # 打印检索结果

# 创建输出解析器
# 用于将模型输出解析为字符串
output_parser = StrOutputParser()

# 构建完整的处理链
# 1. 检索相关文档
# 2. 构建提示
# 3. 使用语言模型生成回答
# 4. 解析输出
chain = setup_and_retrieval | prompt | llm | output_parser

# 执行完整的问答流程
# 使用构建的链处理问题
print(chain.invoke("PMI的组织文化?"))


