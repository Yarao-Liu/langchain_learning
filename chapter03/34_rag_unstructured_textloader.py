# 导入必要的库
from langchain_community.document_loaders import TextLoader  # 文本加载器，用于加载文本文件
from langchain_community.vectorstores import FAISS  # FAISS向量数据库，用于高效的相似度搜索
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # Ollama的嵌入模型和语言模型

# 创建文本加载器
# 加载Ollama的升级日志文件
loader = TextLoader(
    "C:/Users/13439/AppData/Local/Ollama/upgrade.log",  # 日志文件路径
    encoding="utf-8"  # 使用UTF-8编码读取文件
)

# 加载文档
# 将文本文件加载为文档对象
doc = loader.load()
# print(doc)  # 可以取消注释来查看加载的文档内容

# 创建输出解析器
# 用于将模型输出解析为字符串
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

# 创建向量数据库
# 使用FAISS存储文档的向量表示
vectorStoreDB = FAISS.from_documents(
    doc,  # 文档列表
    embedding=embedding  # 使用的嵌入模型
)
print(vectorStoreDB)  # 打印向量数据库信息

# 执行相似度搜索
# 查找与查询最相关的文档
res = vectorStoreDB.similarity_search("ollama的版本是多少?")
print(res)  # 打印搜索结果

# 执行带分数的相似度搜索
# 返回文档及其相似度分数
res = vectorStoreDB.similarity_search_with_score("ollama的版本是多少?")
print(res)  # 打印带分数的搜索结果