# 导入必要的库
from re import search  # 导入正则表达式搜索功能
import os  # 导入操作系统相关功能
import ssl  # 导入SSL支持

from langchain_community.document_loaders import DirectoryLoader  # 目录加载器，用于批量加载目录中的文件

# 导入并下载NLTK的分词器
import nltk
from langchain_community.vectorstores import FAISS  # FAISS向量数据库，用于高效的相似度搜索
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # Ollama的嵌入模型和语言模型

# 处理SSL证书验证问题
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 下载必要的NLTK资源
try:
    nltk.download('punkt')  # 下载NLTK的punkt分词器，用于文本分割
    nltk.download('punkt_tab')  # 下载punkt_tab资源
    nltk.download('averaged_perceptron_tagger_eng')  # 下载英文词性标注器
except Exception as e:
    print(f"警告：NLTK资源下载失败，但程序将继续运行: {str(e)}")

# 确保目标目录存在
if not os.path.exists("./text"):
    os.makedirs("./text")
    print("已创建text目录")

# 创建目录加载器
# 用于加载指定目录下的所有文本文件
loader = DirectoryLoader(
    "./text",  # 指定要加载的目录路径
    glob="**/*.txt",  # 使用glob模式匹配所有.txt文件，**表示递归搜索子目录
    show_progress=True  # 显示加载进度
)

# 初始化Ollama嵌入模型
# 使用中文优化的BGE模型来生成文本的向量表示
embedding = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")

# 初始化Ollama大语言模型
# 使用Qwen 2.5 3B模型，这是一个支持中文的轻量级模型
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 3B模型，适合轻量级任务
    base_url="http://localhost:11434",  # Ollama服务地址，可以修改为其他地址，如 "http://192.168.1.100:11434"
)

try:
    # 加载所有匹配的文档
    # 将目录中的所有匹配文件加载为文档对象列表
    docs = loader.load()
    
    # 创建向量数据库
    # 使用FAISS存储文档的向量表示
    vectorStoreDB = FAISS.from_documents(
        docs,  # 文档列表
        embedding=embedding  # 使用的嵌入模型
    )

    # 创建检索器
    # 使用MMR（Maximum Marginal Relevance）算法进行文档检索
    # MMR算法在保持相关性的同时，确保检索结果的多样性
    retriever = vectorStoreDB.as_retriever(
        search_type="mmr",  # 使用MMR搜索算法
        search_kwargs={
            "k": 1,  # 设置返回最相关的1个文档
            "score_threshold": 0.3  # 设置相似度分数阈值，只返回相似度大于0.3的结果
        }
    )

    # 执行文档检索
    # 使用新的invoke方法替代已弃用的get_relevant_documents方法
    doc = retriever.invoke("爷爷干了什么?")
    print(doc)  # 打印检索到的相关文档

except Exception as e:
    print(f"错误：{str(e)}")
    print("请确保：")
    print("1. text目录中存在.txt文件")
    print("2. 文件内容格式正确")
    print("3. 网络连接正常")