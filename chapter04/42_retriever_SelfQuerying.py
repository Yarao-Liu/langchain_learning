"""
自查询检索器示例
这个脚本展示了如何使用LangChain的自查询检索器来实现基于元数据的智能检索。
主要功能包括：
1. 文档加载和向量化
2. 元数据字段定义
3. 自查询检索
4. 基于自然语言的智能过滤
"""

# 导入必要的库
from langchain.chains.query_constructor.schema import AttributeInfo  # 用于定义元数据字段的结构
from langchain.retrievers import SelfQueryRetriever  # 自查询检索器，支持基于元数据的智能检索
from langchain_chroma import  Chroma  # Chroma向量数据库，用于存储和检索文档向量
from langchain_core.documents import Document  # 文档类，用于创建文档对象
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # Ollama的嵌入模型和语言模型

# 创建测试数据
# 每个文档包含电影简介和元数据（年份、导演、评分、类型）
docs = [
    Document(
        page_content="《肖申克的救赎》是一部关于希望和救赎的电影，讲述了银行家安迪被冤枉入狱后，在监狱中寻找自由的故事。",
        metadata={"year": 1994, "director": "弗兰克·德拉邦特", "rating": 9.7, "genre": "剧情"}
    ),
    Document(
        page_content="《教父》是一部黑帮题材的经典电影，讲述了科莱昂家族的故事，展现了权力、家族和背叛的主题。",
        metadata={"year": 1972, "director": "弗朗西斯·福特·科波拉", "rating": 9.6, "genre": "犯罪"}
    ),
    Document(
        page_content="《盗梦空间》是一部科幻悬疑电影，讲述了通过共享梦境来窃取他人潜意识中秘密的故事。",
        metadata={"year": 2010, "director": "克里斯托弗·诺兰", "rating": 9.3, "genre": "科幻"}
    ),
    Document(
        page_content="《阿甘正传》是一部励志电影，讲述了智商只有75的阿甘，通过自己的努力和善良，创造了一个个奇迹的故事。",
        metadata={"year": 1994, "director": "罗伯特·泽米吉斯", "rating": 9.5, "genre": "剧情"}
    ),
    Document(
        page_content="《泰坦尼克号》是一部爱情灾难电影，讲述了穷画家杰克和贵族少女露丝在泰坦尼克号上发生的凄美爱情故事。",
        metadata={"year": 1997, "director": "詹姆斯·卡梅隆", "rating": 9.4, "genre": "爱情"}
    )
]

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
# 将文档转换为向量并存储在Chroma数据库中
vectorstore = Chroma.from_documents(docs,embedding=embedding)

# 定义元数据字段信息
# 每个字段包含名称、描述和类型
metadata_field_info =[
    AttributeInfo(
        name="genre",  # 电影类型字段
        description='电影的类型:["科幻小说"，"喜剧"，"剧情片"，"惊悚片"，"爱情片"，"动作片"，"动画片"之一',
        type="string"),
    AttributeInfo(
        name="year",  # 发行年份字段
        description="电影上映年份",
        type="integer"),
    AttributeInfo(
        name="director",  # 导演字段
        description="电影导演的名字",
        type="string"),
    AttributeInfo(
        name="rating",  # 评分字段
        description="电影评分为l-l0",
        type="float")
]

# 定义文档内容描述
# 用于帮助模型理解文档内容的性质
document_content_description = "电影的简要概述"

# 创建自查询检索器
# 配置检索器以支持基于元数据的智能查询
retriever = SelfQueryRetriever.from_llm(
    llm=llm,  # 使用之前初始化的大语言模型
    vectorstore=vectorstore,  # 使用之前创建的向量存储
    document_contents=document_content_description,  # 文档内容描述
    metadata_field_info=metadata_field_info,  # 元数据字段信息
    enable_limit=True,  # 启用结果数量限制
    search_kwargs={"k":2}  # 设置返回结果数量为2
)

# 测试不同的查询
test_queries = [
    "1994年上映的电影",
    "诺兰导演的电影",
    "剧情类型的电影",
    "评分9.5的电影"
]

print("\n开始测试查询：")
for query in test_queries:
    print(f"\n查询：{query}")
    try:
        res = retriever.invoke(query)
        print("结果：")
        for doc in res:
            print(f"- {doc.page_content}")
            print(f"  元数据：{doc.metadata}")
    except Exception as e:
        print(f"查询出错：{str(e)}")
