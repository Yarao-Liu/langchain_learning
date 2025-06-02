from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import HTMLHeaderTextSplitter

loader = TextLoader("text/01.html", encoding="utf-8")
doc = loader.load()
# print(doc)

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3")
]
html_spliter = HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits = html_spliter.split_text(doc[0].page_content)
print(html_header_splits)

template = """
只根据以下文档回答问题:
{context}
问题:{question}
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

vectorStoreDB = FAISS.from_documents(html_header_splits, embedding=embedding)
print(vectorStoreDB)
retriever = vectorStoreDB.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2}
)
docs = retriever.invoke("百度是什么?")
# print(docs)

setup_and_retrieval = RunnableParallel(
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
)

final = setup_and_retrieval.invoke("百度是什么?")
# print(final)

chain = setup_and_retrieval |prompt|llm| output_parser

print(chain.invoke("百度是什么?"))