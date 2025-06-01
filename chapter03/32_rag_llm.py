from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM


template ="""
只根据以下文档回答问题:
{context}
问题:{question}
"""
prompt = ChatPromptTemplate.from_template(template)

output_parser= StrOutputParser()
# 初始化Ollama嵌入模型
# 使用中文优化的BGE模型来生成文本的向量表示
embedding = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5:latest")
# 初始化Ollama大语言模型
# 使用Qwen 2.5 7B模型，这是一个支持中文的模型
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 7B模型
    base_url="http://localhost:11434",  # Ollama服务地址，可以修改为其他地址，如 "http://192.168.1.100:11434"
)

vectorstore = FAISS.from_texts(
    ["小明在华为工作", "熊喜欢吃蜂蜜"],
    embedding=embedding
)
retriever= vectorstore.as_retriever()

setup_and_retrieval = RunnableParallel(
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
)
chain = setup_and_retrieval | prompt | llm | output_parser

print(chain.invoke("小明在哪里工作?"))
