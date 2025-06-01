from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM

loader = PyPDFLoader("text/01.pdf")

pages = loader.load_and_split()
print(pages)

template = """
{context}
总结上面的文档
"""

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
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm |output_parser

res = chain.invoke({"context": pages})
print(res)