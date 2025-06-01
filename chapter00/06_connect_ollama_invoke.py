from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
# 创建提示词模板
prompt = ChatPromptTemplate.from_template("请根据下面的主题做一首诗:{topic}")
# 输出格式化
output_parser = StrOutputParser()
# 配置 Ollama 的地址，默认是 http://localhost:11434
# 如果 Ollama 运行在其他机器上，修改为对应的 IP 和端口
llm = OllamaLLM(
    model="qwen3:4b",
    base_url="http://localhost:11434",  # 可以修改为其他地址，如 "http://192.168.1.100:11434"
)
# Example usage
if __name__ == "__main__":
    # 链式调用
    chain = prompt | llm | output_parser
    msg = chain.invoke({"topic":"茉莉花"})
    print(msg)
