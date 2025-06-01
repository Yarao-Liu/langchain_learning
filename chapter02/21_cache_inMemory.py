from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import time

# 初始化Ollama大语言模型
# 使用Qwen 2.5 7B模型，这是一个支持中文的模型
llm = OllamaLLM(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",  # 可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 测试问题
question = "请问两只兔子有多少条腿?"

# 不使用缓存的测试
print("不使用缓存的测试：")
start_time = time.time()
response1 = llm.invoke(question)
end_time = time.time()
print(f"回答: {response1}")
print(f"耗时: {end_time - start_time:.2f} 秒\n")

# 设置缓存
set_llm_cache(InMemoryCache())

# 使用缓存的测试
print("使用缓存的测试：")
start_time = time.time()
response2 = llm.invoke(question)
end_time = time.time()
print(f"回答: {response2}")
print(f"耗时: {end_time - start_time:.2f} 秒\n")

# 再次使用缓存测试（应该会更快）
print("再次使用缓存的测试：")
start_time = time.time()
response3 = llm.invoke(question)
end_time = time.time()
print(f"回答: {response3}")
print(f"耗时: {end_time - start_time:.2f} 秒")