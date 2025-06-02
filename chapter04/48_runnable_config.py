"""
温度参数示例
这个脚本展示了如何使用LangChain的ConfigurableField来动态调整模型参数。
主要功能：
1. 展示不同temperature值对模型输出的影响
2. 使用ConfigurableField实现参数配置
3. 对比确定性输出和随机性输出
"""

from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器
from langchain_core.runnables import ConfigurableField
from langchain_ollama import OllamaLLM  # Ollama语言模型
import time  # 用于添加延迟

# 创建输出解析器实例
# 用于将模型输出转换为字符串格式
output_parser = StrOutputParser()

# 初始化Ollama大语言模型
# 不设置默认temperature，让它在每次调用时都可以被配置
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 3B模型
    base_url="http://localhost:11434",  # Ollama服务地址
)

# 配置可调整的字段
# 允许在运行时动态调整temperature参数
llm.configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="temperature",
        description="控制输出的随机性：0表示完全确定性，1表示最大随机性",
    )
)

# 使用完全相同的提示来测试temperature的效果
prompt = """请生成一个1到100000之间的随机整数。
要求：
1. 只返回数字，不要有任何其他文字
2. 确保数字在1到100000之间
数字："""

# 测试temperature=0（完全确定性）
print("=== 使用temperature=0（完全确定性）===")
for i in range(3):
    print(f"第{i+1}次尝试：")
    print(llm.with_config(configurable={"llm_temperature": 0}).invoke(prompt))
    print()
    time.sleep(2)  # 添加短暂延迟

# 测试temperature=1（最大随机性）
print("=== 使用temperature=1（最大随机性）===")
for i in range(3):
    print(f"第{i+1}次尝试：")
    print(llm.with_config(configurable={"llm_temperature": 1}).invoke(prompt))
    print()
    time.sleep(2)  # 添加短暂延迟