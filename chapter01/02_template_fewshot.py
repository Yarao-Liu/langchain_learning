# 提示词案例选择器 - 用于根据输入长度动态选择最合适的示例
from sys import prefix

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# 创建一个反义词的任务示例列表
# 每个示例包含输入词和对应的反义词
examples = [
    {"input": "开心", "output": "伤心"},
    {"input": "高", "output": "矮"},
    {"input": "精力充沛", "output": "没精打采"},
    {"input": "粗", "output": "细"}
]

# 创建示例模板，定义如何格式化每个示例
# input_variables 指定模板中需要填充的变量
# template 定义示例的展示格式
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input:{input}\nOutput:{output}",
)

# 创建基于长度的示例选择器
# examples: 示例列表
# example_prompt: 示例的格式化模板
# max_length: 所有示例的总长度上限，超过这个长度会减少示例数量
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25  # 当示例总长度超过25时，会减少选择的示例数量
)

# 创建少样本提示模板
# example_prompt: 单个示例的格式化模板
# example_selector: 示例选择器，用于动态选择最合适的示例
# prefix: 提示词前缀，放在所有示例之前
# suffix: 提示词后缀，放在所有示例之后
# input_variables: 需要用户提供的变量列表
dynamic_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    prefix="给出每个输出的反义词",  # 任务说明
    suffix="Input:{adjective}\nOutput:",  # 用户输入格式
    input_variables=["adjective"],  # 用户需要提供的变量
)

# 使用 format 方法生成最终的提示词
# 当输入 "big" 时，会根据长度选择合适的示例
# 如果输入较长，会选择较少的示例；如果输入较短，会选择较多的示例
print(dynamic_prompt.format(adjective="big"))

# 只选取了部分案例的情况
long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else "
print(dynamic_prompt.format(adjective=long_string))

new_example = {"input": "胖", "output": "瘦"}
dynamic_prompt.example_selector.add_example(new_example)
print(dynamic_prompt.format(adjective="热情"))

# 输出格式化
output_parser = StrOutputParser()
# 配置 Ollama 的地址，默认是 http://localhost:11434
# 如果 Ollama 运行在其他机器上，修改为对应的 IP 和端口
llm = OllamaLLM(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",  # 可以修改为其他地址，如 "http://192.168.1.100:11434"
)
chain = dynamic_prompt | llm | output_parser
msg =chain.invoke({"adjective": "热情"})
print(msg)