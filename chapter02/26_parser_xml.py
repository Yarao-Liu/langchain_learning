# 导入必要的库
from langchain_core.output_parsers import XMLOutputParser  # XML输出解析器
from langchain_core.prompts import PromptTemplate  # 提示模板
from langchain_ollama import OllamaLLM  # Ollama语言模型接口

# 初始化Ollama大语言模型
# 使用Qwen 2.5 3B模型，这是一个支持中文的轻量级模型
# base_url指定Ollama服务的地址，默认是本地服务
llm = OllamaLLM(
    model="qwen2.5:3b",  # 使用Qwen 2.5 3B模型，适合轻量级任务
    base_url="http://localhost:11434",  # Ollama服务地址，可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 定义XML格式说明
# 这个说明告诉模型如何格式化输出为XML结构
format_instructions = """
生成{query}的电影目录,生成内容以XML格式返回,不要返回其他多余信息。
示例:
<xml>
<movie>电影1</movie>
<movie>电影2</movie>
</xml>
"""

# 创建XML输出解析器
# 这个解析器会将模型的输出解析为XML格式
parser = XMLOutputParser()

# 创建提示模板
# 将查询和格式说明组合在一起
prompt = PromptTemplate(
    template="""{query} \n{format_instructions}""",  # 模板包含查询和格式说明
    input_variables=["query"],  # 输入变量是查询
    partial_variables={"format_instructions": format_instructions}  # 格式说明作为部分变量
)

# 创建处理链
# 将提示模板、语言模型和解析器组合在一起
# 处理流程：提示模板 -> 语言模型 -> XML解析器
chain = prompt | llm | parser

# 调用处理链并打印结果
# 查询冯小刚导演的电影
print(chain.invoke({"query": "冯小刚"}))