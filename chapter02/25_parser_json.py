# 导入必要的库
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from pydantic.v1 import BaseModel, Field
from langchain.output_parsers import OutputFixingParser  # 导入输出修复解析器

# 初始化Ollama大语言模型
# 使用Qwen 2.5 7B模型，这是一个支持中文的模型
# base_url指定Ollama服务的地址，默认是本地服务
llm = OllamaLLM(
    model="qwen2.5:7b",  # 使用Qwen 2.5 7B模型
    base_url="http://localhost:11434",  # Ollama服务地址，可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 定义书籍信息的数据模型
# 使用Pydantic的BaseModel来定义数据结构
class Book(BaseModel):
    title: str = Field(description="书名")  # 书名字段
    author: str = Field(description="作者")  # 作者字段
    description: str = Field(description="简介")  # 简介字段

# 定义查询问题
query = "请给我介绍学习中国历史的经典书籍"

# 创建JSON输出解析器
# 使用Book模型作为解析目标
parser = JsonOutputParser(pydantic_object=Book)

# 获取解析器的格式说明
# 这会生成一个描述如何格式化输出的说明
format_instructions = parser.get_format_instructions()
print("解析器的格式说明：")
print(format_instructions)

# 自定义格式说明
# 这里提供了一个更简单的JSON结构说明
format_instructions = """
输出应该格式化为符合以下JSON结构的JSON：
`
{
    'title': '<UNK>',  # 书名
    'author': '<UNK>',  # 作者
    'description': '<UNK>',  # 简介
}
`
"""

# 创建提示模板
# 将格式说明和查询组合在一起
prompt = PromptTemplate(
    template="{format_instructions} \n {query} \n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

# 创建输出修复解析器
# 当原始解析器失败时，这个解析器会尝试修复输出
fixing_parser = OutputFixingParser.from_llm(
    parser=parser,  # 原始解析器
    llm=llm,  # 用于修复输出的语言模型
)

# 创建处理链
# 使用修复解析器替代原始解析器
chain = prompt | llm | fixing_parser

# 使用流式输出方式处理结果
# 这样可以逐步看到输出结果
print("\n处理结果：")
for chunk in chain.stream({"query": query}):
    print(chunk)

# 也可以使用普通方式获取结果
# result = chain.invoke({"query": query})
# print("\n完整结果：")
# print(result)