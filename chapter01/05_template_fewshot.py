from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate
from langchain_ollama import OllamaLLM

examples = [
    {
        "question": "乾隆和鸛操谁活得更久？",
        "answer": """
        这里是否需要跟进问题：是的。
        追问：乾隆去世时几岁？
        中间答案：乾隆去世时87岁。
        追问：曹操去世时几岁？
        中间答案：曹操去世时66岁：
        所以最终答案是：乾隆
        """,
    },
    {
        "question": "小米手机的创始人什么时候出生？",
        "answer": """
            这里是否需要跟进问题：是的。
            追问：小米手机的创始人是谁？
            中间答案：小米手机由雷军创立。
            跟进：雪军什么时候出生？
            中间答案：雪军出生于1969年12月16日。
            所以最终的答案是：1969年12月16日
        """
    },
    {
        "question": "乔治·华盛顿的外祖父是谁？",
        "answer": """
    这里是否需要跟进问题：是的。
    追问：乔治·华盛顿的母亲是谁？
    中间答案：乔治·华盛顿的母亲是玛丽·鲍尔·华盛顿
    追问：玛丽·鲍尔·华盛顿的父亲是谁？
    中间答案：玛丽·鲍尔·华盛顿的父亲是约瑟夫·鲍尔。
    所以最终答案是：约瑟夫·鲍尔
"""
    },
    {
        "question": "《大白鲨》和《皇家赌场》的导演是同一个国家的吗？",
        "answer": """
    这里是否需要跟进问题：是的。
    追问：《大白鲨》的导演是谁？
    中间答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。
    追问：史蒂文·斯皮尔伯格来自哪里？
    中间答案：美国。
    追问：皇家赌嗜场的导演是谁？
    中间答案：《皇家赌场》的导演是马丁·坎贝尔。
    跟进：马丁·坎贝尔来自哪里？
    中间答案：新西兰。
    所以最终的答案是：不会
    """
    }
]

examples_prompt = PromptTemplate(input_variables=["question","answer"], template="Question:{question}\nAnswer:{answer}")

print(examples_prompt.format(**examples[0]))

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=examples_prompt,
    suffix="Question:{input}",
    input_variables=["input"]
)

print(prompt.format(input="李白和白居易谁活得更久?"))

# 初始化Ollama大语言模型
# model: 使用Qwen 2.5 7B模型
# base_url: Ollama服务的地址，默认本地运行
llm = OllamaLLM(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",  # 可以修改为其他地址，如 "http://192.168.1.100:11434"
)

# 创建输出解析器，将模型输出转换为字符串
output_parser = StrOutputParser()
# 创建处理链
# 1. 使用提示模板生成提示词
# 2. 将提示词发送给语言模型
# 3. 解析模型输出
chain = prompt | llm | output_parser

# 执行处理链，获取结果
msg = chain.invoke({"input": "李白和白居易谁活得更久?"})
print(msg)