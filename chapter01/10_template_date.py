from datetime import datetime

from langchain_core.prompts import PromptTemplate


def _get_datetime():
    """获取当前日期时间"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


# 方法1：使用 partial 方法
# 创建一个基础模板，然后使用 partial 方法预设部分变量
prompt = PromptTemplate(
    template="给我讲一个关于{date}天的{adjective}故事",  # 修复模板字符串格式
    input_variables=["adjective", "date"],
)
# 使用 partial 方法预设 date 变量
# 注意：这里传入的是函数对象，而不是函数调用
partial_prompt = prompt.partial(date=_get_datetime)
# 格式化时只需要提供剩余的变量
print("方法1输出：")
print(partial_prompt.format(adjective="有趣"))


# 方法2：在创建模板时直接设置部分变量
# 使用 partial_variables 参数预设变量
prompt2 = PromptTemplate(
    template="给我讲一个关于{date}天的{adjective}故事",  # 修复模板字符串格式
    input_variables=["adjective"],  # 只需要列出未预设的变量
    partial_variables={"date": _get_datetime},  # 预设 date 变量
)
print("\n方法2输出：")
print(prompt2.format(adjective="有趣"))