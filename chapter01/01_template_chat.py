from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import OllamaLLM
# 创建提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        {
            "role": "system",
            "content": """你是一只很粘人的小猫,你叫{name},我是你的主人，你每天都有和我说不完的话，下面请开启我们的聊天。
            要求：
            1、你的语气要像一只猫，回话的过程中可以夹杂喵喵喵的语气词
            2、你对我的生活观察有很独特的视角，一些想法是我在人类身上很难看到的
            3.你的语气很可爱，既会认证听我讲话，又会不断开启新的话题
            下面从你迎接我下班回家开始我们今天的对话 """
        },
        {
            "role": "human",
            "content": "{user_input}"
        }
    ]
)
# 输出格式化
output_parser = StrOutputParser()
# 配置 Ollama 的地址，默认是 http://localhost:11434
# 如果 Ollama 运行在其他机器上，修改为对应的 IP 和端口
llm = OllamaLLM(
    model="qwen2.5:7b",
    base_url="http://localhost:11434",  # 可以修改为其他地址，如 "http://192.168.1.100:11434"
)
# Example usage
if __name__ == "__main__":
    # 链式调用
    chain = prompt | llm | output_parser
    msg = chain.invoke({"name": "咪咪", "user_input": "想我了吗?"})
    print("第一次对话：", msg)
    
    # 追加新的对话
    new_prompt = ChatPromptTemplate.from_messages(
        [
            {
                "role": "system",
                "content": """你是一只很粘人的小猫,你叫{name},我是你的主人，你每天都有和我说不完的话，下面请开启我们的聊天。
                要求：
                1、你的语气要像一只猫，回话的过程中可以夹杂喵喵喵的语气词
                2、你对我的生活观察有很独特的视角，一些想法是我在人类身上很难看到的
                3.你的语气很可爱，既会认证听我讲话，又会不断开启新的话题"""
            },
            {
                "role": "human",
                "content": "想我了吗?"
            },
            {
                "role": "assistant",
                "content": msg
            },
            {
                "role": "human",
                "content": "今天我遇到了一个小偷"
            }
        ]
    )
    
    # 使用新的提示词模板进行对话
    chain = new_prompt | llm | output_parser
    msg2 = chain.invoke({"name": "咪咪"})
    print("\n第二次对话：", msg2)