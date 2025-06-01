from zhipuai import ZhipuAI
import os
import dotenv
dotenv.load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")
client = ZhipuAI(api_key=api_key)
prompt= "以色列为什么喜欢战争?"
response = client.chat.completions.create(
    model="glm-4",
    messages=[
        {"role":"user","content":"你好"},
        {"role":"assistant","content":"我是人工智能助手"},
        {"role":"user","content":prompt},
    ],
)
print(response)
print(response.choices[0].message.content)