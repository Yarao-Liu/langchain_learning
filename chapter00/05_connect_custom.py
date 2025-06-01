import os
from typing import ClassVar, List, Dict

import dotenv
from langchain.llms.base import LLM
from langchain_core.messages.ai import AIMessage
from zhipuai import ZhipuAI

dotenv.load_dotenv()


class ChatGLM4(LLM):
    history: ClassVar[List[Dict[str, str]]] = []
    client: object = None

    def __init__(self):
        super().__init__()
        self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

    def _llm_type(self) -> str:
        return "ChatGLM4"

    def invoke(self, prompt: str, history=[]):
        if history is None:
            history = []
        if history is None:
            history = []
        history.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=history
        )
        result = response.choices[0].message.content
        return AIMessage(result)

    def _call(self, prompt, history=[]):
        return self.invoke(prompt, history)

    def stream(self, prompt, history=[]):
        if history is None:
            history = []
        history.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=history,
            stream=True
        )
        for chunk in response:
            yield chunk.choices[0].delta.content


llm = ChatGLM4()
msg = llm.invoke(prompt="如何鼓励自己减肥")
print(msg)
