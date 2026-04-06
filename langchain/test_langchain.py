import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen3.5-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
response = llm.invoke("你是谁")
print(response.content)
