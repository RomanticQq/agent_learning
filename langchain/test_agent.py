# from langchain_openai import ChatOpenAI
# from langchain.agents import create_agent
# from langchain.tools import tool

# qwen = ChatOpenAI(
#     model="qwen3.5-plus",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


# agent = create_agent(qwen)
# response = agent.invoke({"messages": [{"role": "user", "content": "元旦节是几月几号？"}]})
# print(response)

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

agent = create_agent(model="deepseek-chat") # 这样使用需要安装pip install langchain-deepseek
response = agent.invoke({"messages": [{"role": "user", "content": "元旦节是几月几号？"}]})
print(response)