from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

model = ChatOpenAI(
    model="qwen3.5-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

@tool
def search(query: str) -> str:
    """搜索信息。"""
    return f"结果：{query}"

@tool
def get_weather(location: str) -> str:
    """获取位置的天气信息。"""
    return f"{location}的天气是晴天，温度25度"

agent = create_agent(model=model, tools=[search, get_weather])
response = agent.invoke({"messages": [{"role": "user", "content": "郑州今天天气怎么样？"}]})
print(response)