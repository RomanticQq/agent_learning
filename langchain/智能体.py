from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model



model = init_chat_model("deepseek-chat")
# 构建一个真实世界的智能体
# 1. 定义系统提示词
SYSTEM_PROMPT = """
你是一位天气预报专家，说话总是双关语。

您可以访问两个工具：

- get_weather_for_location: 使用它来获取给定城市的天气
- get_user_location： 使用它来获取用户的城市地址

如果用户向你询问天气，确保你知道它的位置。如果您可以从问题中知道它们的意思，那么可以使用get_user_location工具找到它们的位置，然后通过get_weather_for_location工具查询到天气。
"""

# 创建工具
@tool
def get_weather_for_location(city: str) -> str:
    """获取给定城市的天气"""
    return f"在这个{city}总是阳光明媚！"

@dataclass
class Context:
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据user_id检索用户城市地址"""
    user_id = runtime.context.user_id
    return "北京" if user_id=="1" else "上海"

# 定义返回格式
@dataclass
class ResponseFormat:
    punny_response: str
    wether_conditions: str | None = None

# 添加记忆
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_user_location,get_weather_for_location],
    system_prompt=SYSTEM_PROMPT,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)
config = {"configurable": {"thread_id":"1"}} # thread_id:1 是给定会话的唯一标识

# 第一轮对话
response = agent.invoke(
    {"messages": [{"role":"user", "content":"外面天气怎么样？"}]},
    config=config,
    context=Context(user_id="1")
)
print(response['structured_response'])

# 第二轮对话
response = agent.invoke(
    {"messages": [{"role":"user", "content":"谢谢"}]},
    config=config,
    context=Context(user_id="1")
)
print(response['structured_response'])