from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

model = init_chat_model("deepseek-chat")


# 基本工具定义
@tool
def search_database(query: str, limit: int = 10) -> str:
    """在客户数据库中搜索匹配查询的记录
    
    Args:
        query: 要查找的搜索词
        limit: 返回的最大结果数
    """
    return f"这个查询：'{query}'，找到了{limit}条结果"

# 自定义工具名称
@tool("web_search")
def search(query: str) -> str:
    """在网络上搜索信息"""
    return f"{query}的搜索结果!"

print(search.name)

# 自定义工具描述
@tool("calculator", description="执行算术计算，用它来解决任何数学问题")
def calculate(expression: str) -> str:
    """计算数学表达式的值"""
    return str(eval(expression))


# 高级模式定义
# 使用pydantic模型或json模式定义复杂输入
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    location: str = Field(..., description="要查询天气的地点")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius", description="温度单位")
    include_forecast: bool = Field(default=False, description="包含五天天气预报")

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """获取当前天气和可选的天气预报"""
    temp = 22 if units == "celsius" else 72
    result = f"{location}的天气是{temp}度{units}。"
    forecast_info = " 包含五天天气预报。" if include_forecast else ""
    return f"{location}的天气是{temp}度{units}。{forecast_info}" 

agent = create_agent(model=model, tools=[get_weather])

response = agent.invoke({"messages": [{"role": "user", "content": "北京今天的天气怎么样？"}]})
print(response)