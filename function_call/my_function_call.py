from openai import OpenAI
import os
import asyncio
import random
from datetime import datetime
import json
from baidusearch.baidusearch import search


# 模拟天气查询工具。返回结果示例：“北京今天是雨天。”
def get_current_weather(location: str) -> str:
    # 定义备选的天气条件列表
    weather_conditions = ["晴天", "多云", "雨天"]
    # 随机选择一个天气条件
    random_weather = random.choice(weather_conditions)
    # 返回格式化的天气信息
    return f"{location}今天是{random_weather}。"

def baidu_search(query: str, num_results: int = 3) -> str:
    """百度搜索工具

    Args:
        query (str): 搜索关键词

    Returns:
        str: 搜索结果
    """
    results = search(query, num_results=num_results)
    # 转换为json
    results = json.dumps(results, ensure_ascii=False)
    return results

# 查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“
def get_current_time() -> str:
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # 返回格式化后的当前时间
    return f"当前时间：{formatted_time}。"

tools = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "当你想知道现在的时间时非常有用。",
    }
}, {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "当你想查询指定城市的天气时非常有用。",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                }
            },
            "required": ["location"]
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "baidu_search",
        "description": "对于用户提出的问题，如果需要使用搜索引擎查询，请使用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                },
                "num_results": {
                    "type": "integer",
                    "description": "搜索结果数量",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
}]

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        # {"role": "user", "content": "Hello"},
        # {"role": "user", "content": "请问北京的天气如何？"},
        {"role": "user", "content": "请告诉我现在的时间，并且查询一下北京的天气，最后帮我搜索一下'人工智能的未来'。"},
    ]
response = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=messages,
    stream=False,
    tool_choice="auto",
    tools=tools,
)
tool_message = {
    "tool_call_id": "",
    "role": "tool",
    "name": "",
    "content": "",
    }

tool_mapping = {
    "get_current_time": get_current_time,
    "get_current_weather": get_current_weather,
    "baidu_search": baidu_search
}
message = response.choices[0].message
messages.append(message)
if message.tool_calls:
    for tool_call in message.tool_calls:
        print("工具调用：", tool_call)
        now_tool_message = tool_message.copy()
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id
        function = tool_mapping[function_name]
        function_args_dict = json.loads(function_args)
        function_result = function(**function_args_dict)
        now_tool_message["tool_call_id"] = function_id
        now_tool_message["name"] = function_name
        now_tool_message["content"] = function_result
        messages.append(now_tool_message)
    # print("工具调用完成，更新后的消息列表：", messages)

response = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=messages,
    stream=False,
    tool_choice="auto",
    tools=tools,
)

message = response.choices[0].message

if message.tool_calls:
    print("再次调用工具：", message.tool_calls)
else:
    print("最终回答：", message.content)