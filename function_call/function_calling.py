from openai import AsyncOpenAI
import os
import asyncio
import random
from datetime import datetime
import json
from baidusearch.baidusearch import search


client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# 模拟天气查询工具。返回结果示例：“北京今天是雨天。”
async def get_current_weather(location: str) -> str:
    # 定义备选的天气条件列表
    weather_conditions = ["晴天", "多云", "雨天"]
    # 随机选择一个天气条件
    random_weather = random.choice(weather_conditions)
    # 返回格式化的天气信息
    return f"{location}今天是{random_weather}。"


async def baidu_search(query: str, num_results: int = 3) -> str:
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
async def get_current_time() -> str:
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

# 异步任务
async def function_calling(query: str) -> tuple[str, str, str, list, str]:
    """函数调用函数，采用流式输出，兼容普通问答

    Args:
        query (str): 用户输入的query

    Returns:
        function_name (str): 工具名称
        function_arguments (str): 工具入参
        fun_id (str): 工具ID，每次工具调用都会有一个编号id，同一个工具调用多次，id会不同
        origin_messages (list): 原始消息
        response_content (str): 回答
    """
    # 这个是大模型接受输入的标准化格式！直接背板！协议规定！
    # origin_message是一个list，往里面添加一条信息大模型就会多接收一条信息，讲白了多轮对话也是这个原理
    origin_messages = [{
        "role": "system",
        "content": "你是一个AI助手，请根据用户的问题给出回答，可以采用工具调用帮助回答问题"
    }, {
        "role": "user",
        "content": query
    }]

    # 这个就是给大模型一个输入，大模型给你一个输出的命令
    response = await client.chat.completions.create(model="qwen-plus",
                                                    messages=origin_messages,
                                                    tools=tools,
                                                    tool_choice="auto",
                                                    stream=True)
    function_name = ""
    function_arguments = ""
    response_content = ""
    fun_id = None
    first_chunk = True
    # 处理流式输出：当成标准模板背诵！
    async for chunk in response:
        if chunk.choices[0].delta.tool_calls:
            if first_chunk:  # 第一个chunk提取工具名称，同时开始累积函数入参
                function_name = chunk.choices[0].delta.tool_calls[
                    0].function.name
                function_arguments += chunk.choices[0].delta.tool_calls[
                    0].function.arguments
                fun_id = chunk.choices[0].delta.tool_calls[0].id
                first_chunk = False
            else:
                if chunk.choices[0].delta.tool_calls[0].function.arguments:
                    function_arguments += chunk.choices[0].delta.tool_calls[
                        0].function.arguments
        else:
            # 不是函数调用，正常回答
            if chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="", flush=True)

    # 返回工具名称、工具入参、回答
    return function_name, function_arguments, fun_id, origin_messages, response_content

# 大模型输出信息都是字符串，需要根据字符串信息执行函数
# 需要字符串到函数名称的映射 -> 使用字典实现
tool_mapping = {
    "get_current_time": get_current_time,
    "get_current_weather": get_current_weather,
    "baidu_search": baidu_search
}

# 工具调用完成后的结果是不是还要返回给大模型？就采用如下形式进行返回即可！
assistant_messages_template = {
    "content":
    "",
    "refusal":
    None,
    "role":
    "assistant",
    "audio":
    None,
    "function_call":
    None,
    "tool_calls": [{
        "id": "call_xxx",
        "function": {
            "arguments": "",
            "name": "",
        },
        "type": "function",
        "index": 0,
    }],
}


async def main():
    # 用户问题
    # query = "请帮我查一下今天北京天气"
    query = "获取当前时间"
    # query = "黑神话悟空是什么时候发售的"

    # 1. 大模型根据用户问题以字符串形式返回调用工具名称和入参
    
    # 重点关注origin_messages在怎么变化！origin_messages在初始化的时候包括一个system message和user message
    
    function_name, function_arguments, fun_id, origin_messages, response_content = await function_calling(
        query)

    if function_name:
        print(
            f"执行函数调用：工具名称：{function_name}，工具参数：{function_arguments}，工具调用id：{fun_id}"
        )

    # 函数执行过程
    # 2. 根据函数映射获取函数实体
    function = tool_mapping[function_name]
    # 3. 解析函数入参（将字符串转换为字典）
    # "{'arg1': xxxx}" -> {'arg1': xxxx}
    function_arguments_dict = json.loads(function_arguments)
    # 4. 执行函数
    function_result = await function(**function_arguments_dict)
    # 5. 打印函数结果
    # 注意！虽然调什么函数和输入什么参数是大模型告诉你的，但是这个工具调用的过程本身跟大模型无关！！！是在外面进行的，大模型不知道你做了什么，所以是不是应该告诉大模型一下，你这个函数到底返回了什么结果 -> 用assistant message返回函数调用情况 + 用tool message返回函数执行结果
    print(f"函数执行结果：{function_result}")

    # 将函数执行结果告诉大模型，让大模型能够根据函数执行结果得到更准确的答案
    # 6. 更新messages
    # 6.1 依据assistant_messages_template生成assistant_messages
    assistant_messages = assistant_messages_template.copy()
    assistant_messages["tool_calls"][0]["id"] = fun_id
    assistant_messages["tool_calls"][0]["function"].update({
        'arguments':
        function_arguments,
        'name':
        function_name
    })

    # 6.2 将assistant_messages添加到origin_messages中
    # 注意这里，origin_messages又添加了一个assistant message，这个assistant message是工具调用完成后的结果
    origin_messages.append(assistant_messages)

    # 6.3 将函数的输出信息添加到origin_messages中
    origin_messages.append({
        'role': 'tool',
        'content': function_result,
        'tool_call_id': fun_id
    })

    # 重要：到目前为止origin_messages包括system message、user message、assistant messages和tool message，这四个一组就构成了一次完整的function calling过程所必要的四次信息交互

    # 7.将最终的messages发送给大模型，这里直接给一个总结答案就好，还是用流式输出
    print("大模型结合工具调用结果生成答案：")
    response = await client.chat.completions.create(model="qwen-plus",
                                                    messages=origin_messages,
                                                    tools=tools, # 一定要保持一致！
                                                    tool_choice="auto",
                                                    stream=True)
    # 流式输出
    async for chunk in response:
        if chunk.choices[0].delta.content:
            response_content += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())