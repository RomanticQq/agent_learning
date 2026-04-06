"""
多智能体示例：主管（Supervisor）根据用户输入，路由到不同的专业智能体。

依赖安装：
pip install langchain langchain-openai langgraph

环境变量：
需要设置 OPENAI_API_KEY（也可替换为其他 LLM）
"""

import os
from typing import Literal, TypedDict, Annotated, List
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

# ------------------------------
# 1. 定义工具
# ------------------------------
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气（模拟数据）"""
    # 实际使用时可替换为真实 API 调用
    weather_db = {
        "北京": "晴，24°C，湿度45%",
        "上海": "多云，28°C，湿度70%",
        "深圳": "雷阵雨，30°C，湿度85%",
    }
    return weather_db.get(city, f"未找到{city}的天气数据，请尝试北京、上海或深圳。")

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，例如 '2 + 2' 或 '10 * 3'"""
    try:
        # 安全计算（仅支持基本运算）
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"

# ------------------------------
# 2. 初始化 LLM
# ------------------------------
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 替换为您的模型

llm = ChatOpenAI(
    model="qwen3.5-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
# ------------------------------
# 3. 创建专业智能体（每个智能体拥有自己的工具和提示）
# ------------------------------
# 天气智能体
weather_agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt="你是一个天气查询助手。用户会询问某城市的天气，请调用 get_weather 工具获取信息，并给出自然语言回答。"
)

# 计算智能体
calculator_agent = create_react_agent(
    model=llm,
    tools=[calculate],
    prompt="你是一个数学计算助手。用户会给出数学表达式，请调用 calculate 工具计算结果，并直接返回答案。"
)

# ------------------------------
# 4. 定义状态（工作流中传递的数据）
# ------------------------------
class MultiAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]   # 对话历史，operator.add 用于合并
    next_step: str                                          # 下一步要调用的智能体

# ------------------------------
# 5. 主管节点（Supervisor）：决定调用哪个子智能体或结束
# ------------------------------
def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    messages = state["messages"]
    # 构造一个用于决策的提示
    # system_prompt = (
    #     "你是一个主管，负责分析用户的最新请求，并决定将任务交给哪个专业智能体处理。\n"
    #     "可选的智能体：\n"
    #     "- 'weather': 处理天气查询类问题\n"
    #     "- 'calculator': 处理数学计算类问题\n"
    #     "- 'FINISH': 如果已经得到了最终答案，或者用户的问题不需要进一步处理，输出 'FINISH'\n"
    #     "请只输出智能体的名称（weather / calculator / FINISH），不要输出其他内容。"
    # )
    system_prompt = (
    "你是一个主管。首先检查对话历史中的最后一条消息：\n"
    "- 如果最后一条消息是 AI 消息，并且已经回答了用户的最新问题，则只输出 'FINISH'。\n"
    "- 否则，根据用户的最新请求决定交给哪个专业智能体：\n"
    "   * 'weather': 处理天气查询\n"
    "   * 'calculator': 处理数学计算\n"
    "- 如果用户的问题不涉及以上两类，也输出 'FINISH'。\n"
    "只输出一个单词（weather / calculator / FINISH），不要输出其他内容。"
)
    # 将最新的人类消息作为决策依据
    decision_prompt = [{"role": "system", "content": system_prompt}] + [
        {"role": m.type, "content": m.content} for m in messages if isinstance(m, (HumanMessage, AIMessage))
    ]
    decision = llm.invoke(decision_prompt).content.strip().lower()
    if decision not in ["weather", "calculator"]:
        decision = "finish"   # 安全降级
    return {"next_step": decision}

# ------------------------------
# 6. 子智能体节点（包装 create_react_agent 的执行）
# ------------------------------
def weather_node(state: MultiAgentState) -> MultiAgentState:
    """调用天气智能体处理最新消息"""
    messages = state["messages"]
    # 只将最新的用户消息传递给智能体（避免重复历史过长）
    latest_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not latest_human:
        return {"messages": [AIMessage(content="没有收到有效的查询。")]}
    # 调用天气智能体（它是 Runnable，输入是消息列表）
    response = weather_agent.invoke({"messages": [latest_human]})
    # 提取智能体的最终回复（通常最后一条 AIMessage）
    ai_msg = response["messages"][-1]
    return {"messages": [ai_msg]}

def calculator_node(state: MultiAgentState) -> MultiAgentState:
    """调用计算智能体处理最新消息"""
    messages = state["messages"]
    latest_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not latest_human:
        return {"messages": [AIMessage(content="没有收到有效的计算请求。")]}
    response = calculator_agent.invoke({"messages": [latest_human]})
    ai_msg = response["messages"][-1]
    return {"messages": [ai_msg]}

# ------------------------------
# 7. 构建工作流图
# ------------------------------
workflow = StateGraph(MultiAgentState)

# 添加节点
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("weather", weather_node)
workflow.add_node("calculator", calculator_node)

# 设置入口
workflow.set_entry_point("supervisor")

# 添加条件边：从主管出发，根据 next_step 路由到不同智能体或结束
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_step"],   # 取 next_step 字段的值
    {
        "weather": "weather",
        "calculator": "calculator",
        "finish": END,
    }
)

# 子智能体执行完毕后，必须返回到主管，以便决定下一步（是否继续或结束）
workflow.add_edge("weather", "supervisor")
workflow.add_edge("calculator", "supervisor")

# 编译图
app = workflow.compile()

# ------------------------------
# 8. 运行示例
# ------------------------------
if __name__ == "__main__":
    # 示例 1：天气查询
    # print("=== 用户：北京今天天气怎么样？ ===")
    # inputs = {"messages": [HumanMessage(content="北京今天天气怎么样？")]}
    # for output in app.stream(inputs):
    #     # 流式输出各节点的结果
    #     for node_name, node_output in output.items():
    #         print(f"节点 '{node_name}' 输出:")
    #         if "messages" in node_output:
    #             for msg in node_output["messages"]:
    #                 print(f"  {msg.type}: {msg.content}")
    #     print("-" * 50)

    # 示例 2：数学计算
    print("\n=== 用户：计算 25 * 4 + 10 等于多少？ ===")
    inputs = {"messages": [HumanMessage(content="计算 25 * 4 + 10 等于多少？")]}
    print(app.invoke(inputs))   # 直接获取最终状态结果
    # for output in app.stream(inputs):
    #     for node_name, node_output in output.items():
    #         print(f"节点 '{node_name}' 输出:")
    #         if "messages" in node_output:
    #             for msg in node_output["messages"]:
    #                 print(f"  {msg.type}: {msg.content}")
    #     print("-" * 50)

    # 示例 3：通用对话（主管会直接结束）
    # print("\n=== 用户：你好，请介绍一下你自己 ===")
    # inputs = {"messages": [HumanMessage(content="你好，请介绍一下你自己")]}
    # for output in app.stream(inputs):
    #     for node_name, node_output in output.items():
    #         print(f"节点 '{node_name}' 输出:")
    #         if "messages" in node_output:
    #             for msg in node_output["messages"]:
    #                 print(f"  {msg.type}: {msg.content}")
    #     print("-" * 50)