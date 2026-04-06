import os
import logging
import operator
from typing import Annotated, List, Tuple, TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================
# 1. 状态定义
# ==========================
class PlanExecuteState(TypedDict):
    input: str                     # 用户输入
    plan: List[str]                # 剩余待执行的步骤
    past_steps: Annotated[List[Tuple[str, str]], operator.add]  # 已执行步骤及结果
    response: str                  # 最终回答

# ==========================
# 2. 模型配置（建议从环境变量读取）
# ==========================
DEFAULT_MODEL = "qwen3.5-plus"
DEFAULT_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 创建模型实例
llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    api_key=DEFAULT_API_KEY,
    base_url=DEFAULT_BASE_URL,
    temperature=0.7
)

# ==========================
# 3. 计划生成节点
# ==========================
PLANNER_PROMPT = ChatPromptTemplate.from_template(
    "针对以下目标，制定一个分步骤的计划：\n{input}\n"
    "计划应由具体的步骤组成，每个步骤一行，不要包含编号。"
)

def planner_node(state: PlanExecuteState) -> dict:
    """生成执行计划"""
    prompt = PLANNER_PROMPT.format(input=state["input"])
    response = llm.invoke(prompt)
    # 清洗步骤：去除空行和可能存在的序号（如“1. ”）
    steps = [line.strip() for line in response.content.split("\n") if line.strip()]
    # 去除可能的前缀序号（如 "1. 写开头" -> "写开头"）
    cleaned_steps = []
    for step in steps:
        if step[0].isdigit() and (len(step) > 1 and step[1] in '.、'):
            step = step[2:].strip()
        cleaned_steps.append(step)
    logger.info(f"生成计划: {cleaned_steps}")
    return {"plan": cleaned_steps}

# ==========================
# 4. 步骤执行节点
# ==========================
def executor_node(state: PlanExecuteState) -> dict:
    """执行当前计划的第一步"""
    if not state["plan"]:
        # 如果没有计划，直接返回空操作（实际上不应发生）
        return {}

    current_step = state["plan"][0]
    # 构建上下文：包含完整计划和历史步骤
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(state["plan"]))
    history = ""
    if state["past_steps"]:
        history = "已完成的步骤及结果：\n" + "\n".join(
            f"步骤: {step}\n结果: {result}" for step, result in state["past_steps"]
        ) + "\n"
    prompt = f"{history}当前剩余计划：\n{plan_str}\n\n现在请执行步骤：{current_step}"
    logger.info(f"执行步骤: {current_step}")
    response = llm.invoke(prompt)
    result = response.content
    logger.info(f"步骤结果: {result[:100]}...")  # 仅记录前100字符

    return {
        "past_steps": [(current_step, result)],
        "plan": state["plan"][1:]  # 移除已执行的步骤
    }

# ==========================
# 5. 重规划节点（可选）
# ==========================
REPLAN_PROMPT = ChatPromptTemplate.from_template(
    "原始目标：{original_input}\n"
    "原定计划：\n{original_plan}\n"
    "已完成步骤及结果：\n{history}\n"
    "剩余计划：\n{remaining_plan}\n"
    "请根据已执行的结果，判断是否需要调整剩余计划。"
    "如果不需要调整，直接输出「不需要调整」；如果需要调整，请输出新的剩余计划，每行一个步骤。"
)

def replan_node(state: PlanExecuteState) -> dict:
    """根据历史执行结果调整剩余计划"""
    # 如果已经没有剩余计划，无需重规划
    if not state["plan"]:
        return {}

    # 构建原始计划（从过去步骤反推，但这里简化：我们不知道原始计划，可以从状态中获取？）
    # 由于状态中没有保存原始计划，这里简单处理：如果历史步骤为空，则不需要调整。
    if not state["past_steps"]:
        return {}

    # 构建历史信息
    history_str = "\n".join(f"步骤: {step}\n结果: {result}" for step, result in state["past_steps"])
    original_plan = "\n".join(f"{i+1}. {step}" for i, step in enumerate(state["plan"] + [step for step, _ in state["past_steps"]]))
    # 但更好的方式是保存原始计划，这里为了示例，仅用当前剩余计划 + 历史步骤推导
    # 实际上原始计划应该是 state["plan"] 初始值，但已丢失，因此我们简单调用模型判断
    prompt = REPLAN_PROMPT.format(
        original_input=state["input"],
        original_plan=original_plan,
        history=history_str,
        remaining_plan="\n".join(f"{i+1}. {step}" for i, step in enumerate(state["plan"]))
    )
    response = llm.invoke(prompt)
    adjustment = response.content.strip()
    if adjustment == "不需要调整" or not adjustment:
        return {}
    # 解析新的计划（假设每行一个步骤）
    new_plan = [line.strip() for line in adjustment.split("\n") if line.strip()]
    logger.info(f"重规划后新计划: {new_plan}")
    return {"plan": new_plan}

# ==========================
# 6. 最终响应节点
# ==========================
RESPONDER_PROMPT = ChatPromptTemplate.from_template(
    "原始目标：{original_input}\n"
    "执行过程：\n{history}\n"
    "请根据以上执行结果，给出最终的回答。"
)

def responder_node(state: PlanExecuteState) -> dict:
    """生成最终回答"""
    # 构建历史记录
    history_str = "\n".join(f"步骤: {step}\n结果: {result}" for step, result in state["past_steps"])
    prompt = RESPONDER_PROMPT.format(original_input=state["input"], history=history_str)
    response = llm.invoke(prompt)
    final_answer = response.content
    logger.info("生成最终回答")
    return {"response": final_answer}

# ==========================
# 7. 条件路由函数
# ==========================
def should_continue(state: PlanExecuteState) -> str:
    """决定下一步是继续执行、重规划还是结束"""
    # 如果已经没有剩余计划，进入最终响应节点
    if not state["plan"]:
        return "respond"
    # 如果有历史步骤，可以考虑是否需要重规划（这里简单判断是否执行了至少3步且还有计划）
    # 实际可根据业务逻辑调整，本例简单返回继续执行
    # 但为了演示重规划，如果执行了超过2步且还有计划，则进行重规划
    if len(state["past_steps"]) > 2 and state["plan"]:
        return "replan"
    return "execute"

# ==========================
# 8. 构建工作流图
# ==========================
def build_workflow() -> StateGraph:
    """构建并编译工作流图"""
    workflow = StateGraph(PlanExecuteState)

    # 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("replan", replan_node)
    workflow.add_node("responder", responder_node)

    # 设置入口
    workflow.set_entry_point("planner")

    # 规划后进入执行
    workflow.add_edge("planner", "executor")

    # 执行后的条件分支
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "execute": "executor",   # 继续执行下一个步骤
            "replan": "replan",      # 进入重规划
            "respond": "responder",  # 生成最终答案
        }
    )

    # 重规划后重新执行（重规划后的计划从剩余步骤开始）
    workflow.add_edge("replan", "executor")

    # 生成最终答案后结束
    workflow.add_edge("responder", END)

    return workflow.compile()

# ==========================
# 9. 主程序
# ==========================
if __name__ == "__main__":
    # 初始输入
    inputs = {"input": "帮我写一个100字的童话故事"}

    # 编译工作流
    app = build_workflow()

    # 运行（使用线程ID可选）
    config = {"configurable": {"thread_id": "1"}}
    try:
        final_state = app.invoke(inputs, config)
        print("\n--- 最终答案 ---")
        print(final_state.get("response", "未生成答案"))
    except Exception as e:
        logger.exception("工作流执行失败")
        print(f"错误: {e}")