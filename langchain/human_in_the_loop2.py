from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import human_in_the_loop
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from deepagents import create_deep_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

model = init_chat_model("deepseek-chat")

checkpointer = MemorySaver()

@tool
def delete_file(path: str) -> str:
    """从文件系统删除一个文件"""
    print("Tool: delete_file被调用了！")
    return f"文件'{path}'已被删除。"

@tool
def read_file(path: str) -> str:
    """从文件系统读取一个文件的内容"""
    print("Tool: read_file被调用了！")
    return f"文件'{path}'的内容是：这是一个示例文本。"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送一封电子邮件"""
    print("Tool: send_email被调用了！")
    return f"电子邮件已发送给{to}。"

checkpointer = MemorySaver()
agent = create_agent(
    model=model,
    tools=[delete_file, read_file, send_email],
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={
            "delete_file": True,
            "read_file": False,
            "send_email": {"allowed_decisions": ["approve", "reject"]}
        }
    )],
    checkpointer=checkpointer
)


# 处理中断
# 当触发中断时，代理会暂停执行并等待用户输入。用户可以选择批准或拒绝操作，或者提供必要的输入来继续执行。

import uuid
from langgraph.types import Command

# 创建config带thread_id保持会话状态

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "请删除文件'/tmp/important.txt'。"}]},
    config=config)

# 如果遇到需要中断，则进入if

if result.get("__interrupt__"):
    interrupts = result["__interrupt__"][0].value
    action_requests = interrupts["action_requests"]
    review_requests = interrupts["review_configs"]
    

    # 创建一个从工具名称到检查配置的查找映射
    config_map = {cfg["action_name"]: cfg for cfg in review_requests}

    for action in action_requests:
        review_config = config_map[action["name"]]
        print(f"Tool:{action['name']}")
        print(f"Arguments: {action['args']}")
        print(f"Allow decisions: {review_config['allowed_decisions']}")
    
    # 获取用户决策（每个action_request一个，按顺序）
    decisions = [
        # {"type": "approve"},  # 对delete_file批准
        {"type": "reject"},  # 对read_file拒绝
    ]

    # 将用户决策发送回代理以继续执行
    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config # 必须使用相同的配置
    )

print(result["messages"][-1].content)
