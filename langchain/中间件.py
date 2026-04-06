from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

model = init_chat_model("deepseek-chat")
# agent_v1 = create_agent(model=model, tools=[], middleware=[SummaryMiddleware(), HumanMessageMiddleware()])

# 你可以根据需要调整SummaryMiddleware的触发条件，例如基于消息数量、总token数等
# 当接近对话次数上限时，自动汇总对话历史记录，以节省上下文空间并保持对话的连贯性。
agent = create_agent(
    model=model,
    tools=[], 
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=('tokens', 4000),
            keep=("messages", 20),
            # summary_format="对话历史记录总结：{summary}" # 可以自定义进行摘要的提示词....   # 可选
        ),
    ])
result = agent.invoke({"messages": [{"role": "user", "content": "请解释一下什么是机器学习？"}]})
print(result)
