from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain.agents.middleware.types import ModelRequest,ModelResponse
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware.types import wrap_model_call
import os


qwen = ChatOpenAI(model="qwen3.5-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
ds = ChatDeepSeek(model="deepseek-chat")

basic = qwen
advanced = ds

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据会话复杂度选择模型"""
    message_count = len(request.state["messages"])
    if message_count < 5:
        model = basic
    else:
        model = advanced
    return handler(request.override(model=model))

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": 1}}

agent = create_agent(model=basic, tools=[],middleware=[dynamic_model_selection], checkpointer=checkpointer)

for _ in range(5):
    response = agent.invoke({"messages": [{"role": "user", "content": "元旦节是几月几号？"}]}, config=config)
    print(response)

