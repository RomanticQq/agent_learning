from langchain.chat_models import init_chat_model
from langchain.agents.middleware.types import dynamic_prompt, ModelRequest, ModelResponse
from langchain.agents import create_agent
from typing import TypedDict

model = init_chat_model("deepseek-chat")


agent = create_agent(model=model,
                     system_prompt="你是一个有用的助手。", 
                     tools=[])
  
response = agent.invoke(
    {"messages": [{"role": "user", "content": " 元旦是几月几号？"}]}
)
print(response)

