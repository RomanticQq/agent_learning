from langchain.chat_models import init_chat_model
from langchain.agents.middleware.types import dynamic_prompt, ModelRequest, ModelResponse
from langchain.agents import create_agent
from typing import TypedDict

model = init_chat_model("deepseek-chat")

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色来生成系统提示词"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个有用的助手。"

    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术回应。"
    elif user_role == "beginner":
        return f"{base_prompt} 简单低解释概念，避免行话。"
    
    return base_prompt

agent = create_agent(model=model, 
                    tools=[], 
                    middleware=[user_role_prompt], 
                    context_schema=Context)
  
response = agent.invoke(
    {"messages": [{"role": "user", "content": "请解释一下什么是机器学习？"}]}, 
    context={"user_role": "beginner"}
)
print(response)

