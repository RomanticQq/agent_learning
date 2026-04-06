from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str = Field(..., description="姓名")
    email: str = Field(..., description="邮箱地址")
    phone: str = Field(..., description="电话号码")


# ToolStrategy：工具调用策略
# 使用人工调用工具生成结构化输出，这适用于任何支持工具调用的模型

model = init_chat_model("deepseek-chat")
agent = create_agent(
    model=model, 
    tools=[], 
    response_format=ToolStrategy(ContactInfo))

response = agent.invoke({"messages": [{"role": "user", "content": "从：jeff , 123@qq.com , 1334567890 提取联系方式"}]})
print(response["structured_response"])