import operator
from typing import Annotated, List, Tuple, TypedDict, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END



# --- 3. 定义执行者 (Executor) ---
# 这里我们假设有一个简单的搜索工具或模型直接回答
# executor_model = ChatOpenAI(model="gpt-4o")
# executor_model = ChatOpenAI(model="deepseek-chat")
executor_model = ChatOpenAI(
    model="qwen3.5-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


    
prompt = f"你是谁"
response = executor_model.invoke(prompt)
print(response.content)



