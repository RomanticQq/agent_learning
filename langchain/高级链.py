from langchain_core.runnables import RunnableLambda, chain
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

template = ChatPromptTemplate.from_template("{a} + {b}是多少？")

model = init_chat_model("deepseek-chat")
def length(t):
    return len(t)

def mul(t1, t2):
    return len(t1) * len(t2)


# 把函数转换成组件有两种方式，一种是直接用RunnableLambda包装一下，另一种是用@chain装饰器修饰一下
@chain
def mul_length(d):
    return mul(d["t1"], d["t2"])

chain1 = template | model
chain2 = (
    {
        "a": itemgetter("name") | RunnableLambda(length),
        "b": {"t1": itemgetter("name"), "t2": itemgetter("sex") } | mul_length,
    }
    | chain1
    | StrOutputParser()
)
print(chain2.invoke({"name": "Alice", "sex": "female"}))