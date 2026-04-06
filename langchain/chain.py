from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

prompt = ChatPromptTemplate([("system", "把用户输入的中文翻译成{language}。"), ("user", "{text}")])

parser = StrOutputParser()

model = init_chat_model("deepseek-chat")

_chain = prompt | model | parser
result = _chain.invoke({"language": "英文", "text": "我在吃饭"})
print(result)


# prompt / message ->llm -> output parser


# 必粗成为langchain的组件，才能使用langchain的管道符连接