from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


# 只能用在直接和模型对话上面

model = init_chat_model("deepseek-chat")

prompt = ChatPromptTemplate([("system", "把用户输入的中文翻译成{language}。"), ("user", "{text}")])

prompt = prompt.format(language="英文", text="今天天气怎么样？")
print(prompt)

result = model.invoke(prompt)
print(result.content)

# 使用输出解释器
str_parser = StrOutputParser()
str_result = str_parser.invoke(result)
print(str_result)
print(result.content_blocks) # [{'type': 'text', 'text': 'How is the weather today?'}]