from langchain.messages import HumanMessage, SystemMessage,AIMessage
from langchain.chat_models import init_chat_model


model = init_chat_model("deepseek-chat")
messages = [SystemMessage(content="你是我的人工智能助手。"), HumanMessage(content="元旦节是几月几号？", name="user1", id="1")]
response = model.invoke(messages)
print(response)