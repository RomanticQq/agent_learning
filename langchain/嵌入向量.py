from langchain_community.embeddings import DashScopeEmbeddings
import os
from langchain.chat_models import init_chat_model


model = init_chat_model("deepseek-chat")

embeddings = DashScopeEmbeddings(
    model = "text-embedding-v1",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

text = "这是一个测试文本"
embedding = embeddings.embed_query(text)
print(f"嵌入向量：{embedding}")
print(f"单个文本嵌入维度：{len(embedding)}")
