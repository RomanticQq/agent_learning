import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
import os

embeddings = DashScopeEmbeddings(
    model = "text-embedding-v1",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 评分方式
score_measures = [
    "default", # default = "l2"
    "cosine", # 余弦相似度， 1-cos   现在的值的范围变成了[0,2],越接近于0越好
    "l2",
    "ip" # 点积
]

# 创建向量库

db = Chroma(
    collection_name="collection_name",
    embedding_function=embeddings,
    persist_directory="./chroma_db1",
    collection_metadata={"hnsw:space": 'l2'}
)

documents = [
    Document(page_content="这个苹果手机很好用"),
    Document(page_content="我国山东地区盛产苹果")
]

# 添加文档
ids = db.add_documents(documents)
print(ids)
print('*'*20)

# 检索
results = db.similarity_search_with_score("我想买个手机")
for doc, score in results:
    print(doc.page_content,end='\t')
    print(f"score: {score}")