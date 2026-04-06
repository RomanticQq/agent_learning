from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyMuPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model
import os
import shutil


model = init_chat_model("deepseek-chat")

# 加载文档

loader = PyMuPDFLoader("example.pdf")
docs = loader.load()

# 文档切分

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=['\n\n','\n','']
)

documents = text_splitter.split_documents(docs)

# 文档嵌入模型
embeddings = DashScopeEmbeddings(
    model = "text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

if os.path.exists("./demo_test"):
    shutil.rmtree("./demo_test")
# 向量数据库
db = Chroma.from_documents(
    collection_name="demo",
    documents=documents,
    embedding=embeddings,
    persist_directory="./demo_test"
)

# 查询检索
# results1 = db.similarity_search("付强做过AI画室这个项目吗？", k=1)
# print(results1)

# results2 = db.similarity_search_with_score("付强做过AI画室这个项目吗？", k=1)
# print(results2)

# 检索
docs_find = RunnableLambda(db.similarity_search).bind(k=2)
print(docs_find)
# results = docs_find.invoke("付强做过AI画室这个项目吗？")

message = """
仅使用提供的上下文回答下面的问题：
{question}
上下文：
{context}
"""

prompt_template = ChatPromptTemplate([('human', message)])

chain = {"question": RunnablePassthrough(),
         "context": docs_find} | prompt_template | model

# 用大模型生成回答
response = chain.invoke("根据付强的简历，判断付强做过AI画室这个项目吗？")
print(response)