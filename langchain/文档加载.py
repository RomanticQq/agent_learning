# import os
# os.environ['USER_AGENT'] = 'myagent'
from langchain_community.document_loaders import WebBaseLoader, TextLoader, Docx2txtLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
import bs4

model = init_chat_model("deepseek-chat")


"""文档加载"""

# 1.从网页中加载文档

# web_path = 'https://www.guwendao.net/default_1.aspx'

# loader = WebBaseLoader(
#     web_path=[web_path],
#     bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="contson"))
# )
# docs = loader.load()
# print(docs)

# 2. 从txt文件中加载文档

# loader = TextLoader("example.txt", encoding="utf-8")
# docs = loader.load()
# print(docs)

# 3.从docs中加载文档

# loader = Docx2txtLoader("example.docx")
# docs = loader.load()
# print(docs)


# 4.从pdf中加载文档
loader = PyMuPDFLoader("example.pdf")
docs = loader.load()
print(docs)