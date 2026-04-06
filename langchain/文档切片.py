# import os
# os.environ['USER_AGENT'] = 'myagent'
from langchain_community.document_loaders import WebBaseLoader, TextLoader, Docx2txtLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
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
# loader = PyMuPDFLoader("example.pdf")
# docs = loader.load()
# print(docs)

"""文档切分"""
# 1.RecursiveCharacterTextSplitter是一个递归字符切分器，可以根据指定的分隔符将文本切分成更小的块。它会优先使用较大的分隔符进行切分，如果切分后的块仍然超过指定的大小限制，则会继续使用较小的分隔符进行切分，直到满足大小要求为止。
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,  # 每个块的最大字符数
#     chunk_overlap=20,  # 块之间的重叠字符数
#     separators=["\n\n", "\n", " ", ""],  # 切分优先级：先按段落切分，再按行切分，最后按空格切分,也可以用正则表达式切分
# )
# documents = splitter.split_documents(docs)
# for doc in documents:
#     print(doc, end="\n---\n")
# print(f"总共切分成了{len(documents)}个块。")

# 2. CharacterTextSplitter是一个简单的字符切分器，它会根据指定的分隔符将文本切分成块。与RecursiveCharacterTextSplitter不同的是，CharacterTextSplitter不会递归地使用较小的分隔符进行切分，而是直接使用指定的分隔符进行切分。
# 但是CharacterTextSplitter会合并切分后的块，如果某个块的长度超过了指定的chunk_size，那么它会将该块与下一个块合并，直到满足chunk_size的要求为止。
# text = "第一段文本。\n\n第二段文本。\n第三段文本。"
# splitter = CharacterTextSplitter(
#     chunk_size=500,  # 每个块的最大字符数
#     chunk_overlap=2,  # 块之间的重叠字符数
#     separator="\n",  # 切分符：按段落切分
# )
# chunks = splitter.split_text(text)
# print(chunks)
# print(f"总共切分成了{len(chunks)}个块。")

# 3. MarkdownHeaderTextSplitter是一个专门用于切分Markdown文本的切分器。它会根据Markdown中的标题语法（如#、##、###等）将文本切分成块。每个块对应一个标题及其下的内容。这个切分器非常适合处理结构化的Markdown文档，可以帮助你更好地组织和理解文档内容。
# markdown = """
# # 第一章
# ## 第一节
# 这是第一节的内容。
# ## 第二节
# 这是第二节的内容。
# # 第二章
# 这是第二章第一节的内容。
# """
# header = [
#     ("#", "Header1"),
#     ("##", "Header2"),
# ]
# splitter = MarkdownHeaderTextSplitter(headers_to_split_on=header)
# chunks = splitter.split_text(markdown)
# print(chunks)
# [Document(metadata={'Header1': '第一章', 'Header2': '第一节'}, page_content='这是第一节的内容。'), Document(metadata={'Header1': '第一章', 'Header2': '第二节'}, page_content='这是第二节的内容。'), Document(metadata={'Header1': '第 二章'}, page_content='这是第二章第一节的内容。')]

# 4.固定长度切分
# from langchain_text_splitters import CharacterTextSplitter
# text="12345678"
# splitter = CharacterTextSplitter(
#     chunk_size=3,  # 每个块的最大字符数
#     chunk_overlap=0,  # 块之间的重叠字符数
#     separator="",  # 切分符：按字符切分
# )
# chunks = splitter.split_text(text)
# print(chunks)

# 5.使用正则表达式切分
# import re

# def split_sentences_zh(text:str):
#     pattern = re.compile(r'[^ 。！？;]*[。！？;]+|[^ 。！？;]+$')  # 匹配中文句号、感叹号、问号和换行符
#     sentences = [m.group(0).strip() for m in pattern.finditer(text) if m.group(0).strip()]
#     return sentences
# text = "今天天气怎么样？我想知道明天的天气。谢谢！"
# sentences = split_sentences_zh(text)
# print(sentences)