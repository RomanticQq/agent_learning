from rank_bm25 import BM25Okapi

# 1. 准备语料库（通常需要先分词）
corpus = [
    "Hello there good man",
    "It is quite windy in London",
    "How is the weather today"
]
# 将文档转换为词列表（Tokenization）
tokenized_corpus = [doc.split(" ") for doc in corpus]

# 2. 初始化 BM25 实例
bm25 = BM25Okapi(tokenized_corpus)

# 3. 准备查询语句（同样需要分词）
query = "windy London"
tokenized_query = query.split(" ")

# 4. 获取得分
doc_scores = bm25.get_scores(tokenized_query)
print(f"所有文档得分: {doc_scores}")

# 5. 直接获取排名前 n 的文档
top_n = bm25.get_top_n(tokenized_query, corpus, n=1)
print(f"最相关的文档: {top_n}")