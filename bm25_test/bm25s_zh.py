import bm25s
import time


start_time = time.time()
corpus = [
    "自然语言处理是人工智能的核心领域",
    "北京今天的天气非常不错",
    "BM25算法在信息检索中表现优异"
]

corpus = corpus * 100
corpus_tokens = bm25s.tokenize(corpus, stopwords="zh")
retriever = bm25s.BM25()
retriever.index(corpus_tokens)
# start_time = time.time()
# Query the corpus
query = "人工智能检索"
query_tokens = bm25s.tokenize(query, stopwords="zh")
results, scores = retriever.retrieve(query_tokens, k=2)
end_time = time.time()
print(f"查询耗时: {end_time - start_time:.4f} 秒")


# 在创建300个文档，然后对其进行查询，查询耗时约为100ms;