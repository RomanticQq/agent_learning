import jieba
from rank_bm25 import BM25Okapi
import time


start_time = time.time()
corpus = [
    "自然语言处理是人工智能的核心领域",
    "北京今天的天气非常不错",
    "BM25算法在信息检索中表现优异"
]

corpus = corpus * 10000
print(corpus)
print(len(corpus))
# 中文分词
tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
start_time = time.time()
query = "人工智能检索"
tokenized_query = list(jieba.cut(query))

# 获取相关度最高的文档
result = bm25.get_top_n(tokenized_query, corpus, n=1)
print(result)
end_time = time.time()
print(f"查询耗时: {end_time - start_time:.4f} 秒")