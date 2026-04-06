from baidusearch.baidusearch import search
import json

query = "北京今天的天气如何？"
num_results = 3
results = search(query, num_results=num_results)
# 转换为json
results = json.dumps(results, ensure_ascii=False)
print(results)
a = json.dumps(results)
print(type(a))