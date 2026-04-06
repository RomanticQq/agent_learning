import requests
import json
import os

def rerank_text():
    url = 'https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank'
    
    # 建议通过环境变量获取 API Key，或者直接替换为字符串
    api_key = os.getenv("DASHSCOPE_API_KEY")
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "qwen3-vl-rerank",
        "input": {
            "query": "什么是文本排序模型",
            "documents": [
                "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序",
                "量子计算是计算科学的一个前沿领域",
                "预训练语言模型的发展给文本排序模型带来了新的进展"
            ]
        },
        "parameters": {
            "return_documents": True,
            "top_n": 5
        }
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        # 检查请求是否成功
        response.raise_for_status() 
        
        # 打印返回的 JSON 结果
        print(json.dumps(response.json(), indent=4, ensure_ascii=False))
        
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {e}")

if __name__ == "__main__":
    rerank_text()