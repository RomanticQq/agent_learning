from openai import OpenAI
import os

# 设置OpenAI API密钥（建议从环境变量读取）
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def call_llm(prompt, model="qwen3.5-plus", temperature=0.7):
    """
    调用OpenAI LLM，返回生成的文本。
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"调用LLM出错: {e}")
        return None

def reflection_agent(question, max_iterations=2):
    """
    执行Reflection Agent：
    1. 生成初始回答
    2. 反思回答的不足
    3. 根据反思改进回答
    4. 重复反思-改进直到达到最大迭代次数
    """
    # 初始提示
    initial_prompt = f"请回答以下问题：\n{question}"
    current_answer = call_llm(initial_prompt)
    if not current_answer:
        return "生成初始回答失败。"

    print("=== 初始回答 ===")
    print(current_answer)
    print()

    for i in range(max_iterations):
        # 反思提示
        reflection_prompt = f"""
你是一个反思型AI助手。请仔细分析以下回答，指出其中可能存在的不足、错误或遗漏。
你的批评应具体、有建设性，并提出改进方向。

问题：{question}
当前回答：{current_answer}

请列出你认为需要改进的地方：
"""
        reflection = call_llm(reflection_prompt, temperature=0.5)
        if not reflection:
            print("反思失败，停止迭代。")
            break

        print(f"=== 第{i+1}次反思 ===")
        print(reflection)
        print()

        # 改进提示
        improvement_prompt = f"""
基于以下反思意见，请重新回答原问题。改进后的回答应更准确、更全面。

问题：{question}
原回答：{current_answer}
反思意见：{reflection}

请给出改进后的回答：
"""
        improved_answer = call_llm(improvement_prompt, temperature=0.7)
        if not improved_answer:
            print("改进失败，停止迭代。")
            break

        print(f"=== 第{i+1}次改进后回答 ===")
        print(improved_answer)
        print()

        # 更新当前答案，继续下一轮反思（可选）
        current_answer = improved_answer

    return current_answer

if __name__ == "__main__":
    # 示例问题
    question = "解释一下什么是量子纠缠，并用一个简单的类比帮助理解。"
    final_answer = reflection_agent(question, max_iterations=2)
    print("\n=== 最终答案 ===")
    print(final_answer)