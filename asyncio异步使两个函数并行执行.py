import asyncio

async def function_a():
    """立即开始执行的异步函数"""
    print("Function A started")
    await asyncio.sleep(2)   # 模拟异步耗时操作
    print("Function A finished")

async def function_b():
    """延迟 3 秒后开始执行的异步函数"""
    print("Function B will start after 3 seconds")
    await asyncio.sleep(3)   # 等待 3 秒
    print("Function B started")
    await asyncio.sleep(1)   # 模拟耗时操作
    print("Function B finished")

async def main():
    # 同时创建两个任务（并发执行）
    task_a = asyncio.create_task(function_a())
    task_b = asyncio.create_task(function_b())
    
    # 等待两个任务完成
    await task_a
    await task_b

# 运行异步主函数
asyncio.run(function_a())
print("Both functions have finished.")