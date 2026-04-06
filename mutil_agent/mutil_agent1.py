import threading
import queue
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Message:
    """消息结构，用于智能体之间的通信"""
    sender: str      # 发送者 ID
    receiver: str    # 接收者 ID
    content: Any     # 消息内容
    msg_type: str = "data"  # 消息类型，可用于路由


class Agent(threading.Thread):
    """所有智能体的基类，支持异步消息传递"""

    def __init__(self, agent_id: str, stop_event: threading.Event):
        super().__init__(name=f"Agent-{agent_id}")
        self.agent_id = agent_id
        self.mailbox = queue.Queue()          # 消息队列（邮箱）
        self.stop_event = stop_event          # 全局停止信号
        self.running = True

    def send(self, receiver: 'Agent', message: Message):
        """向另一个智能体发送消息"""
        receiver.mailbox.put(message)

    def run(self):
        """线程主循环：持续从邮箱取消息并处理"""
        print(f"[{self.agent_id}] 启动")
        while not self.stop_event.is_set():
            try:
                # 等待消息，超时 0.5 秒以便及时响应停止信号
                msg = self.mailbox.get(timeout=0.5)
                self.handle_message(msg)
            except queue.Empty:
                continue
        print(f"[{self.agent_id}] 停止")

    def handle_message(self, msg: Message):
        """子类必须实现此方法来处理收到的消息"""
        raise NotImplementedError


class CounterAgent(Agent):
    """计数器智能体：统计收到的数字，达到阈值后通知 PrinterAgent"""

    def __init__(self, agent_id: str, stop_event: threading.Event, threshold: int = 3):
        super().__init__(agent_id, stop_event)
        self.threshold = threshold
        self.count = 0
        self.printer_agent: Optional[PrinterAgent] = None  # 后续绑定

    def set_printer(self, printer: 'PrinterAgent'):
        """绑定目标打印机智能体"""
        self.printer_agent = printer

    def handle_message(self, msg: Message):
        if msg.msg_type == "number":
            self.count += msg.content
            print(f"[{self.agent_id}] 当前计数 = {self.count}")

            if self.count >= self.threshold and self.printer_agent:
                # 达到阈值，发送通知消息给 PrinterAgent
                notify_msg = Message(
                    sender=self.agent_id,
                    receiver=self.printer_agent.agent_id,
                    content=f"计数器已达阈值 {self.threshold}，总计数 = {self.count}",
                    msg_type="alert"
                )
                self.send(self.printer_agent, notify_msg)
                # 可选：停止自己，或继续工作
                # self.stop_event.set()   # 如需停止整个系统可打开
        else:
            print(f"[{self.agent_id}] 收到未知类型消息: {msg}")


class PrinterAgent(Agent):
    """打印机智能体：收到消息后打印内容"""

    def handle_message(self, msg: Message):
        if msg.msg_type == "alert":
            print(f"[{self.agent_id}] 🖨️ 打印: {msg.content}")
        else:
            print(f"[{self.agent_id}] 普通消息: {msg.content}")


def main():
    # 创建全局停止事件
    stop_event = threading.Event()

    # 创建两个智能体
    counter = CounterAgent("Counter1", stop_event, threshold=5)
    printer = PrinterAgent("Printer1", stop_event)

    # 建立通信关系（双向发送均可行）
    counter.set_printer(printer)

    # 启动智能体线程
    counter.start()
    printer.start()

    # 主线程向计数器发送一系列数字
    numbers_to_send = [1, 2, 1, 3, 2]   # 总和 9 > 5
    for num in numbers_to_send:
        msg = Message(
            sender="Main",
            receiver=counter.agent_id,
            content=num,
            msg_type="number"
        )
        counter.send(counter, msg)   # 发送给自己（也可由外部直接放入 mailbox）
        time.sleep(1)   # 模拟间隔

    # 等待所有消息处理完毕（简单等待，实际可根据业务需要）
    time.sleep(2)

    # 停止所有智能体
    print("\n主程序发出停止信号...")
    stop_event.set()

    # 等待线程结束
    counter.join(timeout=2)
    printer.join(timeout=2)
    print("所有智能体已停止，程序退出。")


if __name__ == "__main__":
    main()