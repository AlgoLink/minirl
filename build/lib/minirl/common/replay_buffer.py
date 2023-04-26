from queue import Queue
import random

class ReplayMemory:
    """在这个示例中，我们使用Queue创建了一个先进先出队列，作为存储经验的数据结构。ReplayMemory类有以下几个方法：

    __init__(self, capacity)：初始化对象，指定经验回放缓冲区的容量。
    push(self, transition)：将一条新的经验加入到缓冲区中。如果缓冲区已满，则删除最早的经验。
    sample(self, batch_size)：从缓冲区中随机采样一批经验，并返回一个列表。
    __len__(self)：返回当前经验回放缓冲区中的经验数。
    这个类可以被用于任何需要经验回放的强化学习算法中，例如深度Q网络（DQN）等。"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = Queue(maxsize=capacity)

    def push(self, transition):
        if self.memory.full():
            self.memory.get()
        self.memory.put(transition)

    def sample(self, batch_size):
        return random.sample(
            list(self.memory.queue), min(batch_size, self.memory.qsize())
        )

    def __len__(self):
        return self.memory.qsize()