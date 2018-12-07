import random
from collections import namedtuple, deque
import torch.utils.data as data_utils
import torch

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))



def demo(func):
    def push_wrapper(*args, demo = True):
        data_point = func(*args)
        if demo == True:
            x = data_point.state
            y = data_point.action
            train = data_utils.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return push_wrapper


class Memory(object):
    def __init__(self, buffer_size):
        #self.memory = []
        self.memory = deque(maxlen=buffer_size)  

    @demo
    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))
        return self.memory[-1]

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)
    
