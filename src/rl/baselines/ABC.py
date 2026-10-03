import torch
from abc import ABC, abstractmethod

class BaselineABC(ABC):
    def __init__(self, *args):
        pass

    @abstractmethod
    def get_batch_baseline(self, rewards: torch.Tensor):
        pass