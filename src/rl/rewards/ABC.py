import torch
from abc import ABC, abstractmethod

class RewardsABC(ABC):
    def __init__(self, *args):
        pass

    @abstractmethod
    def __call__(self, responses: list[str], answers: list) -> torch.Tensor:
        pass