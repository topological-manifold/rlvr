import torch
from typing import Callable, Any
from abc import ABC, abstractmethod

class RewardsABC(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, responses: list[str], answers: list[Any]) -> torch.Tensor:
        if len(responses)!=len(answers):
            raise ValueError(f"responses is a list of size {len(responses)}\
                             which is not equal to size of answers label {len(answers)}.")