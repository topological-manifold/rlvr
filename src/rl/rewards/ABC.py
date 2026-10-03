import torch
from typing import Callable, Any
from abc import ABC, abstractmethod

class RewardsABC(ABC):
    def __init__(self, answer_extractor: Callable[[str,], Any]):
        self.answer_extractor = answer_extractor

    @abstractmethod
    def __call__(self, responses: list[str], answers: list) -> torch.Tensor:
        pass