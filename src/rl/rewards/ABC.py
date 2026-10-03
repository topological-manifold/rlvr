import torch
from typing import Callable, Any
from abc import ABC, abstractmethod

class RewardsABC(ABC):
    def __init__(self, rollout_answer_extractor: Callable[[str,], Any]):
        self.rollout_answer_extractor = rollout_answer_extractor

    @abstractmethod
    def __call__(self, responses: list[str], answers: list) -> torch.Tensor:
        if len(responses)!=len(answers):
            raise ValueError(f"responses is a list of size {len(responses)}\
                             which is not equal to size of answers label {len(answers)}.")