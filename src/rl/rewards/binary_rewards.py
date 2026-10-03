import torch
from typing import Callable, Any
from rl.rewards.ABC import RewardsABC

class BinaryRewards(RewardsABC):
    def __init__(self, rollout_answer_extractor: Callable[[str,], Any]):
        super().__init__(rollout_answer_extractor)

    def __call__(self, responses: list[str], answers: list) -> torch.Tensor:
        rewards = torch.tensor([self.rollout_answer_extractor(response) == answer\
                                    for response, answer in zip(responses, answers)])
        return rewards