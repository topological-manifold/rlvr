import torch
from typing import Any
from rl.rewards.ABC import RewardsABC
from rl.rewards._drpo_grader import r1_zero_reward_fn

class R1ZeroReward(RewardsABC):
    def __init__(self):
        super().__init__()

    def __call__(self, responses: list[str], answers: list[Any]) -> torch.Tensor:
        super().__call__(responses, answers)
        rewards_dicts: list[dict[str, float]] = [r1_zero_reward_fn(response, ground_truth = answer)\
                                    for response, answer in zip(responses, answers)]
        print(f"Correct answers = {sum(rewards_dict['answer_reward'] for rewards_dict in rewards_dicts)} out of {len(rewards_dicts)}\n\
              Correct format = {sum(rewards_dict['format_reward'] for rewards_dict in rewards_dicts)} out of {len(rewards_dicts)}")
        return torch.tensor([rewards_dict['reward'] for rewards_dict in rewards_dicts])
    
    