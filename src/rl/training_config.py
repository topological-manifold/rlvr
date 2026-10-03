import torch
from typing import Any
from typing import Callable
from dataclasses import dataclass
from rl.baselines import BaselineABC, NoneBaseline
from rl.rewards import RewardsABC

@dataclass(kw_only=True)
class TrainingConfig:
    baseline: BaselineABC = NoneBaseline()
    advantage_normalizer: Callable | None = None
    sequence_normalizer: Callable | None = None
    sampling_temperature: float
    sampling_group_size: int
    max_rollout_tokens: int
    reward_function: RewardsABC
    stop_strings: list[str]
