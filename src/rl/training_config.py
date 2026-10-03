import torch
from dataclasses import dataclass
from rl.baselines import BaselineABC
from typing import Callable

@dataclass(kw_only=True)
class TrainingConfig:
    # TODO: make a lib for baselines. Implement them.
    baseline: BaselineABC
    advantage_normalizer: Callable | None = None
    sequence_normalizer: Callable | None = None
    reward_function: Callable[[str, str], dict[str, float]]