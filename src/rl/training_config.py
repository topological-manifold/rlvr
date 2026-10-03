import torch
from dataclasses import dataclass
from typing import Callable, Literal

@dataclass(kw_only=True)
class TrainingConfig:
    # TODO: make a lib for baselines. Implement them.
    baseline: Literal["mean", "none"] = "none"
    advantage_normalizer: Callable | None = None
    sequence_normalizer: Callable | None = None
    reward_function: Callable[[str, str], dict[str, float]]