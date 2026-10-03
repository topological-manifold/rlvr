import torch
from typing import Any
from typing import Callable
from dataclasses import dataclass
from rl.baselines import BaselineABC

@dataclass(kw_only=True)
class TrainingConfig:
    baseline: BaselineABC
    advantage_normalizer: Callable | None = None
    sequence_normalizer: Callable | None = None
    answer_extractor: Callable[[str,], Any]
    reward_function: Callable[[str, Any], torch.Tensor]