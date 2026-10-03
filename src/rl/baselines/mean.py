import torch
from rl.baselines.ABC import BaselineABC

class MeanBaseline(BaselineABC):
    def __init__(self):
        super().__init__()
    
    def get_batch_baseline(self, rewards: torch.Tensor):
        # rewards.shape = (BATCH_DIM,)
        if rewards.dim > 1:
            raise ValueError(f"Expected 1-dimensional rewards tensor, but it has dimension {rewards.dim}.")
        return rewards.mean()