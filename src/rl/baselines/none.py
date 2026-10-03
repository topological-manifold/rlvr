import torch
from rl.baselines.ABC import BaselineABC

class NoneBaseline(BaselineABC):
    def __init__(self):
        super().__init__()
    
    def get_batch_baseline(self, rewards: torch.Tensor):
        return 0