import torch
from rl.baselines.ABC import BaselineABC

class NoneBaseline(BaselineABC):
    def __init__(self):
        super().__init__()
    
    def __call__(self, rewards: torch.Tensor):
        return 0