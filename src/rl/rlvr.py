import torch
from rl.training_config import TrainingConfig
from model.model_interface import ModelInterface

class RLVR:
    def __init__(self, training_config: TrainingConfig, model_interface: ModelInterface,):
        self.training_config: TrainingConfig = training_config
        self.model_interface: ModelInterface = model_interface
        self.optimizer: torch.optim.Adam = torch.optim.Adam(self.model_interface.model_parameters)

    def compute_loss(self, rewards: torch.Tensor, token_log_probabilities: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        '''
        rewards.shape = (batch_dim, n_rollouts_per_batch)
        '''
        B, G = rewards.shape
        response_log_probs = token_log_probabilities*response_mask
        loss_per_batch: torch.Tensor = (rewards-self.training_config.baseline.get_batch_baseline(rewards))*(response_log_probs.sum(dim=2))
        return loss_per_batch.sum()/(B*G)

    def train_one_step(
        self,
        prompts: list[str],
    ):
        # suppose prompts.shape = (batch_size, common_prompt_len)
        rollouts: list[str] = self.model_interface.generate_rollouts(prompts) # (batch_size, common_prompt_len+L)
        token_log_probs, response_mask  = self.model_interface.get_token_level_log_probs(prompts, rollouts) # (batch_size, L)
        rewards: torch.Tensor = self.training_config.reward_function(rollouts, token_log_probs)
        loss = self.compute_loss(rewards, token_log_probs, response_mask)
        self.model_interface.clear_gradients()
        loss.backward()
        self.optimizer.step()
