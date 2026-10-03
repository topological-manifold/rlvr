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
        rewards.shape = (batch_dim*n_rollouts_per_batch,)
        '''
        B = rewards.shape[0]
        response_log_probs = token_log_probabilities*response_mask
        loss_per_batch: torch.Tensor = (rewards-self.training_config.baseline(rewards))*(response_log_probs.sum(dim=1))
        return loss_per_batch.sum()/B

    def train_one_step(
        self,
        prompts: list[str],
        answers: list
    ):
        # suppose prompts.shape = (batch_size, common_prompt_len)
        repeated_prompts, repeated_answers = prompts*self.training_config.sampling_group_size, answers*self.training_config.sampling_group_size
        rollouts: list[str] = self.model_interface.generate_rollouts(repeated_prompts, 
                                                                     self.training_config.max_rollout_tokens,
                                                                     self.training_config.sampling_temperature,
                                                                     stop_strings=self.training_config.stop_strings,
                                                                     ) # (batch_size, common_prompt_len+L)
        token_log_probs, response_mask = self.model_interface.get_token_level_log_probs(repeated_prompts, rollouts) # (batch_size, L)
        rewards: torch.Tensor = self.training_config.reward_function(rollouts, repeated_answers).to(self.model_interface.device)
        loss = self.compute_loss(rewards, token_log_probs, response_mask)
        self.model_interface.clear_gradients()
        loss.backward()
        self.optimizer.step()
