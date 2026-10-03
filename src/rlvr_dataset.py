import numpy as np

class RLVRDataset:
    def __init__(self, seed, prompts: list[str], answers: list):
        self.prompts = prompts
        self.answers = answers
        self.rng = np.random.default_rng(seed)
        self._validate_prompts_and_answers(prompts, answers)
    
    @property
    def n_examples(self):
        return len(self.prompts)

    @staticmethod
    def _validate_prompts_and_answers(prompts, answers):
        if not isinstance(prompts, list):
            raise TypeError(f"prompts should be a list, not {type(prompts)}")
        if not isinstance(answers, list):
            raise TypeError(f"answers should be a list, not {type(answers)}")
        if len(prompts)!=len(answers):
            raise RuntimeError(f"size of prompts ({len(prompts)})\
                               does not match size of answers ({len(answers)}).")
        for prompt in prompts:
            if not isinstance(prompt, str):
                raise TypeError(f"Each prompt should be a string. One prompt of type {type(prompt)} found.")


    def get_batch(self, batch_size: int) -> dict[str, list]:
        indices = self.rng.choice(self.n_examples, batch_size)
        return {
                "prompts": [self.prompts[idx] for idx in indices],
                "answers": [self.answers[idx] for idx in indices]
                }
    
    @staticmethod
    def perform_train_valid_split(n_training_examples: int, validation_fraction: float=0.2, split_seed: int=0)\
        -> tuple[list[int], list[int]]:
        validation_indices = np.random.default_rng(split_seed).choice(n_training_examples, size = int(n_training_examples*validation_fraction))
        train_indices = np.setdiff1d(np.arange(n_training_examples), validation_indices)
        return train_indices, validation_indices