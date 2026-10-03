import numpy as np
from typing import Any
from abc import ABC, abstractmethod

class DatasetInterfaceABC:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed=seed)

    @abstractmethod
    def load_dataset(self, dataset_path: str):
        pass
    
    @abstractmethod
    def get_train(self) -> dict[str, list]:
        '''
        should return a dict with keys 'prompts' and 'answers', 
        should contain all examples in the training set
        '''
        pass

    @abstractmethod
    def get_validation(self) -> dict[str, list]:
        '''
        should return a dict with keys 'prompts' and 'answers', 
        should contain all examples in the training set
        '''
        pass

    @abstractmethod
    def get_test(self) -> dict[str, list]:
        '''
        should return a dict with keys 'prompts' and 'answers', 
        should contain all examples in the training set
        '''
        pass

    @abstractmethod
    def get_train_batch(self, batch_size) -> dict[str, list]:
        '''
        should return a dict with keys 'prompts' and 'answers', 
        each of which is a list of len batch_size
        '''
        pass