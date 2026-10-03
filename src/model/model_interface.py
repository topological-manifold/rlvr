import torch
import transformers
import torch.nn.functional as F

DEFAULT_MAX_NEW_TOKENS = 10

class ModelInterface:
    def __init__(self, local_model_path: str, device: str, model_dtype: torch.dtype):
        self.device = device
        self._model =  transformers.AutoModelForCausalLM.from_pretrained(
                                local_model_path,
                                local_files_only = True,
                                dtype = model_dtype,
                                ).to(self.device)
        
        self._tokenizer: transformers.TokenizersBackend = transformers.AutoTokenizer.from_pretrained(
                                local_model_path,
                                local_files_only = True,
                                )
        

    @property
    def model_parameters(self):
        return self._model.parameters()

    def tokenize_prompts(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        return self._tokenizer(prompts, padding = True, padding_side = 'left', return_tensors='pt').to(self.device)
    
    def decode_prompt(self, prompts: str|list[str]):
        if isinstance(prompts, str):
            prompts = [prompts]
        return self._tokenizer.batch_decode(prompts, skip_special_tokens=True)

    def generate_rollouts(self, 
                          prompts: list[str], 
                          max_rollout_tokens: int, 
                          sampling_temperature: float
                          ) -> list[str]:
        input_batch = self.tokenize_prompts(prompts)
        output_batch = self._model.generate(input_ids = input_batch['input_ids'],
                                attention_mask = input_batch['attention_mask'],
                                max_new_tokens = max_rollout_tokens,
                                do_sample = True,
                                temperature = sampling_temperature,
                                )
        rollout_batch = self._tokenizer.batch_decode(output_batch, skip_special_tokens=True)
        return rollout_batch

    def get_token_level_log_probs(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        tokenized_prompts = self.tokenize_prompts(prompts)
        tokenized_responses = self.tokenize_prompts(responses)
        
        logits: torch.Tensor = self._model(input_ids = tokenized_responses['input_ids'],
                                           attention_mask = tokenized_responses['attention_mask'])['logits'] # shape = (batch_dim, sequence_len, vocab)
        log_probs = F.log_softmax(logits[:,:-1,:], dim=2)
        # extract log probs for given response tokens
        b, l, _ = log_probs.shape
        A = torch.arange(b).unsqueeze(1).expand(-1, l)
        B = torch.arange(l).unsqueeze(0).expand(b, -1)
        C = tokenized_responses['input_ids'][:,1:]
        return log_probs[A, B, C], self._get_mask_for_log_prob(tokenized_prompts, tokenized_responses)[:,1:]
    
    def clear_gradients(self):
        self._model.zero_grad(set_to_none=True)

    def _get_mask_for_log_prob(self, tokenized_prompts: dict[str, torch.Tensor], 
                               tokenized_responses: dict[str, torch.Tensor]) -> torch.Tensor:
        n_zeros: torch.Tensor = (tokenized_responses['attention_mask']==0).sum(dim=1)+tokenized_prompts['attention_mask'].sum(dim=1)
        b, l = tokenized_responses['input_ids'].shape
        mask = n_zeros.unsqueeze(1).expand(-1, l) < torch.arange(l).unsqueeze(0).expand(b, -1).to(self.device)
        return mask

