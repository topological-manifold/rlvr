#!/Users/towrist/root/rlvr/.venv/bin/python

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/OLMo-2-0425-1B"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32
print(f"Using {device}")

tokenizer = AutoTokenizer.from_pretrained(
  model_path, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
  model_path,
  local_files_only=True,
  dtype=dtype,
).to(device).eval()

if __name__ == "__main__":
  while True:
      prompt = input("\nPrompt (or 'exit'): ")
      if prompt.strip().lower() == "exit":
          break
    
      inputs = tokenizer(
          prompt,
          return_tensors="pt",
          return_token_type_ids=False,
      ).to(device)
    
      with torch.inference_mode():
          output = model.generate(
              **inputs,
              max_new_tokens=100,
              do_sample=False,
              pad_token_id=tokenizer.pad_token_id,
              eos_token_id=tokenizer.eos_token_id,
          )
    
      new_tokens = output[0, inputs["input_ids"].shape[1]:]
      print(tokenizer.decode(new_tokens, skip_special_tokens=True))