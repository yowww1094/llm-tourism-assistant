from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct", torch_dtype=torch.float16)
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"])
model = get_peft_model(model, lora_config)
trainer = SFTTrainer(model=model, train_dataset=dataset, ...)
trainer.train()