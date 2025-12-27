import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ============================
# 配置
# ============================
MODEL_NAME = "Qwen/Qwen3-0.6B" 
OUTPUT_DIR = "sft-agent-0.6b"
DATA_DIR = "my_local_agent_data"

# 1. 加载数据
dataset = load_from_disk(DATA_DIR)

# 2. 模型与分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# 3. LoRA 配置
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM", bias="none"
)

# 4. 训练参数
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,             # 1个 epoch 足够学会语法
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_length=1024,
    dataset_text_field="text",
    packing=False,
    bf16=True, # 显卡支持就开
)

# 5. 训练
trainer = SFTTrainer(
    model=MODEL_NAME,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
)

print("🚀 Starting SFT...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"✅ SFT Model saved to {OUTPUT_DIR}")