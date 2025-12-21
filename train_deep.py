# =========================
# 3️⃣ SABİTLER
# =========================
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_NAME = "Naholav/CodeGen-Deep-5K"
NEW_MODEL_NAME = "Qwen2.5-Coder-1.5B-Instruct-DEEP-LoRA"

# =========================
# 4️⃣ DATASET
# =========================
dataset = load_dataset(DATASET_NAME, split="train")

system_prompt = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."

def format_data(sample):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["input"]},
            {"role": "assistant", "content": sample["solution"]}
        ]
    }

dataset = dataset.map(format_data, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# =========================
# 5️⃣ MODEL & TOKENIZER
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =========================
# 6️⃣ LORA
# =========================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ]
)

# =========================
# 7️⃣ TRAINING ARGS
# =========================
training_args = TrainingArguments(
    output_dir="./results_temp",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=None,   # 🔥 TÜM CHECKPOINTLERİ SAKLA
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    warmup_ratio=0.03,
    max_grad_norm=0.3
)

# =========================
# 8️⃣ TRAINER
# =========================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=1024,
    dataset_text_field="messages",
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        RenameCheckpointCallback()
    ]
)

print("🚀 TRAINING BAŞLIYOR")
trainer.train()
print("✅ TRAINING BİTTİ")
