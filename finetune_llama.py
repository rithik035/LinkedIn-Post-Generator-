from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer



# Load data
dataset = load_dataset("json", data_files="data/train.json", split="train")

# Load tokenizer and model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Preprocess
def tokenize(example):
    prompt = f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"
    return tokenizer(prompt, truncation=True, padding='max_length', max_length=512)

dataset = dataset.map(tokenize)

# Add LoRA config
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
                         r=8, lora_alpha=16, lora_dropout=0.05)
model = get_peft_model(model, peft_config)

# Trainer setup
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    learning_rate=1e-4,
    fp16=True,
    evaluation_strategy="no"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

trainer.train()
