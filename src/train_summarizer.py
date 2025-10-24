from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    Trainer
)
from config import *
from data_ingestion import load_data, split_data

dataset = load_data(DATASET_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def preprocess(example):
    inputs = tokenizer(example["input"], max_length=512, truncation=True)
    labels = tokenizer(example["output"], max_length=128, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=["id", "input", "output"])
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = TrainingArguments(
    output_dir = OUTPUR_DIR,
    eval_strategy = "epoch",
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    num_train_epochs = 1,
    logging_steps = 50,
    save_total_limit = 2,
    push_to_hub = True,
    hub_model_id = HUB_MODEL_ID
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"].select(range(2000)),
    eval_dataset=tokenized["validation"].select(range(500)),
    tokenizer=tokenizer,
    data_collator=data_collator
)


