#%%
import numpy as np
from datasets import load_dataset, load_metric
from peft import LoraConfig, TaskType, get_peft_model


from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer,
                          TrainingArguments)

# %%
task = "mrpc"
num_epochs = 20
lr = 1e-3
batch_size = 32
dataset = load_dataset("glue", task)
padding_side = "right"

metric = load_metric('glue', task)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

#%%
model_checkpoint = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
    return outputs

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, return_dict=True)
print(len(list(model.named_parameters())))
orig_layers = [n for n,p in model.named_parameters()]

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
)

#%%

model = get_peft_model(model, peft_config)
print(len(list(model.named_parameters())))
withpeft_layers = [n.replace('base_model.model.','') for n,p in model.named_parameters()]
model.print_trainable_parameters()
BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 0.01
#%%

training_args = TrainingArguments(
    output_dir="roberta-large-lora-seq1_tests",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(200)),
    eval_dataset=tokenized_datasets["test"].select(range(200)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()
#%%
#model.save_pretrained('saved_model')
# %%

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import torch
from tqdm import tqdm

#%%
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, return_dict=True)
print(len(list(model.named_parameters())))
orig_layers = [n for n,p in model.named_parameters()]

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
)

#%%

model = get_peft_model(model, peft_config)
print(len(list(model.named_parameters())))
withpeft_layers = [n.replace('base_model.model.','') for n,p in model.named_parameters()]
model.print_trainable_parameters()
BATCH_SIZE = 32
NUM_EPOCHS = 1
LR = 1e-3
WEIGHT_DECAY = 0.01
#%%

#accelerator = Accelerator()
#accelerator.wait_for_everyone()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=LR)


train_dataloader = DataLoader(tokenized_datasets["train"].select(range(200)), collate_fn=data_collator, batch_size=BATCH_SIZE)
valid_dataloader = DataLoader(tokenized_datasets["test"].select(range(200)), collate_fn=data_collator, batch_size=BATCH_SIZE)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * NUM_EPOCHS
)
def eval_metrics(model, dataloader, device='cpu'):
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    model.train()
    return metric.compute()

#model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
#        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
#    )

for epoch in range(NUM_EPOCHS):
    model.train()
    losses = []
    for batch in tqdm(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.detach())

    accuracy = eval_metrics(model, valid_dataloader)
    print(accuracy, 'loss: ', np.mean(losses))
# %%
