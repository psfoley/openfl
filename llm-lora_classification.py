#%%
import numpy as np
import torch
from datasets import load_dataset, load_metric
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          get_scheduler)

# %%
task = "mrpc"
dataset = load_dataset("glue", task)
padding_side = "right"

metric = load_metric('glue', task)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

#%%
model_checkpoint = "roberta-base"
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
trainer_model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, return_dict=True)

trainer_peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
)


model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, return_dict=True)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
)

trainer_model = get_peft_model(trainer_model, trainer_peft_config)
trainer_model.print_trainable_parameters()

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

#%%

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
    model=trainer_model,
    args=training_args,
    #train_dataset=tokenized_datasets["train"],
    train_dataset=tokenized_datasets["train"].select(range(1000)),
    #eval_dataset=tokenized_datasets["test"],
    eval_dataset=tokenized_datasets["test"].select(range(1000)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

trainer.train()
#%%
#model.save_pretrained('saved_model')

#%%

#accelerator = Accelerator()
#accelerator.wait_for_everyone()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

from torch import nn

from transformers.trainer_pt_utils import get_parameter_names

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
decay_parameters = [name for name in decay_parameters if "bias" not in name]

optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": WEIGHT_DECAY,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
optimizer = AdamW(optimizer_grouped_parameters, lr=LR)

#%%
train_dataloader = DataLoader(tokenized_datasets["train"].select(range(1000)), shuffle=True, collate_fn=data_collator, batch_size=BATCH_SIZE)
#train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=BATCH_SIZE)
valid_dataloader = DataLoader(tokenized_datasets["test"].select(range(1000)), collate_fn=data_collator, batch_size=BATCH_SIZE)
#valid_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=data_collator, batch_size=BATCH_SIZE)

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
#%%
for epoch in range(NUM_EPOCHS):
    model.train()
    losses = []
    for batch in tqdm(train_dataloader):
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        lr_scheduler.step()
        #optimizer.zero_grad()
        model.zero_grad()
        losses.append(loss.detach())
    accuracy = eval_metrics(model, valid_dataloader)
    print(accuracy, 'loss: ', np.mean(losses))
    
# %%
