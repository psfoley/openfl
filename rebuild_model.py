# %%
import numpy as np
from datasets import load_dataset, load_metric
from peft import LoraConfig, TaskType, get_peft_model, LoraModel


from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.utils import find_adapter_config_file
from peft import PeftConfig, PeftType
from peft.utils.save_and_load import get_peft_model_state_dict
import transformers.modeling_utils
from peft import AutoPeftModelForTokenClassification, PeftModelForSequenceClassification
from peft.utils import set_peft_model_state_dict, get_peft_model_state_dict


# %%
# trained model and tokenizer
model_checkpoint = "/home/oamontoy/workspace/saved_model"
tokenizer_name = "roberta-large"
padding_side = "right"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


task = "mrpc"
metric = load_metric("glue", task)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 0.01


def dataset_function(tokenizer, task="mrpc"):
    dataset = load_dataset("glue", task)

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=None,
        )
        return outputs

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    return data_collator, tokenized_datasets


data_collator, tokenized_datasets = dataset_function(tokenizer, task)

# %%
# get initial model

model = AutoModelForSequenceClassification.from_pretrained(
    tokenizer_name, return_dict=True
)
model = PeftModelForSequenceClassification.from_pretrained(model,
    model_checkpoint
)
# %%
# test performance
training_args = TrainingArguments(
    output_dir="roberta-large-lora-seq",
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
    train_dataset=tokenized_datasets["train"].select(range(500)),
    eval_dataset=tokenized_datasets["test"].select(range(200)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)
# %%
print(trainer.evaluate())
# %%
lora_params = get_peft_model_state_dict(model)
tokenizer_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model_re = AutoModelForSequenceClassification.from_pretrained(
    tokenizer_name, return_dict=True
)
loaded_peft_config = PeftConfig.from_pretrained(model_checkpoint)
model_re = get_peft_model(model_re, loaded_peft_config)
# %%


def set_params(
    model,
    processed_adapter_state_dict,
    adapter_name="default",
    device_map="auto",
    max_memory=None,
    offload_folder=None,
    offload_index=None,
):
    model._hf_peft_config_loaded = True
    incompatible_keys = set_peft_model_state_dict(
        model, processed_adapter_state_dict, adapter_name
    )
    model._dispatch_accelerate_model(
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_folder,
        offload_index=offload_index,
    )
set_params(model_re, lora_params, 'default')
print(model_re.peft_config["default"])
# %%


# %%
data_collator, tokenized_datasets = dataset_function(tokenizer, task)
training_args = TrainingArguments(
    output_dir="roberta-large-lora-seq",
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
    model=model_re,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(500)),
    eval_dataset=tokenized_datasets["test"].select(range(200)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# %%
print(trainer.evaluate())
#%%

p = tokenized_datasets["train"][0]['input_ids'].unsqueeze(0)
model_re(p)
# %%
model(tokenized_datasets["train"][0]['input_ids'].unsqueeze(0))
# %%
