#%%
from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig

#path = '/home/oamontoy/workspace/weights-llama-2-7B'
path = '/home/oamontoy/workspace/weights-llama-2-7B-chat'
tokenizer = LlamaTokenizer.from_pretrained(path)
model = LlamaForCausalLM.from_pretrained(path)
# %%
from peft import PeftModel
model = PeftModel.from_pretrained(model, "dominguesm/alpaca-lora-ptbr-7b")
# %%

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is a statement that describes a task, paired with an input that provides more context. Write a response that appropriately completes the request.

### instruction:
{instruction}

### input:
{input}

### response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### instruction:
{instruction}

### response:"""

# %%
from pprint import pprint
# %%
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
)

def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        pprint("response: " + output.split("### response:")[1].strip())
# %%
#evaluate(input("instruction: "))
# %%
instruction = 'print a long paragraph of giberish'
prompt = generate_prompt(instruction, None)
inputs = tokenizer(prompt, return_tensors="pt")
inputs
#%%
input_ids = inputs["input_ids"]
# %%
generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
# %%
for s in generation_output.sequences:
    output = tokenizer.decode(s)
    pprint("response: " + output.split("### response:")[1].strip())
# %%
