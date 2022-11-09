from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import evaluate
import json
import numpy as np
import torch
import sys

ds_file = sys.argv[1] if len(sys.argv) > 1 else "geoqueries880"
out_dir = "out"
checkpoint = "distilgpt2"
max_length = 96
batch_size = 32
num_epochs = 500
eval_steps = 500
learning_rate = 2e-5

# s = parse("parse([could,you,tell,me,what,is,the,highest,point,in,the,state,of,oregon,?], answer(A,highest(A,(place(A),loc(A,B),state(B),const(B,stateid(oregon)))))).")
# print(s)

print("GPU: ", torch.cuda.get_device_properties("cuda"))
print("CUDA: ", torch.version.cuda)

with open(ds_file, 'r') as f:
    lines = f.read().splitlines()

ds_pairs = [(" ".join(pl[0][1][1:]), tree_to_str(pl[0][2][2])) for l in lines for pl in [parse(l)]]

ds = Dataset.from_list([{"source": s, "target": t} for s, t in ds_pairs])
print(ds)

#NOTE: preprocessing - concat source and target with [SEP]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
def preprocess(e):
    alt_bodies = [s + " " + tokenizer.eos_token + " " + t for s, t in zip(e["source"], e["target"])]
    data = tokenizer(alt_bodies, padding = "max_length", truncation = True, max_length = max_length)  
    return data

preprocessed_ds = ds.map(preprocess, batched = True, remove_columns = ["source", "target"])    
#assert all data fit into given max_length 
longed_data_ids = [i for i, r in enumerate(preprocessed_ds['attention_mask']) if r[-1] == 1]
assert len(longed_data_ids) == 0

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
def custom_data_collator(*args):
    ''' we do not need to deduce preefix parts - change all labels till first -100 to -100 '''
    res = data_collator(*args)
    for l in res['labels']:
        i = 0
        while l[i] != -100:
            l[i] = -100 
            i += 1 
    return res

# tokenizer.decode([7])    
# out = custom_data_collator([geo_ds[i] for i in range(5)])
model = AutoModelForCausalLM.from_pretrained(checkpoint, n_ctx = max_length, max_length = max_length).to("cuda")
dss = preprocessed_ds.train_test_split(test_size = 280, seed = 42)

bleu = evaluate.load("bleu")
exact_match = evaluate.load("exact_match")
def compute_metrics(eval_pred):
    predictions = []
    references = []
    first_not_matched = 2
    for ps, ls in zip(eval_pred.predictions, eval_pred.label_ids):
        idx = np.where(ls != -100)[0]
        p_idx = np.append(idx[0] - 1, idx[:-1])
        p_text = tokenizer.decode(np.argmax(ps[p_idx], axis=-1))
        l_text = tokenizer.decode(ls[idx])
        predictions.append(p_text)
        references.append(l_text)
        if p_text != l_text and first_not_matched > 0:
            print("Predictions: ", p_text)
            print("Labels: ", l_text)
            print()
            first_not_matched -= 1
    # metric = bleu.compute(predictions = predictions, references = references)   
    metric = exact_match.compute(predictions = predictions, references = references)   
    return metric

args = TrainingArguments(
    output_dir=out_dir, overwrite_output_dir = True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="steps",
    num_train_epochs = num_epochs,
    logging_steps=eval_steps,
    eval_steps = eval_steps,
    gradient_accumulation_steps=1,
    weight_decay=0.1,
    # warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=learning_rate,
    save_steps=eval_steps,
    fp16=True
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    compute_metrics = compute_metrics,
    data_collator=custom_data_collator,
    train_dataset=dss["train"],
    eval_dataset=dss["test"],
)    

trainer.train()