''' g0 - base implementation which picks source and target sentences as is '''

import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import evaluate
import torch
import numpy as np

geo_ds_file = "geoqueries880"
out_dir = "out/g0"
result_path = "result/g0"
checkpoint = "distilgpt2"
max_length = 128
batch_size = 32
num_epochs = 1000
eval_steps = 1000
learning_rate = 2e-5
seed = 17

np.random.seed(seed)
torch.manual_seed(seed)

with open(geo_ds_file, 'r') as f:
  lines = f.read().splitlines()

def parse(line: str):
  prefix_queryl = "parse(["
  prefix_astl = " answer(A,"
  querys, asts = line.split("],")
  queryl = querys[len(prefix_queryl):].split(",")
  queryl[-1] = queryl[-1].replace("'.'", '.')
  source = " ".join(queryl)
  target = asts[len(prefix_astl):-3]
  return {"source": source, "target": target}

geo_ds = Dataset.from_list([parse(l) for l in lines])
geo_dss = geo_ds.train_test_split(test_size = 280, seed = seed)

#NOTE: preprocessing - concat source and target with [SEP]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'additional_special_tokens':[*symbols]})
def preprocess(e):
    alt_bodies = []
    for s, t in zip(e["source"], e["target"]):
      tgt = t # t.replace("(", "[LPAR]").replace(")", "[RPAR]").replace("[RPAR] [LPAR]", "[RPAR][LPAR]")
      alt_bodies.append(s + tokenizer.eos_token + tgt)
    # print(alt_bodies)
    data = tokenizer(alt_bodies, padding = "max_length", truncation = True, max_length = max_length)  
    return data

processed_dss = geo_dss.map(preprocess, batched = True, remove_columns = ["source", "target"])

model = AutoModelForCausalLM.from_pretrained(checkpoint, n_ctx = max_length, max_length = max_length)
# model.resize_token_embeddings(len(tokenizer))
model.to("cuda")

bleu = evaluate.load("bleu")
chrF = evaluate.load("chrf")
exact_match = evaluate.load("exact_match")
def compute_metrics(eval_pred):
  shift_labels = eval_pred.label_ids[...,1:]
  shift_logits = eval_pred.predictions[..., :-1, :]
  prediction_labels = np.argmax(shift_logits, axis=-1)
  predictions = []
  references = []
  first_not_matched = 4
  for preds, labels in zip(prediction_labels, shift_labels):      
    label_map = labels >= 0
    labels_view = labels[label_map]
    pred_view = preds[label_map]
    p_text = tokenizer.decode(pred_view)
    l_text = tokenizer.decode(labels_view)    
    predictions.append(p_text)
    references.append(l_text)
    if p_text != l_text and first_not_matched > 0:      
      print("EV L", l_text)
      print("EV P", p_text) 
      print()
      first_not_matched -= 1
  accuracy_metric = exact_match.compute(predictions = predictions, references = references)   
  chrF_metric = chrF.compute(predictions = predictions, references = references)
  return {**accuracy_metric, **chrF_metric}

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
eos_id = tokenizer.eos_token_id
def custom_data_collator(*args):
  ''' we do not need to deduce preefix parts - change all labels till first -100 to -100 '''
  res = data_collator(*args)
  for l in res['labels']:
    i = 0
    while l[i] != -100:
      l[i] = -100 
      i += 1 
  # res['l'] = res['labels']
  # del res['labels']
  return res

args = TrainingArguments(
    output_dir=out_dir, overwrite_output_dir = True,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    evaluation_strategy = "steps",
    save_strategy = "steps",
    num_train_epochs = num_epochs,
    logging_steps = eval_steps,
    eval_steps = eval_steps,
    gradient_accumulation_steps = 1,
    weight_decay = 0.1,
    # warmup_steps=1_000,
    lr_scheduler_type = "cosine",
    learning_rate = learning_rate,
    save_steps = eval_steps,
    fp16 = True,
    load_best_model_at_end = True, 
    metric_for_best_model = "exact_match",
    seed = seed
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    compute_metrics = compute_metrics,
    data_collator=custom_data_collator,
    train_dataset=processed_dss["train"],
    eval_dataset=processed_dss["test"],
)

trainer.train()

trainer.save_model(result_path)