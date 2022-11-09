''' g3 == g2 + all funcs are additional tokens '''

import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import evaluate
import torch
import numpy as np

geo_ds_file = "geoqueries880"
out_dir = "out/g3"
result_path = "result/g3"
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

def prolog_expr_preprocess(s: str) -> str:
  return s.replace('\\+','not ').replace('[', '(').replace(']', ')').replace("'.'", '.')[:-1]

def parse_sexpr(s: str):
    ''' converts string to tree - assuming Uniterpreted functions of form f(...)! '''
    s = prolog_expr_preprocess(s)
    symbs = {'(', ')', ','}
    def parse_name(acc, i):
        i0 = i
        while i < len(s) and s[i] not in symbs:
            i += 1
        if i < len(s) and s[i] == '(':
            f = [s[i0:i].strip()]
            acc.append(f)
            i += 1
            i = parse_params(f, i) #after this f contains all args
            assert s[i] == ')', f"Not at ) {i} for: {s}"
            i += 1 #passing )
        else: #end 
            acc.append(s[i0:i])
        return i
    def parse_params(acc, i):
        while i < len(s) and s[i] != ')':
            i = parse_name(acc, i)
            if s[i] == ',':
                i += 1
        return i
    acc = []
    i = parse_name(acc, 0)
    assert i == len(s), f"Not at end {i} for: {s}"
    return acc[0]
  
symbols = set()
def s_expr_to_str(t, lpar="(", rpar=")"):
  if type(t) != list:
    return t
  if len(t) > 1:
    args = " ".join(s_expr_to_str(x, lpar = lpar, rpar = rpar) for x in t[1:])
    if t[0] == '':      
      sep = ''
      symbol = None 
    else:
      sep = " "    
      symbol = "[" + t[0] + "]"
    sep_args = sep + args 
  else:
    symbol = "[" + t[0] + "]"
    sep_args = ""
  if symbol:
    symbols.add(symbol)
  return lpar + (symbol if symbol is not None else t[0]) + sep_args + rpar

def parse(line: str): 
  [_, queryl, ast] = parse_sexpr(line)
  source = " ".join(x for x in queryl if x != "")
  target = s_expr_to_str(ast, lpar="[LPAR]", rpar="[RPAR]").replace("[RPAR] [LPAR]", "[RPAR][LPAR]").replace(" [", "[").replace("] ", "]")
  return {"source": source, "target": target}

geo_ds = Dataset.from_list([parse(l) for l in lines])
geo_dss = geo_ds.train_test_split(test_size = 280, seed = seed)

#NOTE: preprocessing - concat source and target with [SEP]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens':["[LPAR]", "[RPAR]", *list(symbols)]})
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
model.resize_token_embeddings(len(tokenizer))
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