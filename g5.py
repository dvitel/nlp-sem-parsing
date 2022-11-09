''' g5 == g4 + func:arity are made as new tokens '''

import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import evaluate
import torch
import numpy as np

geo_ds_file = "geoqueries880"
out_dir = "out/g4"
result_path = "result/g4"
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
            name = s[i0:i].strip()            
            if name.startswith("not "):
                name = name.split(' ')[-1]
                f = [name]
                acc.append(["not", f])
            else:
                f = [name]
                acc.append(f)
            i += 1
            i = parse_params(f, i) #after this f contains all args
            if f[0] == "not": #make not to have only one arg
                f[1:] = [['', *f[1:]]] 
            # if f[0] == "":
            #     f[0] = "AND"
            #     f[1:] = [len(f) - 1, f[1:]] 
            # f[0] = f[0] if f[0] == "AND" or f[0] == "NOT" else f"{f[0]}:{str(len(f) - 1)}"
            assert s[i] == ')', f"Not at ) {i} for: {s}"
            i += 1 
        else:
            acc.append(s[i0:i].strip("'"))
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
    def split_list(l):
        if type(l) != list:
            return l
        elif l[0] == '':
            if not any(type(x) == list for x in l):
                return l[1:]
            root = subt = []
            for i, el in enumerate(l):
                if i == 0:
                    continue
                subt.append('and')                
                subt.append(split_list(el))
                if i == len(l) - 2:
                    subt.append(split_list(l[-1]))
                    break
                else:                
                    new_subt = []
                    subt.append(new_subt)
                    subt = new_subt
            return root
        else:
            return [split_list(el) for el in l]                
    res = split_list(acc)
    return res[0]
  
def add_arity(t, symbol_arities = {}, rev_symbol_arities = {}, terminals = set()):
    if type(t) != list:
        terminals.add(t)
        return t
    arity = str(len(t) - 1)
    symbol_arities.setdefault(arity, set()).add(t[0])
    rev_symbol_arities.setdefault(t[0], set()).add(arity)
    return [(t[0], arity), *[add_arity(ch, symbol_arities = symbol_arities, rev_symbol_arities = rev_symbol_arities, terminals = terminals) for ch in t[1:]]] 

symbols = set()
def plain_print(t, sep = ":", symbol_categories = {}):
    tokens = []
    def pprint_inner(t):
        for ch in t:
            if type(ch) == list:
                pprint_inner(ch)                
            elif type(ch) == tuple:
                if len(symbol_categories.get(ch[0], [])) == 1:
                    tokens.append(str(ch[0]))
                else:
                    symbol = "[" + sep.join([str(x) for x in ch]) + "]"
                    symbols.add(symbol)
                    tokens.append(symbol)
            else: 
                tokens.append(str(ch))
    pprint_inner(t)
    return " ".join(tokens)

symbol_arities = {}
rev_symbol_arities = {}
terminals = set()
def parse(line: str): 
  [_, queryl, ast] = parse_sexpr(line)
  source = " ".join(x for x in queryl if x != "")
  ast_with_arity = add_arity(ast, symbol_arities = symbol_arities, rev_symbol_arities = rev_symbol_arities, terminals = terminals)
  target = plain_print(ast_with_arity).replace("] ", "]").replace(" [", "[")
  return {"source": source, "target": target}

# parse("parse([how,many,states,border,colorado,and,border,new,mexico,?], answer(A,count(B,(state(B),next_to(B,C),const(C,stateid(colorado)),next_to(B,D),const(D,stateid('new mexico'))),A))).")

geo_ds = Dataset.from_list([parse(l) for l in lines])
geo_dss = geo_ds.train_test_split(test_size = 280, seed = seed)

#NOTE: preprocessing - concat source and target with [SEP]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens':[x for x in symbols]})
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
  return accuracy_metric

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