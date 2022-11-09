import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import evaluate
import json
import numpy as np
import torch
from parsing import parse2, plain_print, add_arity

# geo_ds_file = "/content/drive/MyDrive/NLP/sem/geoqueries880"
geo_ds_file = sys.argv[2] if len(sys.argv) > 2 else "geoqueries880"
# out_dir = "/content/drive/MyDrive/NLP/sem/out"
out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
checkpoint = "distilgpt2"
max_length = 128
batch_size = 32
num_epochs = 500
eval_steps = 500
learning_rate = 2e-5

with open(geo_ds_file, 'r') as f:
  lines = f.read().splitlines()

# symbols = set()
# shallow_trees = set()
# max_shallow_tree_depth = 2

# def get_tree_depth(t):
#   if type(t) != list:
#     return 0 
#   res = max(get_tree_depth(c) for c in t) + 1
#   if res <= max_shallow_tree_depth:
#     shallow_trees.add(tree_to_str(t))
#   return res

# [[x[1] for x in pl[0][1][2:][0]] for l in lines for pl in [parse(l)]][0]

geo_ds_pairs = []
symbol_arities = {}
rev_symbol_arities = {}
terminals = set()
for l in lines:
  [_, queryl, ast] = parse2(l)
  query = " ".join(queryl)
  ast_with_arity = add_arity(ast, symbol_arities = symbol_arities, rev_symbol_arities = rev_symbol_arities, terminals = terminals)
  pprinted = plain_print(ast_with_arity, symbol_categories=rev_symbol_arities)
  geo_ds_pairs.append({"source": query, "target": pprinted })

# geo_ds_pairs[0]

# l2 = [{"source": s, "target": s} for s in shallow_trees]
geo_ds = Dataset.from_list(geo_ds_pairs)
geo_dss = geo_ds.train_test_split(test_size = 280)
# geo_dss["train"] = Dataset.from_list([*[l for l in geo_dss["train"]]])

# symbols
# geo_dss["train"][0]
# tokenizer("fewest")
# tokenizer.decode(11925)

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

categories = {}
for arity, group in symbol_arities.items(): #detect symbol categories and add to specific rows
  for s in group: #categories
    tokens = tokenizer(s).input_ids
    if len(tokens) == 1: #note that arity is only checked for 1 token funcs
      categories.setdefault(arity, set()).add(tokens[0])
    if s.endswith("id") and len(s) > 2: #stateid, riverid etc should only be under const 
      categories.setdefault("const", set()).add(tokens[0])
category_ids = {}
groups = [] 
max_len = 0
for i, (c, ids) in enumerate(sorted(categories.items(), key = lambda x: x[0])):
  category_ids[i] = c
  groups.append(list(ids))
  max_len = max(max_len, len(ids))
pad_id = -100
for g in groups:
  for i in range(len(g), max_len):
    g.append(pad_id)

var_ids = []
for t in terminals:
  if t.isupper() and len(t) == 1:
    var_ids.append(tokenizer(t).input_ids[0])

# var_ids_tensor = torch.tensor(var_ids, device = "cuda")
groups_tensor = torch.tensor(groups, device = "cuda")

processed_dss = geo_dss.map(preprocess, batched = True, remove_columns = ["source", "target"])

# geo_dss['train'][161]['target']

# len([t for t in processed_dss['train'][391]['input_ids'] if t != 50256]) + 1

# tokenizer.decode(processed_dss['train'][391]['input_ids'])

# [i for i, r in enumerate(processed_dss['train']['attention_mask']) if r[-1] == 1]

# tokenizer.decode(50256)

model = AutoModelForCausalLM.from_pretrained(checkpoint, n_ctx = max_length, max_length = max_length)
# model.resize_token_embeddings(len(tokenizer))
model.to("cuda")

bleu = evaluate.load("bleu")
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
    metric = exact_match.compute(predictions = predictions, references = references)   
    return metric

# and_id = tokenizer("[AND]").input_ids[0]
# eos_id = tokenizer.eos_token_id
# rpar_id = tokenizer("[RPAR]").input_ids[0]
# ws_id = tokenizer(" ").input_ids[0]
# enforced_tokens = [lpar_id, rpar_id]
# eos_id

# torch.set_printoptions(edgeitems=100)

# groups_tensor.shape

# x = torch.tensor([[1,2,3],[4,5,6]])
# x_shape = x.shape
# y = torch.tensor([2,3,5,7])
# # y[:,None]
# y.unsqueeze(-1)
# z = (x.view(-1).repeat(len(y), 1) == y.unsqueeze(-1)).float().reshape((len(y), *x_shape)).sum(dim=-1)
# z
# # z.sum(axis=[1,2])
# x.unsqueeze(0).repeat(len(y), 1, 1) == y

def grammar_weighted_loss(label_ids, logits, var_err=2.0, func_err=5.0, category_err = 10.0):
  """ Computes weighted loss according to grammar of target sentence language """
  shift_labels = label_ids[...,1:].contiguous()
  shift_logits = logits[..., :-1, :].contiguous()
  predictions = torch.argmax(shift_logits, dim=-1)
  # print("Predictions", predictions[0])
  # print("Shifted logits", shift_logits.shape)
  # print("Shifted labels", shift_labels[0])  
  loss_fct = torch.nn.CrossEntropyLoss(reduce = False)
  shift_logits_view = shift_logits.view(-1, shift_logits.size(-1))
  shift_labels_view = shift_labels.view(-1)
  # print("Shifted logits view", shift_logits_view.shape)
  # print("Shifted labels view", shift_labels_view.shape)
  loss = loss_fct(shift_logits_view, shift_labels_view)
  # print("Loss", loss.shape)
  loss_view = loss.view(shift_logits.size(0), shift_logits.size(1))
  # print("Loss/token", loss_view.shape)
  # ------------------------
  w = torch.ones_like(loss_view)
  misses = (shift_labels != -100).float()
  misses *= (predictions != shift_labels).float()
  # print("Misses", misses)
  # missed_predicted = tokenizer.decode()
  # missed_golden = tokenizer.decode()
  missed_shift_labels = shift_labels[misses > 0]
  # print("ML", missed_shift_labels)
  missed_predictions = predictions[misses > 0]
  # print("MP", missed_predictions)
  golden = (groups_tensor.view(-1).repeat(len(missed_shift_labels), 1) == missed_shift_labels.unsqueeze(-1)).float().reshape((len(missed_shift_labels), *groups_tensor.shape)).sum(dim=-1)
  # print("G", golden)
  pred = (groups_tensor.view(-1).repeat(len(missed_predictions), 1) == missed_predictions.unsqueeze(-1)).float().reshape((len(missed_predictions), *groups_tensor.shape)).sum(dim=-1)
  # print("P", pred)
  delta = torch.abs(golden - pred).sum(dim=-1)
  # print("D", delta)
  w[misses > 0] += func_err * (golden.sum(dim=-1) > 0).float() 
  w[misses > 0] += category_err * delta  
  for var_id in var_ids:
    w[((shift_labels == var_id).float() * misses) > 0] += var_err
  
  # for i, (g, p) in enumerate(zip(shift_labels[misses > 0], predictions[misses > 0])):
  #   print("G", g)
  #   print("P", p)    
  #   pos_g = (groups_tensor == g).float().sum(dim=-1)
  #   pos_p = (groups_tensor == p).float().sum(dim=-1)
  #   delta = torch.abs(pos_g - pos_p).sum()
  #   w_m[i] += arity_err * delta
  #   print("delta", delta)
  # w += wrong_paren * misses * ((predictions == lpar_id).float() + (predictions == rpar_id).float())
  # w += missing_paren * misses * ((shift_labels == lpar_id).float() + (shift_labels == rpar_id).float())
  # print("W", w)
  loss_view *= w
  # --------------------------------
  loss_per_sample = loss_view.mean(axis=1)
  # print("Loss/sample", loss_per_sample.shape)
  #this should be done according to grammar
  # weights_view = torch.stack([(label_ids == kt).float() for kt in enforced_tokens])
  # weights = weights_view.sum(axis=[0, 2])
  # print("Weights view", weights_view.shape)
  # print("Weights", weights.shape)
  # print("Weigths ", weights)
  # weights = alpha * (1.0 + weights)
  # weighted_loss = (loss_per_sample * weights).mean()
  weighted_loss = loss_per_sample.mean()
  # print("Weighted loss", weighted_loss.shape)
  return weighted_loss

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

class GrammarTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs = False):
    input_ids = inputs["input_ids"]    
    # print("Present data", inputs.keys())
    model_inputs = {k:v for k,v in inputs.items() if  k != "labels"}
    outputs = model(**model_inputs)
    loss = grammar_weighted_loss(inputs["labels"], outputs.logits)
    return (loss, outputs) if return_outputs else loss

trainer = GrammarTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    compute_metrics = compute_metrics,
    data_collator=custom_data_collator,
    train_dataset=processed_dss["train"],
    eval_dataset=processed_dss["test"],
)

# import gc
# gc.collect()
# torch.cuda.empty_cache()

trainer.train()

# l = tokenizer(["[AND]2[size:2][id:1]B[id:1]A[const:2][id:1]B[cityid:2][id:1]new york[id:1]_", "[count:3][id:1]B[AND]2[major:1][id:1]B[city:1][id:1]B[id:1]A"], return_tensors="pt", padding = "max_length", truncation = True, max_length = max_length).input_ids
# p = tokenizer(["[AND]2[size:2][id:1]B[id:1]A[const:2][id:1]B[stateid:1][id:1]new york[id:1]A", "[count:3][id:1]B[AND]4[major:1][id:1]B[city:1][id:1]B[loc:2]C"], return_tensors="pt", padding = "max_length", truncation = True, max_length = max_length).input_ids
# l[l == 50256] = -100

# l

# w = torch.ones_like(l, dtype=torch.float)

# misses = (l != -100).float()
# misses *= (p != l).float()
# misses

# # print("Misses", misses)
# # missed_predicted = tokenizer.decode()
# # missed_golden = tokenizer.decode()
# missed_shift_labels = l[misses > 0]
# # print("ML", missed_shift_labels)
# missed_predictions = p[misses > 0]
# missed_shift_labels
# missed_predictions

# golden = (groups_tensor.cpu().view(-1).repeat(len(missed_shift_labels), 1) == missed_shift_labels.unsqueeze(-1)).float().reshape((len(missed_shift_labels), *groups_tensor.shape)).sum(dim=-1)
# golden

# pred = (groups_tensor.cpu().view(-1).repeat(len(missed_predictions), 1) == missed_predictions.unsqueeze(-1)).float().reshape((len(missed_predictions), *groups_tensor.shape)).sum(dim=-1)
# pred

# delta = torch.abs(golden - pred).sum(dim=-1)
# delta

# w[misses > 0] += 2.0 * delta
# w

