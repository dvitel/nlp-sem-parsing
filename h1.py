""" h1 == h0 with preprocessing to ast. NN searches ast """
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
import evaluate
import numpy as np
import os
import ast
from ast import *
import astunparse
import torch 

ELIST = "[ELIST]"

def normalize(line:str):
    return line.strip().replace("§", "\n").replace("    ", "\t").replace("\\ ", "").replace("\n\n", "\n")

def process(line):
    tree = ast.parse(line)
    return ast.dump(tree, annotate_fields=False).replace("[]", ELIST)

def unprocess(line):
    no_elist = line.replace(ELIST, "[]")
    res_mod = eval(no_elist)
    res = astunparse.unparse(res_mod).strip().replace("\n\n", "\n").replace("    ", "\t")
    return res
# s = 'class AcidicSwampOoze(MinionCard):§    def __init__(self):§        super().__init__("Acidic Swamp Ooze", 2, CHARACTER_CLASS.ALL, CARD_RARITY.COMMON, battlecry=Battlecry(Destroy(), WeaponSelector(EnemyPlayer())))§§    def create_minion(self, player):§        return Minion(3, 2)§'
# s1 = s.replace("§", "\n").strip().replace("\n\n", "\n").replace("    ", "\t").replace("\\ ", "")
# z = unprocess(process(s1))
# t = ast.parse(s1)
# t.body[0]
# re.compile("(?<=\W)\w+?=\[\]")
# re.sub("(,\s*?|(?<=\W))\w+?=\[\]", "", ast.dump(t.body[0], annotate_fields=True))
# ast.unparse(t)
# dir(ast)
# help(ast.FunctionDef)

ds_name = "dvitel/hearthstone"
out_dir = "out/h1"
result_path = "result/h1"
checkpoint = "distilgpt2"
max_length = 912
batch_size = 4
num_epochs = 200
eval_steps = 1600
learning_rate = 2e-5
seed = 17

np.random.seed(seed)
torch.manual_seed(seed)


def preprocess0(e):
    return {"source":e["source"], "target":[process(normalize(x)) for x in e["target"]]}

ds = load_dataset(ds_name)
ds0 = ds.map(preprocess0, batched = True)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens':[ELIST]})
def preprocess(e):
    alt_bodies = []
    for s, t in zip(e["source"], e["target"]):
      tgt = t # t.replace("(", "[LPAR]").replace(")", "[RPAR]").replace("[RPAR] [LPAR]", "[RPAR][LPAR]")
      alt_bodies.append(s + tokenizer.eos_token + tgt)
    # print(alt_bodies)
    data = tokenizer(alt_bodies, padding = "max_length", truncation = True, max_length = max_length)  
    return data

ds1 = ds0.map(preprocess, batched = True, remove_columns = ["source", "target"])

model = AutoModelForCausalLM.from_pretrained(checkpoint, n_ctx = max_length, max_length = max_length)
model.resize_token_embeddings(len(tokenizer))
model.to("cuda")

bleu = evaluate.load("bleu")
codebleu = evaluate.load("dvitel/codebleu")
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
      p_text = unprocess(tokenizer.decode(pred_view))
      l_text = unprocess(tokenizer.decode(labels_view))
      predictions.append(p_text)
      references.append(l_text)
      if p_text != l_text and first_not_matched > 0:      
        print("EV L", l_text)
        print("EV P", p_text) 
        print()
        first_not_matched -= 1
    accuracy_metric = exact_match.compute(predictions = predictions, references = references)   
    bleu_metric = bleu.compute(predictions = predictions, references = references)   
    codebleu_metric = codebleu.compute(predictions = predictions, references = references)  
    chrf_metric = chrF.compute(predictions = predictions, references = references)  
    return {"exact_match": accuracy_metric["exact_match"], "bleu": bleu_metric["bleu"], **codebleu_metric, "chrf": chrf_metric['score']}

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
    return res

args = TrainingArguments(
    output_dir=out_dir, overwrite_output_dir = True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="steps",
    num_train_epochs = num_epochs,
    logging_steps=eval_steps,
    eval_steps = eval_steps,
    eval_accumulation_steps = 4,
    gradient_accumulation_steps=1,
    weight_decay=0.1,
    # warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=learning_rate,
    save_steps=eval_steps,
    fp16=True, 
    load_best_model_at_end = True, 
    metric_for_best_model = "exact_match",    
    seed = seed,
    hub_model_id = "h1"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    compute_metrics = compute_metrics,
    data_collator=custom_data_collator,
    train_dataset=ds1["train"],
    eval_dataset=ds1["validation"],
)

trainer.train()

output = trainer.predict(ds1["test"])
print(output.metrics) #test set metrics

trainer.save_model(result_path)
trainer.push_to_hub()