""" h2 == h0 with light name removing """
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import evaluate
import numpy as np
import os
import ast
import torch 
import astunparse
CLSN = "[CLSN]"
# INIT = "[INIT]"
# NOARG = "[NOARG]"

# dir(ast.parse("Minion(1,2)").body[0])

class NameRemover(ast.NodeVisitor):
    ''' TODO; reverse mode or renaming back '''
    def __init__(self) -> None:
        super().__init__()
        self.reset()
    def reset(self):
        self.mapping = {}
        self.class_mapping = {}
        self.rev_mapping = {}
        self.rev_class_mapping = {}
        self.i = 0   
        self.cls_i = 0
    def add_to_vocab(self, name, token = None):
        if name not in self.mapping:
            new_name = self.mapping[name] = token or ("[v" + str(self.i) + "]")
            self.rev_mapping[new_name] = name
            self.i += 1
        return self.mapping[name]
    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store):
            node.id = self.add_to_vocab(node.id)
        elif node.id in self.mapping:
            node.id = self.mapping[node.id]
        elif node.id in self.class_mapping:
            node.id = self.class_mapping[node.id]
    def generic_visit(self, node):
        if isinstance(node, ast.ClassDef):
            if node.name not in self.class_mapping:
                new_name = self.class_mapping[node.name] = "[CLS" + str(self.cls_i) + "]"
                self.rev_class_mapping[new_name] = node.name
                self.cls_i += 1
            node.name = self.class_mapping[node.name]
        if isinstance(node, ast.arguments):
            for arg in node.args:
                if isinstance(arg, ast.arg) and arg.arg != "self":
                    arg.arg = self.add_to_vocab(arg.arg)
        return super().generic_visit(node)

name_remover = NameRemover()
name_symbols = set()
def process(line):
    tree = ast.parse(line)
    name_remover.reset()
    name_remover.visit(tree)
    name_symbols.update(name_remover.rev_mapping.keys())
    name_symbols.update(name_remover.rev_class_mapping.keys())
    res = astunparse.unparse(tree).strip().replace("\n\n", "\n").replace("    ", "\t")
    return res

import re
symbol_pattern = r"\[(CLS\d+|v\d+)\]"
def unprocess(line: str):
    res = re.sub(symbol_pattern, r"\1", line)
    return res #we preserve name of symbols but remove []

# s = 'class AcidicSwampOoze(MinionCard):§    def __init__(self):§        super().__init__("Acidic Swamp Ooze", 2, CHARACTER_CLASS.ALL, CARD_RARITY.COMMON, battlecry=Battlecry(Destroy(), WeaponSelector(EnemyPlayer())))§§    def create_minion(self, player):§        return Minion(3, 2)§'
# s1 = s.replace("§", "\n")
# s2 = 'class FlameOfAzzinoth(MinionCard):§    def __init__(self):§        super().__init__("Flame of Azzinoth", 1, CHARACTER_CLASS.ALL, CARD_RARITY.COMMON, False)§§    def create_minion(self, player):§        return Minion(2, 1)§'
# s2 = s2.replace("§", "\n")
# process(s1)
# print()
# process(s2)
# name_symbols
# t = ast.parse(s1)
# t.body[0]
# re.compile("(?<=\W)\w+?=\[\]")
# re.sub("(,\s*?|(?<=\W))\w+?=\[\]", "", ast.dump(t.body[0], annotate_fields=True))
# ast.unparse(t)
# dir(ast)
# help(ast.FunctionDef)

# geo_ds_file = "/content/drive/MyDrive/NLP/sem/geoqueries880"
hs_folder = sys.argv[2] if len(sys.argv) > 2 else "hearthstone"
train_file_name = "train_hs"
test_file_name = "test_hs"
dev_file_name = "dev_hs"
# out_dir = "/content/drive/MyDrive/NLP/sem/out"
# out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
out_dir = "out/h3"
result_path = "result/h3"
checkpoint = "distilgpt2"
max_length = 912
batch_size = 4
num_epochs = 200
eval_steps = 1600
learning_rate = 2e-5
seed = 17

np.random.seed(seed)
torch.manual_seed(seed)

def normalize(line:str):
    return line.strip().replace("§", "\n").replace("    ", "\t").replace("\\ ", "").replace("\n\n", "\n")

def read_samples(file_name):
    with open(os.path.join(hs_folder, file_name + ".in"), 'r') as f:
        train_source_lines = f.read().splitlines()

    with open(os.path.join(hs_folder, file_name + ".out"), 'r') as f:
        train_target_lines = f.read().splitlines()    

    return [{"source": s, "target": process(normalize(t))} 
                for (s, t) in zip(train_source_lines, train_target_lines)]

train_set = Dataset.from_list(read_samples(train_file_name))
dev_set = Dataset.from_list(read_samples(dev_file_name))
test_set = Dataset.from_list(read_samples(test_file_name))

#First we experiment without any code preprocessing 

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens':[*list(name_symbols)]})
def preprocess(e):
    alt_bodies = []
    for s, t in zip(e["source"], e["target"]):
      tgt = t # t.replace("(", "[LPAR]").replace(")", "[RPAR]").replace("[RPAR] [LPAR]", "[RPAR][LPAR]")
      alt_bodies.append(s + tokenizer.eos_token + tgt)
    # print(alt_bodies)
    data = tokenizer(alt_bodies, padding = "max_length", truncation = True, max_length = max_length)  
    return data

p_train_set = train_set.map(preprocess, batched = True, remove_columns = ["source", "target"])
p_test_set = test_set.map(preprocess, batched = True, remove_columns = ["source", "target"])
p_dev_set = dev_set.map(preprocess, batched = True, remove_columns = ["source", "target"])
#print("Max length train", len(max(p_train_set['input_ids'], key=lambda x: len(x))))
#print("Max length dev", len(max(p_dev_set['input_ids'], key=lambda x: len(x))))
#print("Max length test", len(max(p_test_set['input_ids'], key=lambda x: len(x))))

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
    seed = seed
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    compute_metrics = compute_metrics,
    data_collator=custom_data_collator,
    train_dataset=p_train_set,
    eval_dataset=p_dev_set,
)

trainer.train()

output = trainer.predict(p_test_set)
print(output.metrics) #test set metrics

trainer.save_model(result_path)