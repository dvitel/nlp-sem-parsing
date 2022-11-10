import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import evaluate
import numpy as np
import os
import torch 

# geo_ds_file = "/content/drive/MyDrive/NLP/sem/geoqueries880"
hs_folder = sys.argv[2] if len(sys.argv) > 2 else "hearthstone"
train_file_name = "train_hs"
test_file_name = "test_hs"
dev_file_name = "dev_hs"
# out_dir = "/content/drive/MyDrive/NLP/sem/out"
# out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
out_dir = "out/h0"
result_path = "result/h0"
checkpoint = "distilgpt2"
max_length = 912
batch_size = 4
num_epochs = 100
eval_steps = 800
learning_rate = 2e-5
seed = 17

np.random.seed(seed)
torch.manual_seed(seed)


def read_samples(file_name):
    with open(os.path.join(hs_folder, file_name + ".in"), 'r') as f:
        train_source_lines = f.read().splitlines()

    with open(os.path.join(hs_folder, file_name + ".out"), 'r') as f:
        train_target_lines = f.read().splitlines()    

    return [{"source": s, "target": t.replace("§", "\n").replace("    ", "\t").replace("\\ ", "")} for (s, t) in zip(train_source_lines, train_target_lines)]

train_set = Dataset.from_list(read_samples(train_file_name))
dev_set = Dataset.from_list(read_samples(dev_file_name))
test_set = Dataset.from_list(read_samples(test_file_name))

#First we experiment without any code preprocessing 

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

p_train_set = train_set.map(preprocess, batched = True, remove_columns = ["source", "target"])
p_test_set = test_set.map(preprocess, batched = True, remove_columns = ["source", "target"])
p_dev_set = dev_set.map(preprocess, batched = True, remove_columns = ["source", "target"])
#print("Max length train", len(max(p_train_set['input_ids'], key=lambda x: len(x))))
#print("Max length dev", len(max(p_dev_set['input_ids'], key=lambda x: len(x))))
#print("Max length test", len(max(p_test_set['input_ids'], key=lambda x: len(x))))

model = AutoModelForCausalLM.from_pretrained(checkpoint, n_ctx = max_length, max_length = max_length)
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

trainer.save_model(result_path)