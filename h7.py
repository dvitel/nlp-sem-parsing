""" h7 - h0 with distilbert and distilgpt2 together """
import sys
from typing import Optional
from transformers import AutoTokenizer, GPT2LMHeadModel,  DistilBertModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
import evaluate
import numpy as np
import os
import torch 

os.environ["TOKENIZERS_PARALLELISM"] = "true"

ds_name = "dvitel/hearthstone"
out_dir = "out/h7"
result_path = "result/h7"
decoder = "distilgpt2"
encoder = "distilbert-base-uncased"
decoder_max_length = 768
encoder_max_length = 128
batch_size = 4
num_epochs = 200
eval_steps = 1600
learning_rate = 2e-5
seed = 17

np.random.seed(seed)
torch.manual_seed(seed)

def normalize(line:str):
    return line.strip().replace("§", "\n").replace("    ", "\t").replace("\\ ", "").replace("\n\n", "\n")

def preprocess0(e):
    return {"source":e["source"], "target":[normalize(x) for x in e["target"]]}

def unprocess(line):
    return line

ds = load_dataset(ds_name)
ds0 = ds.map(preprocess0, batched = True)

encoder_tokenizer = AutoTokenizer.from_pretrained(encoder)
decoder_tokenizer = AutoTokenizer.from_pretrained(decoder)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
def preprocess(e):
    sources = e["source"]
    encoder_input = encoder_tokenizer(sources, padding = "max_length", truncation = True, max_length = encoder_max_length)  
    targets = [decoder_tokenizer.bos_token + t for t in e["target"]]
    decoder_input = decoder_tokenizer(targets, padding = "max_length", truncation = True, max_length = decoder_max_length)  
    return {"decoder_input_ids": decoder_input["input_ids"], "decoder_attention_mask": decoder_input["attention_mask"],
            "encoder_input_ids": encoder_input["input_ids"], "encoder_attention_mask": encoder_input["attention_mask"]}

ds1 = ds0.map(preprocess, batched = True, remove_columns = ["source", "target"])

encoder_model = DistilBertModel.from_pretrained(encoder, max_length = encoder_max_length)
encoder_model.to("cuda")
decoder_model = GPT2LMHeadModel.from_pretrained(decoder, n_ctx = decoder_max_length, max_length = decoder_max_length, add_cross_attention = True)
decoder_model.to("cuda")

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
      p_text = unprocess(decoder_tokenizer.decode(pred_view))
      l_text = unprocess(decoder_tokenizer.decode(labels_view))
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

class BertGPT2(torch.nn.Module):
    def __init__(self):
        super(BertGPT2, self).__init__()
        self.decoder = decoder_model
        self.encoder = encoder_model

    def forward(
        self, 
        encoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,        
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None
    ):
        bert_result = self.encoder(input_ids = encoder_input_ids, attention_mask = encoder_attention_mask, return_dict = True)
        labels = torch.clone(decoder_input_ids)
        labels[labels == decoder_tokenizer.pad_token_id] = -100
        gpt2_result = self.decoder(input_ids = decoder_input_ids, attention_mask = decoder_attention_mask, labels = labels,
                                    encoder_hidden_states = bert_result.last_hidden_state, return_dict = True)

        return gpt2_result

model = BertGPT2()

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
)

trainer = Trainer(
    model=model,
    args=args,
    compute_metrics = compute_metrics,
    train_dataset=ds1["train"],
    eval_dataset=ds1["validation"],
)

trainer.train(ignore_keys_for_eval = ["past_key_values", "hidden_states", "attentions", "cross_attentions"])

output = trainer.predict(ds1["test"])
print(output.metrics) #test set metrics

trainer.save_model(result_path)
# trainer.push_to_hub()