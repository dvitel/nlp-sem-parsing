""" h6 == h4 but ignore all loss after first error """
import sys
from typing import Optional
from transformers import AutoTokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from datasets import Dataset
import evaluate
import numpy as np
import os
import ast
import torch 
import astunparse
from grammar import LST, NEND, GrammarCollector, Symbol, SymbolAttr, start_symbol

# torch.autograd.set_detect_anomaly(True)

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
grammar_collector = GrammarCollector()
name_symbols = set()
def process_to_ast(line):
    tree = ast.parse(line)    
    name_remover.reset()
    name_remover.visit(tree)
    grammar_collector.collect_metadata(tree) #after removing name
    name_symbols.update(name_remover.rev_mapping.keys())
    name_symbols.update(name_remover.rev_class_mapping.keys())
    return tree
    # res = astunparse.unparse(tree)
    # return res.strip().replace("    ", "\t").replace("\n\n", "\n") #.replace(".__init__", INIT).replace("()", NOARG)

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
out_dir = "out/h6"
result_path = "result/h6"
checkpoint = "distilgpt2"
max_length = 912
# max_length = 32 #for debugging
batch_size = 4
num_epochs = 200
eval_steps = 1600
# eval_steps = 8 #for debugging
learning_rate = 2e-5
seed = 17

np.random.seed(seed)
torch.manual_seed(seed)


def read_samples(file_name):
    with open(os.path.join(hs_folder, file_name + ".in"), 'r') as f:
        train_source_lines = f.read().splitlines()

    with open(os.path.join(hs_folder, file_name + ".out"), 'r') as f:
        train_target_lines = f.read().splitlines()    

    return [{"source": s, "target": process_to_ast(t.replace("§", "\n").replace("    ", "\t").replace("\\ ", ""))} 
                for (s, t) in zip(train_source_lines, train_target_lines)]

def two_pass_preprocess(file_name):
    ds_list = read_samples(file_name)
    for x in ds_list:
        msg = grammar_collector.build_message(x["target"], [])
        x["target"] = "".join(msg)
    return ds_list

train_set = Dataset.from_list(two_pass_preprocess(train_file_name))
dev_set = Dataset.from_list(two_pass_preprocess(dev_file_name))
test_set = Dataset.from_list(two_pass_preprocess(test_file_name))

#First we experiment without any code preprocessing 

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens':[*list(name_symbols), *list(grammar_collector.symbols.keys())]})
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

# model = AutoModelForCausalLM.from_pretrained(checkpoint, n_ctx = max_length, max_length = max_length)
# model.resize_token_embeddings(len(tokenizer))
# model.to("cuda")

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

nend_id = tokenizer(NEND).input_ids[0]
lst_id = tokenizer(LST).input_ids[0]
symbol_id_map = {symbol:tokenizer(symbol).input_ids[0] for symbol in grammar_collector.symbols.keys()}
id_symbol_map = {v:k for k,v in symbol_id_map.items()}

torch.set_printoptions(edgeitems=100) #for debugging only

#https://docs.python.org/3/library/ast.html
class PythonGrammarGPT2(torch.nn.Module):
    def __init__(self):
        super(PythonGrammarGPT2, self).__init__()
        self.transformer = GPT2LMHeadModel.from_pretrained(checkpoint) #TODO: pass config as in normal NN 
        self.transformer.resize_token_embeddings(len(tokenizer))
        self.transformer.to("cuda")
        # self.softmax = torch.nn.Softmax(dim=-1) #cannot learn if put last
        # self.max_possible_const_size = 10
        # self.length_proba = 0.95 #with each new token the logit will be decreased by this value
        self.depth_scaler = 0.9 #depth penalty scaled from 1 (deepest error) to depth_scaler (shallow error)
        self.depth_max_penalty = 10
        self.enable_logging = False
        #logits batch_size x sentence_length x size of vocab (logits)        

    def _decode_constant_arg(self, grammar_mask, sample_tensor, depths, labels, attr: SymbolAttr, parent: Symbol, token_id, depth, mistake_made):
        # now we need to detect a chunk of labels that NN dedicated to this constant. 
        # we do this by taking argmax of NEND logits for next k tokens, k is hyper parameter, weighted by distance from start 
        #NOTE: next 1 means that at least 1 token should be read
        if token_id >= sample_tensor.size(0):
            return sample_tensor.size(0)
        if parent.type == ast.Constant:
            #first token in a chunk should be type 
            label_ids = [ symbol_id_map[label] for label in grammar_collector.non_ast_types.keys() ]
            # logits_filter = torch.zeros_like(sample_tensor[token_id, :]) #token id is position of [type] token         
            logits_filter = grammar_mask[token_id, :]
            logits_filter[:] = 0
            logits_filter[label_ids] = 1
            symbol_tensor = sample_tensor[token_id, :] * logits_filter
            depths[token_id] = depth
            next_token_id = token_id + 1
            if mistake_made:
                labels[token_id] = -100

            #NEXT code is for debugging
            if self.enable_logging:
                prediction = torch.argmax(symbol_tensor).item()
                symbol_name = id_symbol_map[prediction]
                padding = "\t" * depth
                print(f"{padding}[{token_id}] --> {symbol_name}")                  
        else:
            next_token_id = token_id

        while next_token_id < sample_tensor.size(0):

            logits_filter = grammar_mask[next_token_id, :]
            logits_filter[:] = 1
            label_ids = [ symbol_id_map[label] for label in grammar_collector.symbols.keys() ]
            label_ids.remove(nend_id)
            logits_filter[label_ids] = 0
            depths[next_token_id] = depth

            if mistake_made:
                labels[next_token_id] = -100            

            symbol_tensor = sample_tensor[next_token_id] * logits_filter + logits_filter
            # print("Masked p", masked_t)
            prediction = torch.argmax(symbol_tensor).item()
            # if prediction not in id_symbol_map:
            #     print(f"Cannot find {prediction} {tokenizer.decode(prediction)} in id_symbol_map", file = sys.stderr)
            # symbol = id_symbol_map[prediction]            
            if prediction == nend_id:
                if self.enable_logging:
                    padding = "\t" * depth
                    print(f"{padding}[{next_token_id}] --> [NEND]")                
                next_token_id += 1 
                break             
            next_token_id += 1 

        return next_token_id

    def _decode_list_arg(self, grammar_mask, sample_tensor, depths, labels, attr: SymbolAttr, token_id, depth, mistake_made):
        if token_id >= sample_tensor.size(0):
            return sample_tensor.size(0)        
        assert attr.is_seq and attr.group is not None, f"Cannot read sequence for {attr}"

        #first symbol have to be LST
        # logits_filter = torch.zeros_like(sample_tensor[token_id, :])
        logits_filter = grammar_mask[token_id, :]
        logits_filter[:] = 0
        logits_filter[lst_id] = 1
        # sample_tensor[token_id, :] *= logits_filter
        depths[token_id] = depth

        if mistake_made: #ignore new errors because mistake was alreeady made at root node
            labels[token_id] = -100

        if self.enable_logging:
            padding = "\t" * depth
            print(f"{padding}[{token_id}] --> [LST]")

        next_token_id = token_id + 1
        one_attr = SymbolAttr("", is_seq=False, has_values=True, group = attr.group)
        #NOTE: we do not know how long list should be
        # at one moment we can check current logits for next_token_id and if it is probable to have NEND, we can terminate loop
        # we need to compare logits for nend (decision to terminate) with logits of any other symbol probability. from group attr.group
        while next_token_id < sample_tensor.size(0):

            # logits_filter = torch.zeros_like(sample_tensor[token_id, :])      

            logits_filter = grammar_mask[next_token_id, :]
            logits_filter[:] = 0
            possible_labels = grammar_collector.groups[attr.group]
            label_ids = [ symbol_id_map[label] for label in possible_labels ]
            label_ids.append(nend_id)
            logits_filter[label_ids] = 1
            depths[next_token_id] = depth

            
            # mask = torch.zeros_like(sample_tensor[next_token_id, :])
            # mask[label_ids] = 1
            symbol_tensor = sample_tensor[next_token_id] * logits_filter + logits_filter
            # print("Masked p", masked_t)
            prediction = torch.argmax(symbol_tensor).item()
            symbol_name = id_symbol_map[prediction]            
            if symbol_name == NEND:
                #enforce NEND and break 

                # # logits_filter = torch.zeros_like(sample_tensor[next_token_id, :])
                # logits_filter = grammar_mask[next_token_id, :]
                # logits_filter[:] = 0
                # logits_filter[nend_id] = 1
                # sample_tensor[next_token_id, :] *= logits_filter
                # depths[next_token_id] = depth
                # next_token_id += 1 
                if mistake_made:
                    labels[next_token_id] = -100
                # elif prediction != labels[next_token_id]: #wrong length of list
                #     mistake_made = True 
                
                if self.enable_logging:
                    padding = "\t" * depth
                    print(f"{padding}[{next_token_id}] --> [NEND]")                
                next_token_id += 1 
                break 
            
            # next_token_id = self._decode_symbol_arg(grammar_mask, sample_tensor, depths, one_attr, next_token_id, depth)            
            symbol = grammar_collector.symbols[symbol_name]
            next_token_id += 1 
            for a in symbol.attrs:
                if not a.has_values: #note that we ignore this assuming that input follows the trained schema
                    continue #tensor does not have logits for this attr
                elif (not a.is_seq) and a.group is None:
                    next_token_id = self._decode_constant_arg(grammar_mask, sample_tensor, depths, labels, a, symbol, next_token_id, depth + 1, mistake_made)
                elif not a.is_seq:
                    next_token_id = self._decode_symbol_arg(grammar_mask, sample_tensor, depths, labels, a, next_token_id, depth + 1, mistake_made) 
                else: #list 
                    next_token_id = self._decode_list_arg(grammar_mask, sample_tensor, depths, labels, a, next_token_id, depth + 1, mistake_made)

        return next_token_id

    def _decode_symbol_arg(self, grammar_mask, sample_tensor, depths, 
            labels, attr: SymbolAttr, token_id, depth, mistake_made):
        if token_id >= sample_tensor.size(0): 
            return sample_tensor.size(0) # we already set all logits ilter
        assert (not attr.is_seq) and attr.group is not None, f"Cannot generate symbol for attrs {attr}"
        assert attr.group in grammar_collector.groups, f"Symbol group was not found in groups for {attr}"

        #NOTE: here we let NN to pick symbol from grammar
        # logits_filter = torch.zeros_like(sample_tensor[token_id, :])              
        logits_filter = grammar_mask[token_id, :]
        logits_filter[:] = 0
        possible_labels = grammar_collector.groups[attr.group]
        label_ids = [ symbol_id_map[label] for label in possible_labels ]
        logits_filter[label_ids] = 1
        symbol_tensor = sample_tensor[token_id] * logits_filter + logits_filter
        depths[token_id] = depth
        # print(sample_tensor[token_id, :])
        prediction = torch.argmax(symbol_tensor).item()
        if mistake_made: #if mistake made before in ast - do not try correct errors after
            labels[token_id] = -100 
        elif prediction != labels[token_id]: #we made first mistake at ast node 
            mistake_made = True
            
        symbol_name = id_symbol_map[prediction]

        if self.enable_logging:
            padding = "\t" * depth
            print(f"{padding}[{token_id}] --> {symbol_name}")

        symbol = grammar_collector.symbols[symbol_name]
        next_token_id = token_id + 1
        for a in symbol.attrs:
            if not a.has_values: #note that we ignore this assuming that input follows the trained schema
                continue #tensor does not have logits for this attr
            elif (not a.is_seq) and a.group is None:
                next_token_id = self._decode_constant_arg(grammar_mask, sample_tensor, depths, labels, a, symbol, next_token_id, depth + 1, mistake_made)
            elif not a.is_seq:
                next_token_id = self._decode_symbol_arg(grammar_mask, sample_tensor, depths, labels, a, next_token_id, depth + 1, mistake_made) 
            else: #list 
                next_token_id = self._decode_list_arg(grammar_mask, sample_tensor, depths, labels, a, next_token_id, depth + 1, mistake_made)
        return next_token_id

    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None
    ):
        # print("Keys:", list(kwargs.keys()), file = sys.stderr)
        gpt2_result = self.transformer(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
        # attrs = [start_symbol for _ in range(gpt2_result.logits.size(0))]
        attrs = start_symbol
        #NOTE: next line requires much memory: batch_size * sentence_size * vocab_size
        # logits_filter = torch.zeros_like(gpt2_result.logits, device = "cpu") # we prepare tensor on cpu and then send it to gpu

        #During traversal we collect here depths of each label according to parsed tree
        #we use them for loss penalty later

        # print("Enforcing grammar...")

        scores = torch.nn.functional.softmax(gpt2_result.logits, dim=-1)

        depths = torch.ones((gpt2_result.logits.size(0), gpt2_result.logits.size(1)), device = "cpu")
        useful_labels = torch.clone(labels) if labels is not None else torch.full((gpt2_result.logits.size(0), gpt2_result.logits.size(1)), -100)
        grammar_mask = torch.ones_like(gpt2_result.logits)
        for sample_id in range(gpt2_result.logits.size(0)):
            #NOTE: each sample has its own grammar flow. Cannot be parallelized 
            # print(f"Batch {sample_id}")
            # self.enable_logging = sample_id == 0                
            token_id = (labels[sample_id] != -100).nonzero()[0].item() - 1
            # print("First token is ", token_id)
            self._decode_symbol_arg(grammar_mask[sample_id, :-1, :], scores[sample_id, :-1], depths[sample_id, :-1], 
                                        useful_labels[sample_id, 1:], attrs, token_id, 1, False) #updates logits corresponding to grammar
            # self.enable_logging = False
            # print()

        grammar_logits = gpt2_result.logits * grammar_mask

        print("Labels", useful_labels)

        # predictions = torch.argmax(grammar_logits, dim=-1).cpu()
        # label_ids = labels.cpu()
        # for i, (slabels, sample) in enumerate(zip(label_ids, predictions)):
        #     print("P:")
        #     for j, (l, p) in enumerate(zip(slabels, sample)):
        #         lt = tokenizer.decode(l) if l != -100 else None
        #         print(f"\t{lt} {l}    {tokenizer.decode(p)} {p} {grammar_logits[i, j, p]}  {grammar_mask[i, j, p]}  {logits[i, j, p]}")            
        #     print()
        #     exit(1)

        # print("Enforcing grammar done...")

        # print("Depthes", depths)
        if labels is not None:
            # we need to reecompute loss now, because we modified logits
            # #NOTE: we weight loss - if misake was closer to the root node it has bigger rippler effect - so we panish root errors more
            # max_depthes = torch.max(depths, dim = -1).values.reshape(depths.size(0), 1)
            # depthes_diffs = max_depthes - depths
            # max_depthes_diffs = torch.max(depthes_diffs, dim = -1).values
            # max_depthes_diffs[max_depthes_diffs == 0] = 1
            # depthes_diffs_w = (depthes_diffs / (max_depthes_diffs.reshape(depthes_diffs.size(0), 1))) * self.depth_penalty_scaler

            # print("Depth weights", depthes_diffs_w)
            # if "labels" in kwargs:
                # Shift so that tokens < n predict n
                # labels = kwargs["labels"]
            shift_logits = grammar_logits[..., :-1, :].contiguous()
            shift_labels = useful_labels[..., 1:].contiguous()
            # shift_depth = depthes_diffs_w[..., :-1]
            # predictions = torch.argmax(shift_logits, dim=-1)
            # misses = (shift_labels != -100).float()
            # misses *= (predictions != shift_labels).float()

            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # loss_view = loss.view(shift_logits.size(0), shift_logits.size(1))
            # w = torch.ones_like(loss_view)

            # w += (shift_depth.to(w.device) * misses)

            # loss_view *= w
            # loss_per_sample = loss_view.mean(axis=1)    
            # weighted_loss = loss_per_sample.mean()        
        return CausalLMOutputWithCrossAttentions(
            loss = loss, #weighted_loss,
            logits = grammar_logits,
            past_key_values = gpt2_result.past_key_values,
            hidden_states = gpt2_result.hidden_states,
            attentions = gpt2_result.attentions,
            cross_attentions = gpt2_result.cross_attentions
        ) 

# t1 (10)  t2 t3 

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
eos_id = tokenizer.eos_token_id
def custom_data_collator(*args):
    ''' we do not need to deduce preefix parts - change all labels till first -100 to -100 '''
    res = data_collator(*args)
    for l in res['labels']:
        i = 0
        while i < len(l) and l[i] != -100:
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
    seed = seed, label_names = ["labels"]
)

model = PythonGrammarGPT2()
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    compute_metrics = compute_metrics,
    data_collator=custom_data_collator,
    train_dataset=p_train_set,
    eval_dataset=p_dev_set,
)

trainer.train(ignore_keys_for_eval = ["past_key_values", "hidden_states", "attentions", "cross_attentions"])

trainer.save_model(result_path)