""" h9, for each grammar class use separate FNN """
import sys
from typing import Optional
from transformers import AutoTokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from datasets import Dataset, DatasetDict, load_dataset
import evaluate
import numpy as np
import os
import ast
import torch 
import astunparse
from grammar import LST, NEND, GrammarCollector, Symbol, SymbolAttr, start_symbol
from torch.nn.utils.rnn import pad_sequence

os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
    def add_to_vocab(self, name):
        if name not in self.mapping:
            new_name = self.mapping[name] = ("v" + str(self.i))
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
                new_name = self.class_mapping[node.name] = "c" + str(self.cls_i)
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
# name_symbols = set()
def process_to_ast(line):
    tree = ast.parse(line)    
    name_remover.reset()
    name_remover.visit(tree)
    grammar_collector.collect_metadata(tree) #after removing name
    # name_symbols.update(name_remover.rev_mapping.keys())
    # name_symbols.update(name_remover.rev_class_mapping.keys())
    return tree

import re
# symbol_pattern = r"\[(CLS\d+|v\d+)\]"
def unprocess(message: 'list[str]'):
    # m = [re.sub(symbol_pattern, r"\1", x) for x in message]
    code_module = grammar_collector.unparse(message, constructor = grammar_collector.build_module)
    code = astunparse.unparse(code_module).strip().replace("\n\n", "\n").replace("    ", "\t")   
    return code #we preserve name of symbols but remove []

ds_name = "dvitel/hearthstone"
out_dir = "out/h9"
result_path = "result/h9"
checkpoint = "distilgpt2"
max_length = 912
batch_size = 4
num_epochs = 200
eval_steps = 100
learning_rate = 2e-5
seed = 17

np.random.seed(seed)
torch.manual_seed(seed)

def normalize(line:str):
    return line.strip().replace("§", "\n").replace("    ", "\t").replace("\\ ", "").replace("\n\n", "\n")

def preprocess0(e):
    return {"source":e["source"], "target":[process_to_ast(normalize(x)) for x in e["target"]]}

ds = load_dataset(ds_name)
ds_dict = {k:[{**el, "target": process_to_ast(normalize(el["target"]))} for el in one_ds] for k, one_ds in ds.items()}

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

#we map symbols of grammar to existing tokens
def map_symbols(symbols, default_map = {}):
    res = {}
    for s in symbols:
        default_value = default_map.get(s, None)
        if type(default_value) == int: #token id 
            tokens = [default_value]
            found_tid = default_value if default_value not in res else None 
        else:
            s_value = default_value or (' ' + s[1:-1].lower())
            tokens = tokenizer(s_value)['input_ids']                 
            found_tid = next((tid for tid in tokens if tid not in res), None)
        if found_tid is None:
            conflicts = {tid: res[tid] for tid in tokens}
            raise Exception(f"Cannot find token for symbol: {s}. Conflicts: {conflicts}")
        res[found_tid] = (s, tokenizer.decode(found_tid))
    return res 
default_map = {'[Eq]':' ==', '[IsNot]':' unlike', '[Lt]': ' <', '[Gt]': ' >', LST:' <[', 
                '[GtE]':' >=', '[LtE]':' <=', NEND:' ``', '[USub]':' [-',
                '[NotEq]':' unequal', '[BinOp]':' bin', '[ListComp]':' ===', '[Arguments]':'args',
                '[NoneType]':' none', '[Assign]':' =', '[BoolOp]': ' bool', '[bool]': ' boolean',
                '[Sub]':' -', '[Add]': ' +'}
sorted_symbols = sorted(grammar_collector.symbols.keys(), key = lambda x: len(x))
symbols_map = map_symbols(sorted_symbols, default_map = default_map)

symbol_to_token_map = {v[0]: v[1] for k, v in symbols_map.items()}
token_to_symbol_map = {v[1]: v[0] for k, v in symbols_map.items()}
symbol_to_tid_map = {v[0]: k for k, v in symbols_map.items()}
tid_to_symbol_map = {k:v[0] for k, v in symbols_map.items()}

def preprocess1(e):
    target_message = grammar_collector.build_message(e["target"], [])
    return {"source":e["source"], 
            "target":"".join([symbol_to_token_map.get(w, ' ||' + w.strip()) for w in target_message]) }

ds01_dict = {k:Dataset.from_list([preprocess1(el) for el in one_ds]) for k, one_ds in ds_dict.items()}
ds01 = DatasetDict(ds01_dict)

# tokenizer.add_special_tokens({'additional_special_tokens':[*list(name_symbols), *list(grammar_collector.symbols.keys())]})
def preprocess(e):
    alt_bodies = []
    for s, t in zip(e["source"], e["target"]):
      tgt = t # t.replace("(", "[LPAR]").replace(")", "[RPAR]").replace("[RPAR] [LPAR]", "[RPAR][LPAR]")
      alt_bodies.append(s + tokenizer.eos_token + tgt)
    # print(alt_bodies)
    data = tokenizer(alt_bodies, padding = "max_length", truncation = True, max_length = max_length)  
    return data

ds1 = ds01.map(preprocess, batched = True, remove_columns = ["source", "target"])

nend_id = symbol_to_tid_map[NEND]
lst_id = symbol_to_tid_map[LST]

torch.set_printoptions(edgeitems=10) #for debugging only

#4891758117 - total params
#157306242
#81962496
class FFN(torch.nn.Module):
    
    def __init__(self, input_size, output_size):
        super(FFN, self).__init__()
        self.FC1 = torch.nn.Linear(in_features=input_size, out_features=100)
        self.relu1 = torch.nn.ReLU()
        self.FC2 = torch.nn.Linear(in_features=100, out_features=output_size)
        self.relu2 = torch.nn.ReLU()
        # self.FC3 = torch.nn.Linear(in_features=out_out_size, out_features=output_size)
        pass
    
    def forward(self, x):        
        out = self.relu1(self.FC1(x))
        out = self.relu2(self.FC2(out))
        # return self.FC3(out)
        return out

#https://docs.python.org/3/library/ast.html
class PythonGrammarGPT2(torch.nn.Module):
    def __init__(self):
        super(PythonGrammarGPT2, self).__init__()
        self.transformer = GPT2LMHeadModel.from_pretrained(checkpoint) #TODO: pass config as in normal NN 
        # self.transformer.resize_token_embeddings(len(tokenizer))
        input_size = len(tokenizer) #size of vocab
        #NOTE: output size == number of labels in the group
        self.ffns = {group:{'ffn': FFN(input_size, len(constructors) + 1), 'labels': [*constructors, NEND]}
                        for group, constructors in grammar_collector.groups.items()}

        if len(grammar_collector.non_ast_types) > 0: 
            #NOTE: interesting subject of research is typing systems in the NN 
            # currently we do not restrain type of vars - programs are well-formatted but not well-typed
            ffn = FFN(input_size, len(grammar_collector.non_ast_types))
            self.ffns['types'] = {'ffn': ffn, 'labels': list(grammar_collector.non_ast_types.keys())}

        self.max_label_num = len(tokenizer) #we still allow all labels to be used for literals and names
        for g, ffn_labels in self.ffns.items(): #note that auto registering with __setattr__ does not work
            self.add_module(g if type(g) == str else g.__name__, ffn_labels['ffn'])    
            ffn_labels['labels_map'] = {l: i for i, l in enumerate(ffn_labels["labels"])}
            # label_count = len(ffn_labels['labels'])
            # if self.max_label_num < label_count:
            #     self.max_label_num = label_count

    def _decode_constant_arg(self, ffn_logits, sample_logits, parent: Symbol, token_id):
        # now we need to detect a chunk of labels that NN dedicated to this constant. 
        # we do this by taking argmax of NEND logits for next k tokens, k is hyper parameter, weighted by distance from start 
        #NOTE: next 1 means that at least 1 token should be read
        if token_id >= sample_logits.size(0):
            return sample_logits.size(0)
        if parent.type == ast.Constant:

            ffn = self.ffns["types"] #pick FFN corresponding to current group
            logits = ffn['ffn'](sample_logits[token_id])
            ffn_logits.append(logits) #we use them later to cat and backpropagate

            prediction = torch.argmax(logits).item()
                
            symbol_name = ffn['labels'][prediction]
            #TODO: use symbol_name to limit logits in next loop

            # global_prediction = symbol_to_tid_map[symbol_name]
            # preds[token_id] = global_prediction

            token_id += 1


        while token_id < sample_logits.size(0):

            #NOTE: we take whole vocab as legit for current type 
            # but probably it make sence to limit it accorning to type - ignore for now
            # int - regex for numbers, bool - True, False, NoneType - nothing
            ffn_logits.append(sample_logits[token_id])
            prediction = torch.argmax(sample_logits[token_id]).item()
            # preds[token_id] = prediction

            token_id += 1 

            if prediction == nend_id:
                break             

        return token_id

    def _decode_list_arg(self, ffn_logits, sample_logits, attr: SymbolAttr, token_id):
        if token_id >= sample_logits.size(0):
            return sample_logits.size(0)        
        assert attr.is_seq and attr.group is not None, f"Cannot read sequence for {attr}"

        ffn_logits.append(torch.abs(sample_logits[token_id][lst_id:lst_id+1]))
        # preds[token_id] = lst_id
        
        token_id += 1

        #NOTE: we do not know how long list should be
        # at one moment we can check current logits for token_id and if it is probable to have NEND, we can terminate loop
        # we need to compare logits for nend (decision to terminate) with logits of any other symbol probability. from group attr.group
        while token_id < sample_logits.size(0):

            ffn = self.ffns[attr.group] #pick FFN corresponding to current group
            logits = ffn['ffn'](sample_logits[token_id])
            ffn_logits.append(logits) #we use them later to cat and backpropagate

            prediction = torch.argmax(logits).item()                
            symbol_name = ffn['labels'][prediction]
            # global_prediction = symbol_to_tid_map[symbol_name]
            # preds[token_id] = global_prediction
      
            token_id += 1 
            if symbol_name == NEND:                
                break 
            
            symbol = grammar_collector.symbols[symbol_name]            
            for a in symbol.attrs:
                if not a.has_values: #note that we ignore this assuming that input follows the trained schema
                    continue #tensor does not have logits for this attr
                elif (not a.is_seq) and a.group is None:
                    token_id = self._decode_constant_arg(ffn_logits, sample_logits, symbol, token_id)
                elif not a.is_seq:
                    token_id = self._decode_symbol_arg(ffn_logits, sample_logits, a, token_id) 
                else: #list 
                    token_id = self._decode_list_arg(ffn_logits, sample_logits, a, token_id)

        return token_id

    def _decode_symbol_arg(self, ffn_logits, sample_logits, attr: SymbolAttr, token_id):
        if token_id >= sample_logits.size(0): 
            return sample_logits.size(0) # we already set all logits ilter
        assert (not attr.is_seq) and attr.group is not None, f"Cannot generate symbol for attrs {attr}"
        assert attr.group in grammar_collector.groups, f"Symbol group was not found in groups for {attr}"

        ffn = self.ffns[attr.group] #pick FFN corresponding to current group
        logits = ffn['ffn'](sample_logits[token_id])
        logits_no_nend = logits[:-1]
        ffn_logits.append(logits_no_nend) #we use them later to cat and backpropagate

        prediction = torch.argmax(logits_no_nend).item() #exclude last NEND                    
        symbol_name = ffn['labels'][prediction] #tid_to_symbol_map[prediction]

        symbol = grammar_collector.symbols[symbol_name]
        token_id += 1
        for a in symbol.attrs:
            if not a.has_values: #note that we ignore this assuming that input follows the trained schema
                continue #tensor does not have logits for this attr
            elif (not a.is_seq) and a.group is None:
                token_id = self._decode_constant_arg(ffn_logits, sample_logits, symbol, token_id)
            elif not a.is_seq:
                token_id = self._decode_symbol_arg(ffn_logits, sample_logits, a, token_id) 
            else: #list 
                token_id = self._decode_list_arg(ffn_logits, sample_logits, a, token_id)
        return token_id

    def _label_constant_arg(self, local_labels, labels, parent: Symbol, token_id):
        if token_id >= labels.size(0) or labels[token_id].item() == -100: #ignore suffix
            return labels.size(0)
        if parent.type == ast.Constant:

            ffn = self.ffns["types"] #pick FFN corresponding to current group

            symbol_name = tid_to_symbol_map[labels[token_id].item()]
            assert symbol_name in ffn['labels_map'], f"Cannot find label symbol {symbol_name} in symbols of types: {ffn['labels_map']}"            
            local_label_id = ffn['labels_map'][symbol_name]
            local_labels[token_id] = local_label_id

            # logits = ffn['ffn'](sample_logits[token_id])
            # ffn_logits.append(logits) #we use them later to cat and backpropagate

            # prediction = torch.argmax(logits).item()
                
            # symbol_name = ffn['labels'][prediction]
            #TODO: use symbol_name to limit logits in next loop

            # global_prediction = symbol_to_tid_map[symbol_name]
            # preds[token_id] = global_prediction

            token_id += 1


        while token_id < labels.size(0) and labels[token_id].item() != -100:

            #NOTE: we take whole vocab as legit for current type 
            # but probably it make sence to limit it accorning to type - ignore for now
            # int - regex for numbers, bool - True, False, NoneType - nothing
            # ffn_logits.append(sample_logits[token_id])
            # prediction = torch.argmax(sample_logits[token_id]).item()
            prediction = labels[token_id].item()
            local_labels[token_id] = prediction
            # preds[token_id] = prediction

            token_id += 1 

            if prediction == nend_id:
                break             

        return token_id

    def _label_list_arg(self, local_labels, labels, attr: SymbolAttr, token_id):
        if token_id >= labels.size(0) or labels[token_id].item() == -100: #ignore suffix
            return labels.size(0)        
        assert attr.is_seq and attr.group is not None, f"Cannot read sequence for {attr}"
        assert labels[token_id].item() == lst_id, f"Should be start of list, but {labels[token_id].item()} at {token_id}. All labels: {labels}"

        # ffn = self.ffns[attr.group] #pick FFN corresponding to current group        
        # symbol_name = tid_to_symbol_map[lst_id]
        # local_label_id = ffn['labels_map'][LST] #TODO: better to add assert probably
        local_labels[token_id] = 0 #at start we have only 1 label with local id == 0

        # ffn_logits.append(torch.abs(sample_logits[token_id][lst_id:lst_id+1]))
        # preds[token_id] = lst_id
        
        token_id += 1

        #NOTE: we do not know how long list should be
        # at one moment we can check current logits for token_id and if it is probable to have NEND, we can terminate loop
        # we need to compare logits for nend (decision to terminate) with logits of any other symbol probability. from group attr.group
        while token_id < labels.size(0) and labels[token_id].item() != -100:

            ffn = self.ffns[attr.group] #pick FFN corresponding to current group
            symbol_name = tid_to_symbol_map[labels[token_id].item()]
            assert symbol_name in ffn['labels_map'], f"Cannot find label symbol {symbol_name} in symbols of {attr.group}: {ffn['labels_map']}"            
            local_label_id = ffn['labels_map'][symbol_name]
            local_labels[token_id] = local_label_id

            token_id += 1 
            if symbol_name == NEND:                
                break 
            
            symbol = grammar_collector.symbols[symbol_name]            
            for a in symbol.attrs:
                if not a.has_values: #note that we ignore this assuming that input follows the trained schema
                    continue #tensor does not have logits for this attr
                elif (not a.is_seq) and a.group is None:
                    token_id = self._label_constant_arg(local_labels, labels, symbol, token_id)
                elif not a.is_seq:
                    token_id = self._label_symbol_arg(local_labels, labels, a, token_id) 
                else: #list 
                    token_id = self._label_list_arg(local_labels, labels, a, token_id)

        return token_id

    def _label_symbol_arg(self, local_labels, labels, attr: SymbolAttr, token_id):
        if token_id >= labels.size(0) or labels[token_id].item() == -100: #ignore suffix
            return labels.size(0) # we already set all logits ilter
        assert (not attr.is_seq) and attr.group is not None, f"Cannot generate symbol for attrs {attr}"
        assert attr.group in grammar_collector.groups, f"Symbol group was not found in groups for {attr}"

        ffn = self.ffns[attr.group] #pick FFN corresponding to current group        
        symbol_name = tid_to_symbol_map[labels[token_id].item()]
        assert symbol_name in ffn['labels_map'], f"Cannot find label symbol {symbol_name} in symbols of {attr.group}: {ffn['labels_map']}"
        local_label_id = ffn['labels_map'][symbol_name]
        local_labels[token_id] = local_label_id

        symbol = grammar_collector.symbols[symbol_name]
        token_id += 1
        for a in symbol.attrs:
            if not a.has_values: #note that we ignore this assuming that input follows the trained schema
                continue #tensor does not have logits for this attr
            elif (not a.is_seq) and a.group is None:
                token_id = self._label_constant_arg(local_labels, labels, symbol, token_id)
            elif not a.is_seq:
                token_id = self._label_symbol_arg(local_labels, labels, a, token_id) 
            else: #list 
                token_id = self._label_list_arg(local_labels, labels, a, token_id)
        return token_id

    def _symbol_constant_arg(self, global_labels, labels, parent: Symbol, token_id):
        if token_id >= labels.shape[0] or labels[token_id] == -100: #ignore suffix
            return labels.shape[0]
        if parent.type == ast.Constant:

            ffn = self.ffns["types"] #pick FFN corresponding to current group

            # symbol_name = tid_to_symbol_map[labels[token_id]]
            assert labels[token_id] < len(ffn['labels']), f"Cannot find label {labels[token_id]} in symbols of types: {ffn['labels']}"            
            symbol_name = ffn['labels'][labels[token_id]]
            global_labels[token_id] = symbol_to_tid_map[symbol_name]

            token_id += 1


        while token_id < labels.shape[0] and labels[token_id] != -100:

            prediction = labels[token_id]
            global_labels[token_id] = prediction

            token_id += 1 

            if prediction == nend_id:
                break             

        return token_id

    def _symbol_list_arg(self, global_labels, labels, attr: SymbolAttr, token_id):
        if token_id >= labels.shape[0] or labels[token_id] == -100: #ignore suffix
            return labels.shape[0]
        assert attr.is_seq and attr.group is not None, f"Cannot read sequence for {attr}"
        assert labels[token_id] == 0, f"Should be start of list, but {labels[token_id]} at {token_id}. All labels: {labels}"

        # ffn = self.ffns[attr.group] #pick FFN corresponding to current group        
        # symbol_name = tid_to_symbol_map[lst_id]
        # local_label_id = ffn['labels_map'][LST] #TODO: better to add assert probably
        global_labels[token_id] = lst_id #at start we have only 1 label with local id == 0

        # ffn_logits.append(torch.abs(sample_logits[token_id][lst_id:lst_id+1]))
        # preds[token_id] = lst_id
        
        token_id += 1

        #NOTE: we do not know how long list should be
        # at one moment we can check current logits for token_id and if it is probable to have NEND, we can terminate loop
        # we need to compare logits for nend (decision to terminate) with logits of any other symbol probability. from group attr.group
        while token_id < labels.shape[0] and labels[token_id] != -100:

            ffn = self.ffns[attr.group] #pick FFN corresponding to current group
            # symbol_name = tid_to_symbol_map[labels[token_id]]
            assert labels[token_id] < len(ffn['labels']), f"Cannot find label {labels[token_id]} in symbols of {attr.group}: {ffn['labels']}"            
            symbol_name = ffn['labels'][labels[token_id]]
            global_labels[token_id] = symbol_to_tid_map[symbol_name]

            token_id += 1 
            if symbol_name == NEND:                
                break 
            
            symbol = grammar_collector.symbols[symbol_name]            
            for a in symbol.attrs:
                if not a.has_values: #note that we ignore this assuming that input follows the trained schema
                    continue #tensor does not have logits for this attr
                elif (not a.is_seq) and a.group is None:
                    token_id = self._symbol_constant_arg(global_labels, labels, symbol, token_id)
                elif not a.is_seq:
                    token_id = self._symbol_symbol_arg(global_labels, labels, a, token_id) 
                else: #list 
                    token_id = self._symbol_list_arg(global_labels, labels, a, token_id)

        return token_id

    def _symbol_symbol_arg(self, global_labels, labels, attr: SymbolAttr, token_id):
        if token_id >= labels.shape[0] or labels[token_id] == -100: #ignore suffix
            return labels.shape[0] # we already set all logits ilter
        assert (not attr.is_seq) and attr.group is not None, f"Cannot generate symbol for attrs {attr}"
        assert attr.group in grammar_collector.groups, f"Symbol group was not found in groups for {attr}"

        ffn = self.ffns[attr.group] #pick FFN corresponding to current group        
        # symbol_name = tid_to_symbol_map[labels[token_id]]
        assert labels[token_id] < len(ffn['labels']), f"Cannot find label {labels[token_id]} in symbols of {attr.group}: {ffn['labels']}"
        symbol_name = ffn['labels'][labels[token_id]]
        global_labels[token_id] = symbol_to_tid_map[symbol_name]

        symbol = grammar_collector.symbols[symbol_name]
        token_id += 1
        for a in symbol.attrs:
            if not a.has_values: #note that we ignore this assuming that input follows the trained schema
                continue #tensor does not have logits for this attr
            elif (not a.is_seq) and a.group is None:
                token_id = self._symbol_constant_arg(global_labels, labels, symbol, token_id)
            elif not a.is_seq:
                token_id = self._symbol_symbol_arg(global_labels, labels, a, token_id) 
            else: #list 
                token_id = self._symbol_list_arg(global_labels, labels, a, token_id)
        return token_id

    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None
    ):
        gpt2_result = self.transformer(input_ids = input_ids, attention_mask = attention_mask)        
        all_logits_list = []
        for sample_id in range(gpt2_result.logits.size(0)):
            token_id = (input_ids[sample_id] == tokenizer.eos_token_id).nonzero()[0].item() #position of separator between <s>_<t>
            ffn_logits = []
            for _ in range(token_id):            
                ffn_logits.append(torch.tensor([]))
            self._decode_symbol_arg(ffn_logits, gpt2_result.logits[sample_id], start_symbol, token_id) #updates logits corresponding to grammar
            padded_logits = []
            for logits in ffn_logits:
                num_to_pad = self.max_label_num - len(logits)
                if num_to_pad > 0:
                    logits = torch.nn.functional.pad(logits, (0, num_to_pad))
                padded_logits.append(logits)                
            sample_logits = torch.stack(padded_logits)
            positions_to_pad = max_length - sample_logits.size(0)
            sample_logits_padded = sample_logits
            if positions_to_pad > 0:
                sample_logits_padded = torch.nn.functional.pad(sample_logits, (0, 0, 0, positions_to_pad))
            all_logits_list.append(sample_logits_padded)
        all_logits = pad_sequence(all_logits_list, batch_first=True)
        # for x, y in zip(all_logits[0], gpt2_result.logits[0]):
        #     print("x:", x)
        #     print("y:", y)

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
            shift_logits = all_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # shift_mistakes = mistakes[..., :-1].contiguous()
            # shift_depth = depthes_diffs_w[..., :-1]
            # predictions = torch.argmax(shift_logits, dim=-1)
            # sh_l = shift_labels[0]
            # pd = predictions[0]
            # print("TR L", sh_l[sh_l != -100])
            # print("TR P", pd[sh_l != -100])

            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss() #reduction = "none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # loss_view = loss.view(shift_logits.size(s0), shift_logits.size(1))

            # loss_view *= shift_mistakes
            # loss_per_sample = loss_view.mean(axis=1)    
            # weighted_loss = loss_per_sample.mean()        
        return CausalLMOutputWithCrossAttentions(
            loss = loss,
            logits = all_logits,
            past_key_values = gpt2_result.past_key_values,
            hidden_states = gpt2_result.hidden_states,
            attentions = gpt2_result.attentions,
            cross_attentions = gpt2_result.cross_attentions
        ) 

# t1 (10)  t2 t3 

model = PythonGrammarGPT2()
model.to("cuda")
# model.to("cpu")

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
      decoded_labels = np.full_like(labels_view, -100)
      model._symbol_symbol_arg(decoded_labels, labels_view, start_symbol, 0)
      pred_view = preds[label_map]
      decoded_pred = np.full_like(pred_view, -100)
      model._symbol_symbol_arg(decoded_pred, pred_view, start_symbol, 0)
      l_text = tokenizer.decode(decoded_labels)
      p_text = tokenizer.decode(decoded_pred)
    #   p_text = unprocess([tokenizer.decode(x) for x in pred_view])
    #   l_text = unprocess([tokenizer.decode(x) for x in labels_view])
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
    labels = res['labels']
    local_labels = torch.full_like(labels, -100) #-100 - ignore label
    for sample_id in range(labels.size(0)):
        i = 0
        while i < labels[sample_id].size(0) and labels[sample_id, i] != -100:
            labels[sample_id, i] = -100 
            i += 1             
        token_id = i + 1 #position after separator <s>_<t>
        model._label_symbol_arg(local_labels[sample_id], labels[sample_id], start_symbol, token_id)

    res['labels'] = local_labels          

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
    # fp16=True, 
    no_cuda = True,
    load_best_model_at_end = True, 
    metric_for_best_model = "exact_match",    
    seed = seed, label_names = ["labels"]
    # hub_model_id = "h8", push_to_hub = True
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

trainer.train(ignore_keys_for_eval = ["past_key_values", "hidden_states", "attentions", "cross_attentions"])

output = trainer.predict(ds1["test"], ignore_keys = ["past_key_values", "hidden_states", "attentions", "cross_attentions"])
print(output.metrics) #test set metrics

# trainer.save_model(result_path)
# trainer.push_to_hub()