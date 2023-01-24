""" ge == grammar enforcement on top of gpt: no name stripping, no skip on mistake, no depth weighted loss"""
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime
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

CATEGORY_SYMBOL = 1 
CATEGORY_TYPE = 2
CATEGORY_LITERAL = 3
CATEGORY_META = 4
seed = 17 if len(sys.argv) < 3 else int(sys.argv[2]) 
os.environ["TOKENIZERS_PARALLELISM"] = "true"
grammar_enforcement_down_level_str = "1.0" if len(sys.argv) == 1 else sys.argv[1]
print(f"Starting Grammar Enforcement with down level {grammar_enforcement_down_level_str}")
grammar_enforcement_down_level_str_safe = grammar_enforcement_down_level_str.replace(".", "_")
grammar_enforcement_down_level = float(grammar_enforcement_down_level_str)
grammar_enforcement_up_level = 1.0
ds_name = "dvitel/hearthstone"
out_dir = f"out/ge-{seed}-{grammar_enforcement_down_level_str_safe}"
result_path = f"result/ge-{seed}-{grammar_enforcement_down_level_str_safe}"
checkpoint = "distilgpt2"
metric_file = "ge-metrics.csv"
max_length = 912
batch_size = 8
num_epochs = 200
eval_steps = 800
learning_rate = 4e-5
num_debug_tokens = 15
num_debug_eval_samples = 0 if len(sys.argv) < 4 else int(sys.argv[3])
logit_depth_penalty = 0.97 #each time we consider constructor with group alternative, we multiply its up level to accumulated depth_penalty
logit_length_penalty = 0.97 #used for literal synthesis

# torch.autograd.set_detect_anomaly(True)
grammar_collector = GrammarCollector()
# name_symbols = set()
def process_to_ast(line):
    tree = ast.parse(line)    
    grammar_collector.collect_metadata(tree) #after removing name
    return tree

import re
# symbol_pattern = r"\[(CLS\d+|v\d+)\]"
def unprocess(message: 'list[str]'):
    # m = [re.sub(symbol_pattern, r"\1", x) for x in message]
    code_module = grammar_collector.unparse(message, constructor = grammar_collector.build_module)
    code = astunparse.unparse(code_module).strip().replace("\n\n", "\n").replace("    ", "\t")   
    return code #we preserve name of symbols but remove []

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
literal_start_symbol = ' ||'
literal_start_id = tokenizer(literal_start_symbol)['input_ids'][0]

def preprocess1(e):
    target_message = grammar_collector.build_message(e["target"], [])
    return {"source":e["source"], 
            "target":"".join([symbol_to_token_map.get(w, literal_start_symbol + w.strip()) for w in target_message]) }

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


#NOTE: type values regexes with tokenizer
#int type -?(\d)+
unary_minus_id = tokenizer('-').input_ids[0]
int_digit_tids = [tokens.input_ids[0] for i in range(10) for tokens in [tokenizer(str(i))]]
def int_allowed_tids(prev_tokens, logits_filter):
    if len(prev_tokens) == 0:
        logits_filter[[unary_minus_id, *int_digit_tids]] = grammar_enforcement_up_level
    elif len(prev_tokens) > 1 or (prev_tokens[0] != unary_minus_id):
        logits_filter[int_digit_tids] = (logit_length_penalty ** len(prev_tokens)) * grammar_enforcement_up_level
        logits_filter[nend_id] = grammar_enforcement_up_level
    else:
        logits_filter[int_digit_tids] = grammar_enforcement_up_level
#bool type (True|False)
true_tid = tokenizer("True").input_ids[0] #checked - it is one token
false_tid = tokenizer("False").input_ids[0] #same
def bool_allowed_tids(prev_tokens, logits_filter):
    if len(prev_tokens) == 0:        
        logits_filter[[true_tid, false_tid]] = grammar_enforcement_up_level
    else:
        logits_filter[nend_id] = grammar_enforcement_up_level
#NoneType 
def none_allowed_tids(prev_tokens, logits_filter):
    logits_filter[nend_id] = grammar_enforcement_up_level
#str 
# def str_allowed_tids(prev_tokens, logits_filter):
#     logits_filter[:] = (logit_length_penalty ** len(prev_tokens)) * grammar_enforcement_up_level
#     logits_filter[nend_id] = grammar_enforcement_up_level

type_allowed_tids = {'[int]':int_allowed_tids, '[bool]':bool_allowed_tids,'[NoneType]':none_allowed_tids}

# def compute_avg_miss_pos(prediction_labels, shift_labels):
#     first_miss_idxs = []
#     for preds, labels in zip(prediction_labels, shift_labels):              
#         label_map = labels >= 0
#         labels_view = labels[label_map]
#         pred_view = preds[label_map]
#         miss_idxs = np.where(labels_view != pred_view)[0]
#         if len(miss_idxs) > 0:
#             first_miss_idxs.append(miss_idxs[0])
#     miss_idxs_in_sentence = np.mean(first_miss_idxs) if len(first_miss_idxs) > 0 else None
#     return {"miss_pos": miss_idxs_in_sentence}

def compute_correct_percent(prediction_labels, shift_labels, matches):
    correct_count = 0
    all_count = 0
    unparse_type_errors = 0
    unparse_value_errors = 0
    errs_to_print = 3
    for preds, labels, was_match in zip(prediction_labels, shift_labels, matches):              
        label_map = labels >= 0
        start_tid = np.where(label_map)[0][0]
        pred_view = preds[start_tid:]
        filtered_pred_view = [x for x in pred_view if x != literal_start_id]
        message = [tid_to_symbol_map.get(x, tokenizer.decode(x)) for x in filtered_pred_view]
        all_count += 1
        try: 
            p_text = unprocess(message)
            correct_count += 1
        except ValueError as e:
            print("Error in unprocess on match", e, file = sys.stderr)
            print(f"Msg len {len(message)}. token len {len(filtered_pred_view)}/{len(pred_view)} at {start_tid} ({max_length - start_tid} left)")
            print("MSG:", message, file = sys.stderr)
            unparse_value_errors += 1
        except TypeError as e:
            print("Error in unprocess on match", e, file = sys.stderr)
            print(f"Msg len {len(message)}. token len {len(filtered_pred_view)}/{len(pred_view)} at {start_tid} ({max_length - start_tid} left)")
            print("MSG:", message, file = sys.stderr)
            unparse_type_errors += 1            
        except Exception as e:
            if was_match and errs_to_print > 0:
                print("Error in unprocess on match", e, file = sys.stderr)
                print(f"Msg len {len(message)}. token len {len(filtered_pred_view)}/{len(pred_view)} at {start_tid} ({max_length - start_tid} left)")
                print("MSG:", message, file = sys.stderr)
                errs_to_print -= 1
    return {"correct_percent": correct_count / all_count , "unparse_type_errors_percent": unparse_type_errors / all_count, "unparse_value_errors_percent": unparse_value_errors / all_count}

# first_error_depths = []
#NOTE: these are global refs to tensors obtained from eval or test of the model. They are used in metrics to study types of errors
testset_depths = [] #elements are tensors - one per batch sample
testset_predictions = []
testset_categories = []
testset_labels = []
testset_programlen = []
testset_starts = []

def compute_error_stats():
    """ Note that this metric is only available after eval and will be reset on call """
    stats = defaultdict(list)
    stats['proglen'].extend(testset_programlen)
    for sample_labels, sample_predictions, sample_depths, sample_categories, sample_start in zip(testset_labels, testset_predictions, testset_depths, testset_categories, testset_starts):
        #NOTE: there are also tail misses - when predictions == -100 while labels is not - we do not count them here
        start_id = torch.where(sample_labels != -100)[0][0].item() #NOTE: there should be label which is not -100
        main_sample_labels = sample_labels[start_id:]
        main_sample_predictions = sample_predictions[start_id:]
        misses = torch.logical_and(main_sample_predictions != -100, main_sample_labels != main_sample_predictions)
        misses_idxs = torch.where(misses)[0]
        first_miss_idx = misses_idxs[0].item() if misses_idxs.numel() > 0 else None
        misses_count = torch.sum(misses)
        stats['total_miss'].append(misses_count)
        if (sample_predictions[-1] != -100).item():            
            stats['complete_miss_avg'].append(misses_count)        
            stats['incomplete_progcount'].append(1)            
        def get_symbol_category(tid):
            symbol_name = tid_to_symbol_map.get(tid, None)            
            if symbol_name is None:
                return None 
            symbol = grammar_collector.symbols[symbol_name]
            if symbol.group is None:
                return None     
            return symbol.group.__name__               
        tid_to_category = np.vectorize(get_symbol_category)
        label_categories = tid_to_category(main_sample_labels)
        prediction_categories = tid_to_category(main_sample_predictions)
        stats['group_miss'].append(np.sum(label_categories != prediction_categories))
        # label_categories = sum(main_sample_labels == token for token in token_to_symbol_map.keys())
        # prediction_categories = sum(main_sample_labels == token for token in token_to_symbol_map.keys())
        # literal_miss = torch.logical_and(bare_misses, sample_categories == CATEGORY_LITERAL)
        # symbol_miss = torch.logical_and(bare_misses, sample_categories == CATEGORY_SYMBOL)
        # meta_miss = torch.logical_and(bare_misses, sample_categories == CATEGORY_META)
        # type_miss = torch.logical_and(bare_misses, sample_categories == CATEGORY_TYPE)
        # misses = torch.logical_or(literal_miss, torch.logical_or(symbol_miss, torch.logical_or(meta_miss, type_miss)))        
        # stats['literal_miss'].append(torch.sum(literal_miss))
        # stats['symbol_miss'].append(torch.sum(symbol_miss))
        # stats['meta_miss'].append(torch.sum(meta_miss))
        # stats['type_miss'].append(torch.sum(type_miss))
        if first_miss_idx is not None:
            stats['first_miss_depth'].append(sample_depths[first_miss_idx])
            stats['first_miss_pos'].append(first_miss_idx - sample_start)
        # stats['literal_miss_depth'].extend(sample_depths[literal_miss]) 
        # stats['symbol_miss_depth'].extend(sample_depths[symbol_miss])         
        # stats['meta_miss_depth'].extend(sample_depths[meta_miss])
        # stats['type_miss_depth'].extend(sample_depths[type_miss])
        # miss_idxs = torch.where(literal_miss)[0]
        # if miss_idxs.numel() > 0:
        #     stats['first_literal_miss_pos'].append(miss_idxs[0].item() - sample_start)
        # miss_idxs = torch.where(symbol_miss)[0]
        # if miss_idxs.numel() > 0:
        #     stats['first_symbol_miss_pos'].append(miss_idxs[0].item() - sample_start)
        # miss_idxs = torch.where(meta_miss)[0]
        # if miss_idxs.numel() > 0:
        #     stats['first_meta_miss_pos'].append(miss_idxs[0].item() - sample_start)
        # miss_idxs = torch.where(type_miss)[0]
        # if miss_idxs.numel() > 0:
        #     stats['first_type_miss_pos'].append(miss_idxs[0].item() - sample_start)
    # avg_depth = None if len(first_error_depths) == 0 else np.mean(first_error_depths)
    res = {k:None if len(v) == 0 else np.sum(v) if k.endswith("_miss") or k.endswith("count") else np.mean(v) for k,v in stats.items()}
    res.setdefault("complete_miss_avg", 0)
    res.setdefault("incomplete_progcount", 0)
    res.setdefault("first_miss_depth", None)
    res.setdefault("first_miss_pos", None)
    testset_depths.clear()
    testset_predictions.clear()
    testset_categories.clear()
    testset_labels.clear()
    testset_programlen.clear()
    testset_starts.clear()
    return res #{"error_depth":avg_depth}


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
    matches = []
    for preds, labels in zip(prediction_labels, shift_labels):      
      label_map = labels >= 0
      labels_view = labels[label_map]
      pred_view = preds[label_map]
      l_text = tokenizer.decode(labels_view)
      p_text = tokenizer.decode(pred_view)
      predictions.append(p_text)
      references.append(l_text)
      matches.append(p_text == l_text)
      if p_text != l_text and first_not_matched > 0:      
        print("EV L", l_text)
        print("EV P", p_text) 
        print()
        first_not_matched -= 1
    accuracy_metric = exact_match.compute(predictions = predictions, references = references)   
    bleu_metric = bleu.compute(predictions = predictions, references = references)   
    codebleu_metric = codebleu.compute(predictions = predictions, references = references)  
    chrf_metric = chrF.compute(predictions = predictions, references = references)  
    # miss_pos_metric = compute_avg_miss_pos(prediction_labels, shift_labels)
    correct_percent_metric = compute_correct_percent(prediction_labels, shift_labels, matches)
    error_metrics = compute_error_stats()
    return {"exact_match": accuracy_metric["exact_match"], **error_metrics, **correct_percent_metric, 
                "bleu": bleu_metric["bleu"], **codebleu_metric, "chrf": chrf_metric['score']}

nend_id = symbol_to_tid_map[NEND]
lst_id = symbol_to_tid_map[LST]

torch.set_printoptions(edgeitems=100) #for debugging only

@dataclass 
class GELayerData: 
    sample_id: int
    transformer_positive_logits: 'torch.tensor' #3d cube of logits
    logits_filter: 'torch.tensor'
    labels: 'torch.tensor'
    depths: 'torch.tesnor' #annotation of each token with depth in the ast 
    categories: 'torch.tensor' #annotation of each token with grammar category
    predictions: 'Optional[torch.tensor]' #detected predictions in GELayer
    max_token_num: int = max_length
    cur_debug_tokens: Optional[int] = None    

def get_symbol_weights(depth, possible_symbols):
    return [(logit_depth_penalty ** depth 
                if any(a.group in grammar_collector.recursive_groups 
                        for a in grammar_collector.symbols[symbol_name].attrs) 
                else 1) * grammar_enforcement_up_level for symbol_name in possible_symbols ]
#https://docs.python.org/3/library/ast.html
class PythonGrammarGPT2(torch.nn.Module):
    def __init__(self):
        super(PythonGrammarGPT2, self).__init__()
        self.transformer = GPT2LMHeadModel.from_pretrained(checkpoint) #TODO: pass config as in normal NN 
        self.transformer.resize_token_embeddings(len(tokenizer))
        self.transformer.to("cuda")
        self.nontraining_sample_id = None 

    def pick_symbol_or_token(self, data: GELayerData, token_id: int, depth: int, category: CATEGORY_SYMBOL | CATEGORY_TYPE | CATEGORY_LITERAL | CATEGORY_META):
        """ finds symbol for token. If teacher-forced, returns symbol according to labels"""
        try:
            data.depths[token_id] = depth
            data.categories[token_id] = category        
            label = data.labels[token_id].item()    
            if data.predictions is not None:
                filtered_logits = data.transformer_positive_logits[token_id] * data.logits_filter[token_id] 
                prediction = torch.argmax(filtered_logits).item()
                data.predictions[token_id] = prediction
                label_logit = filtered_logits[label]
                prediction_logit = filtered_logits[prediction]
            else: 
                prediction = -100
                label_logit = prediction_logit = None             
            tid = label if (label != -100) and self.training else prediction
            symbol_name = tid_to_symbol_map.get(tid, None) #NOTE: teacher forcing - we only allow gold labels during training
            if data.cur_debug_tokens and data.cur_debug_tokens < num_debug_tokens:
                label_token = tokenizer.decode(label)                
                prediction_token = tokenizer.decode(prediction) if label != prediction else label_token
                print(f"\t[{token_id}] {label == prediction} l {label} '{label_token}' {label_logit}, p {prediction} '{prediction_token}' {prediction_logit} | {data.cur_debug_tokens}")
                data.cur_debug_tokens += 1
            return (symbol_name, tid) #NOTE: symbol_name could be None
        except KeyError as e: 
            print("Error, cannot find key", e, file = sys.stderr)
            print("Token id", token_id, file = sys.stderr)
            print("Labels at pos", data.labels[token_id:], file = sys.stderr)
            print("Pred at pos", prediction, file = sys.stderr)
            print("Keys: ", tid_to_symbol_map.keys(), file = sys.stderr)
            left_labels = data.labels[token_id:]
            left_nonemoty_labels = left_labels[left_labels != -100]
            errored_token = tokenizer.decode(data.labels[token_id])
            expected_symbol = token_to_symbol_map.get(errored_token, None)
            print("Tokenizer tokens: ", tokenizer.decode(left_nonemoty_labels), file = sys.stderr)
            print("Error token ", errored_token, " Expected symbol: ", expected_symbol, 
                    "Expected tid: ", symbol_to_tid_map.get(expected_symbol, None), file = sys.stderr)
            raise 

    def _decode_constant_arg(self, data: GELayerData, parent: Symbol, token_id, depth) -> int:
        """ Check that predicted tokens will build Literal in grammar """
        if token_id >= data.max_token_num:
            return data.max_token_num
        tids_setter = None
        if parent.type == ast.Constant:
            #first token in a chunk should be a type of literal
            label_ids = [ symbol_to_tid_map[label] for label in grammar_collector.non_ast_types.keys() ]
            logits_filter = data.logits_filter[token_id, :]
            logits_filter[:] = grammar_enforcement_down_level
            logits_filter[label_ids] = grammar_enforcement_up_level            
            literal_type_symbol_name, _ = self.pick_symbol_or_token(data, token_id, depth, CATEGORY_TYPE)     
            tids_setter = type_allowed_tids.get(literal_type_symbol_name, None)
            token_id += 1        
        #first symbol have to be literal start
        logits_filter = data.logits_filter[token_id, :]
        logits_filter[:] = grammar_enforcement_down_level
        logits_filter[literal_start_id] = grammar_enforcement_up_level
        self.pick_symbol_or_token(data, token_id, depth, CATEGORY_META)
        token_id += 1
        literal_tokens = []
        while token_id < data.max_token_num:
            logits_filter = data.logits_filter[token_id, :]
            if tids_setter is None:
                logits_filter[:] = (logit_length_penalty ** len(literal_tokens)) * grammar_enforcement_up_level
                logits_filter[nend_id] = grammar_enforcement_up_level
            else:
                logits_filter[:] = grammar_enforcement_down_level
                tids_setter(literal_tokens, logits_filter)
            symbol_name, tid = self.pick_symbol_or_token(data, token_id, depth, CATEGORY_LITERAL)
            token_id += 1 
            literal_tokens.append(tid)
            if symbol_name == NEND: #exit the constant synth                
                break             
        return token_id

    def _decode_list_arg(self, data: GELayerData, attr: SymbolAttr, token_id, depth) -> int:
        if token_id >= data.max_token_num:
            return data.max_token_num
        # assert attr.is_seq and attr.group is not None, f"Cannot read sequence for {attr}"
        #first symbol have to be LST
        logits_filter = data.logits_filter[token_id, :]
        logits_filter[:] = grammar_enforcement_down_level
        logits_filter[lst_id] = grammar_enforcement_up_level
        self.pick_symbol_or_token(data, token_id, depth, CATEGORY_META)
        token_id += 1
        siblings_count = 0
        while token_id < data.max_token_num:
            possible_labels = grammar_collector.groups[attr.group]
            label_ids = [ symbol_to_tid_map[label] for label in possible_labels ]
            label_ids.append(nend_id)
            symbol_weights = get_symbol_weights(depth, possible_labels)
            symbol_weights.append(grammar_enforcement_up_level)
            symbol_weights_tensor = torch.tensor(symbol_weights, dtype=data.logits_filter.dtype, device = data.logits_filter.device)  
            symbol_weights_tensor[:-1] *= (logit_length_penalty ** siblings_count)
            logits_filter = data.logits_filter[token_id, :]
            logits_filter[:] = grammar_enforcement_down_level
            logits_filter[label_ids] = symbol_weights_tensor
            symbol_name, _ = self.pick_symbol_or_token(data, token_id, depth, CATEGORY_SYMBOL)
            token_id += 1 
            siblings_count += 1
            if symbol_name == NEND: #enforce NEND and break                                 
                break 
            if symbol_name is not None: #we just skip this logit and continue 
                symbol = grammar_collector.symbols[symbol_name]
                for a in symbol.attrs:
                    if not a.has_values: #note that we ignore this assuming that input follows the trained schema
                        continue #tensor does not have logits for this attr
                    elif (not a.is_seq) and a.group is None:
                        token_id = self._decode_constant_arg(data, symbol, token_id, depth + 1)
                    elif not a.is_seq:
                        token_id = self._decode_symbol_arg(data, a, token_id, depth + 1) 
                    else: #list 
                        token_id = self._decode_list_arg(data, a, token_id, depth + 1)
        return token_id

    def _decode_symbol_arg(self, data: GELayerData, attr: SymbolAttr, token_id, depth) -> int:
        if token_id >= data.max_token_num: 
            return data.max_token_num # we already set all logits ilter
        # assert (not attr.is_seq) and attr.group is not None, f"Cannot generate symbol for attrs {attr}"
        # assert attr.group in grammar_collector.groups, f"Symbol group was not found in groups for {attr}"
        possible_labels = grammar_collector.groups[attr.group]
        label_ids = [ symbol_to_tid_map[label] for label in possible_labels ]
        symbol_weights = get_symbol_weights(depth, possible_labels)
        symbol_weights_tensor = torch.tensor(symbol_weights, dtype=data.logits_filter.dtype, device = data.logits_filter.device)        
        logits_filter = data.logits_filter[token_id, :]
        logits_filter[:] = grammar_enforcement_down_level
        logits_filter[label_ids] = symbol_weights_tensor                
        symbol_name, _ = self.pick_symbol_or_token(data, token_id, depth, CATEGORY_SYMBOL)
        token_id += 1
        if symbol_name is not None:
            symbol = grammar_collector.symbols[symbol_name]            
            for a in symbol.attrs:
                if not a.has_values: #note that we ignore this assuming that input follows the trained schema
                    continue #tensor does not have logits for this attr
                elif (not a.is_seq) and a.group is None:
                    token_id = self._decode_constant_arg(data, symbol, token_id, depth + 1)
                elif not a.is_seq:
                    token_id = self._decode_symbol_arg(data, a, token_id, depth + 1) 
                else: #list 
                    token_id = self._decode_list_arg(data, a, token_id, depth + 1)
            return token_id
        else:
            return self._decode_symbol_arg(data, attr, token_id, depth)

    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None
    ):
        transformer_result = self.transformer(input_ids = input_ids, attention_mask = attention_mask, labels = labels)        
        if self.training:
            self.nontraining_sample_id = None
        elif labels is not None and self.nontraining_sample_id is None:
            self.nontraining_sample_id = 0

        min_logit = torch.min(transformer_result.logits)

        positive_logits = transformer_result.logits - min_logit.item() + 0.01 #shift level        

        depths = torch.zeros((positive_logits.size(0), positive_logits.size(1)), device = "cpu", dtype = torch.int)
        categories = torch.zeros((positive_logits.size(0), positive_logits.size(1)), device = "cpu", dtype = torch.int)
        useful_labels = torch.clone(labels) if labels is not None else torch.full((positive_logits.size(0), positive_logits.size(1)), -100, device = positive_logits.device)
        predictions = None if self.training else torch.full_like(useful_labels, -100)
        grammar_mask = torch.full_like(positive_logits, grammar_enforcement_up_level)

        for sample_id in range(positive_logits.size(0)):
            #NOTE: each sample has its own grammar flow. Cannot be parallelized ??
            debug_mistakes = not self.training and (labels is not None) and (self.nontraining_sample_id < num_debug_eval_samples)
            cur_debug_tokens = None
            if debug_mistakes:
                print(f"Debugging sample {sample_id}/{self.nontraining_sample_id}:")
                cur_debug_tokens = 0

            data = GELayerData(sample_id, positive_logits[sample_id, :-1], grammar_mask[sample_id, :-1], 
                                labels = useful_labels[sample_id, 1:], depths = depths[sample_id, :-1], 
                                categories = categories[sample_id, :-1], predictions = None if predictions is None else predictions[sample_id, :-1], 
                                cur_debug_tokens=cur_debug_tokens, max_token_num = max_length - 1)  
            
            # non_empty_labels = (labels[sample_id] != -100).nonzero()
            # label_token_id = non_empty_labels[0].item() - 1 if non_empty_labels.numel() > 0 else 0 #TODO: should be not 0 but id of position after init sentence
            separators = (input_ids[sample_id] == tokenizer.eos_token_id).nonzero()
            start_token_id = separators[0].item() if separators.numel() > 0 else 0 #NOTE: 0 cannot be!!!! - Too long initial sentence 

            # print(f"First token is label> {label_token_id}, input> {start_token_id}")
            end_token_id = self._decode_symbol_arg(data, start_symbol, start_token_id, 1) #updates logits corresponding to grammar
            if predictions is not None and labels is not None: 
                testset_programlen.append(end_token_id - start_token_id)
                testset_starts.append(start_token_id)
            # if data.predictions is not None and labels is not None: #not training and not production - eval 

                # error_depths = data.depths[(data.labels != -100) and (data.predictions != data.labels)]
                # if error_depths.numel() > 0:
                #     first_error_depths.append(torch.min(error_depths))
            if debug_mistakes:
                self.nontraining_sample_id += 1
                sample_grammar_logits = positive_logits[sample_id] * grammar_mask[sample_id]
                sample_predictions = torch.argmax(sample_grammar_logits, dim=-1)[start_token_id:start_token_id + num_debug_tokens]
                sample_val = tokenizer.decode(sample_predictions)
                print("Decoded grammar logits:\n",sample_val)
                # print("Grammar logits:\n",sample_predictions)
                for i, sample_grammar_prediction in enumerate(sample_predictions):
                    gp = data.transformer_positive_logits[start_token_id + i]
                    gp1 = data.transformer_positive_logits[start_token_id + i] * data.logits_filter[start_token_id + i]
                    gm = data.logits_filter[start_token_id + i]
                    print(f"--> [{i}] Gmask: {gm[sample_grammar_prediction]},  G prediction: {sample_grammar_prediction}, G logit: {gp[sample_grammar_prediction]}/{gp1[sample_grammar_prediction]}")                    

        if predictions is not None and labels is not None: 
            cpu_predictions = predictions.to("cpu")
            cpu_labels = useful_labels.to("cpu")            
            for sample_id in range(positive_logits.size(0)):
                testset_depths.append(depths[sample_id, :-1])
                testset_categories.append(categories[sample_id, :-1])
                testset_predictions.append(cpu_predictions[sample_id, :-1])
                testset_labels.append(cpu_labels[sample_id, 1:])

        grammar_logits = positive_logits * grammar_mask

        if labels is not None:

            shift_logits = grammar_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss = loss,
            logits = grammar_logits,
            past_key_values = transformer_result.past_key_values,
            hidden_states = transformer_result.hidden_states,
            attentions = transformer_result.attentions,
            cross_attentions = transformer_result.cross_attentions
        ) 

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
    model = model,
    tokenizer = tokenizer,
    args = args,
    compute_metrics = compute_metrics,
    data_collator = custom_data_collator,
    train_dataset = ds1["train"],
    eval_dataset = ds1["validation"],
)

trainer.train(ignore_keys_for_eval = ["past_key_values", "hidden_states", "attentions", "cross_attentions"])

output = trainer.predict(ds1["test"], ignore_keys = ["past_key_values", "hidden_states", "attentions", "cross_attentions"])
print(output.metrics) #test set metrics

def save_testset_metrics(out_metrics):
    fieldnames = ["down_level", "test_exact_match", "test_correct_percent", "test_unparse_type_errors_percent", "test_proglen",
                    "test_total_miss", "test_complete_miss_avg", "test_incomplete_progcount",
                    "test_group_miss", "test_first_miss_depth", "test_first_miss_pos", 
                    "test_bleu", "test_chrf", "test_CodeBLEU", "seed", "timestamp"]
    filtered_metrics = {k:v for k, v in out_metrics.items() if k in fieldnames}
    filtered_metrics['timestamp'] = datetime.now()
    filtered_metrics['down_level'] = grammar_enforcement_down_level    
    filtered_metrics['seed'] = seed
    should_append_header = not os.path.exists(metric_file) #NOTE: not atomic but we do not care
    with open(metric_file, 'a', newline='') as metrics: 
        writer = csv.DictWriter(metrics, fieldnames)
        if should_append_header:
            writer.writeheader()
        writer.writerow(filtered_metrics)

save_testset_metrics(output.metrics)

# import csv
# from datetime import datetime
# metric_file = "test_metric_file"
# grammar_enforcement_down_level = 0.77
# test_metrics_dict = {'test_loss': 6.072345733642578, 'test_exact_match': 0.2727272727272727, 'test_miss_pos': 96.4375, 'test_correct_percent': 0.6212121212121212, 'test_error_depth': 2.6666667461395264, 'test_bleu': 0.7600833147465573, 'test_CodeBLEU': 0.7523784629454893, 'test_ngram_match_score': 0.6956593072629136, 'test_weighted_ngram_match_score': 0.6960693189444551, 'test_syntax_match_score': 0.9127673796791443, 'test_dataflow_match_score': 0.705017845895444, 'test_chrf': 81.52314050969377, 'test_runtime': 38.6851, 'test_samples_per_second': 1.706, 'test_steps_per_second': 0.439}
# save_testset_metrics(test_metrics_dict)

# trainer.save_model(result_path)
# trainer.push_to_hub()