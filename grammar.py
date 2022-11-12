''' module collects elements of the grammar which are in use in training set '''
from collections import deque
from dataclasses import dataclass, field
import json
import os
import sys
import ast
from typing import Any, Optional, Union
import astunparse
import numpy as np

NEND = "[NEND]" #end of parameter/node
LST = "[LST]" #start of list 

# ZERO_EL = 0 
# ONE_EL = 1
# SEQ_EL = 2 

# ast.dump(ast.parse("class A(B): pass"))

hs_folder = sys.argv[2] if len(sys.argv) > 2 else "hearthstone"
train_file_name = "train_hs"
test_file_name = "test_hs"
dev_file_name = "dev_hs"

def read_samples(file_name):
    with open(os.path.join(hs_folder, file_name + ".in"), 'r') as f:
        train_source_lines = f.read().splitlines()

    with open(os.path.join(hs_folder, file_name + ".out"), 'r') as f:
        train_target_lines = f.read().splitlines()    

    return [{"source": s, "target": t.replace("§", "\n").replace("\\ ", "")} for (s, t) in zip(train_source_lines, train_target_lines)]

if __name__ == "__main__":
    train_set = read_samples(train_file_name)
    dev_set = read_samples(dev_file_name)
    test_set = read_samples(test_file_name)

    tasts = [ast.parse(s["target"]) for s in train_set]

# reduced ast 
# [ClassDef] [CLSN1] base* [NEND] body [FunctionDef] name* [NEND] 

# grammar_classes = {}
# def has_ast_base(cls: type):
#     if cls == object:
#         return False     
#     if cls.__base__ == ast.AST:
#         return True 
#     return has_ast_base(cls.__base__)
# for name in dir(ast):
#     cls = getattr(ast, name)
#     if type(cls) == type and has_ast_base(cls): 
#         symbol = 
#         cur = cls 
#         while cur.__base__ != ast.AST:
#             group = f"[{cur.__base__.__name__}]"
#             grammar_classes.setdefault(group, set()).add(symbol)
#             cur = cur.__base__        

def find_child_ast_clss(bcls):
    chlds = [cls for name in dir(ast) for cls in [getattr(ast, name)] if type(cls) == type and cls.__base__ == bcls]
    return chlds

# find_child_ast_clss(ast.stmt)    

@dataclass
class SymbolAttr:
    name: str 
    is_seq: bool
    has_values: bool
    group: Optional[type] = None #grammar type from ast.AST
    type: Optional[type] = None

@dataclass
class Symbol:
    name: str
    type: type
    group: Optional[type] = None # derived classes from ast.AST
    attrs: 'list[SymbolAttr]' = field(default_factory = list)

start_symbol = SymbolAttr("", is_seq = False, has_values = True, group = ast.mod)

class GrammarCollector():
    ''' traverse training sample and collects used nodes from grammar '''
    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self):
        # self.schema = {}
        self.LST = Symbol(LST, LST)
        self.NEND = Symbol(NEND, NEND)        
        self.groups = {}
        self.symbols = {LST: self.LST, NEND: self.NEND}
    
    def get_name(self, v): 
        return f"[{v.__name__ if type(v) == type else str(v)}]"

    def get_group(self, v) -> Optional[type]:
        if not isinstance(v, ast.AST):
            return None
        cur = v.__class__
        g = cur.__base__ if cur.__base__ != ast.AST else cur
        return g

    def collect_metadata(self, node):
        if isinstance(node, ast.AST):
            symbol_name = self.get_name(node.__class__)

            if symbol_name not in self.symbols:
                self.symbols[symbol_name] = Symbol(symbol_name, node.__class__)
            symbol = self.symbols[symbol_name]

            symbol.group = self.get_group(node)
            self.groups.setdefault(symbol.group, set()).add(symbol_name)

            for attr_pos, (attr_name, attr_val) in enumerate(ast.iter_fields(node)):
                if attr_pos >= len(symbol.attrs):
                    symbol.attrs.append(SymbolAttr(attr_name, False, False))
                attr = symbol.attrs[attr_pos]
                if type(attr_val) == list:
                    attr.is_seq = True
                    attr.has_values = attr.has_values or (len(attr_val) > 0)
                    if attr.group is None and len(attr_val) > 0:
                        attr.group = self.get_group(attr_val[0])
                elif attr_val is not None:
                    attr.has_values = True
                    attr.group = self.get_group(attr_val)
                    if attr.group is None and not isinstance(node, ast.Constant):
                        attr.type = type(attr_val)
                self.collect_metadata(attr_val)
        elif type(node) == list:
            for n in node:
                self.collect_metadata(n)
        elif node is not None:
            node_type = type(node)
            literal_type = self.get_name(node_type)
            self.symbols[literal_type] = Symbol(literal_type, node_type)
    
    def build_message(self, node, message, qattr: SymbolAttr = start_symbol):
        if node is None:
            assert not qattr.is_seq, f"None values and empty lists could only be part of ? schema"
            message.append(NEND)
        elif isinstance(node, ast.AST):
            symbol_name = self.get_name(node.__class__)
            symbol = self.symbols.get(symbol_name, " ")
            assert not qattr.is_seq, f"AST requires arity of one"            
            assert symbol.group == qattr.group, f"Symbol {symbol_name} of group {qattr.group} was requested but current is of group {symbol.group}"
            message.append(symbol_name)
            for attr in symbol.attrs:
                attr_val = getattr(node, attr.name)
                if not attr.has_values: #only zeroes in schema - ignore
                    assert attr_val is None or attr_val == [], f"Symbol {symbol_name} has attr {attr.name} which should be of arity 0 but given {attr_val}"
                    continue
                self.build_message(attr_val, message, attr)
        elif type(node) == list:
            assert qattr.is_seq, f"Rendered node {node} should have list arity, but {qattr} is given. Last node: {message[-1] if len(message) > 0 else None}"
            message.append(LST)
            ch_attr = SymbolAttr("", is_seq=False, has_values=True, group = qattr.group)
            for n in node:
                self.build_message(n, message, ch_attr)
            message.append(NEND)
        else:
            assert not qattr.is_seq, f"Value {node} should have arity of one but {qattr} was given. Last node: {message[-1] if len(message) > 0 else None}"
            if len(message) > 0 and message[-1] == "[Constant]":
                node_type = type(node)            
                literal_type = self.get_name(node_type)
                assert literal_type in self.symbols, f"Unknown literal type {literal_type}"
            else:
                literal_type = None
            if literal_type:
                message.append(literal_type)
            message.append(str(node))
            message.append(NEND)
        return message

    def _decode_constant_arg(self, attr: SymbolAttr, mq: deque):
        assert attr.group is None and (not attr.is_seq), f"Node without group but with non-1 arity {attr}"
        #we read till we bump into [NEND]
        chunk = []
        while (cur := mq.popleft()) != NEND:
            chunk.append(cur)
        if len(chunk) == 0: #None
            return None
        elif chunk[0] in self.symbols: #only case for [int], [str] .. and other Constant node funcs
            if chunk[0] == "[bool]":
                return "".join(chunk[1:]) != "False"
            attr_val = self.symbols[chunk[0]].type("".join(chunk[1:]))
            return attr_val
        else:
            attr_val = "".join(chunk)
            if attr.type is not None:
                attr_val = attr.type(attr_val)
            return attr_val

    def _decode_symbol_arg(self, attr: SymbolAttr, mq:deque, constructor = lambda x: x):
        assert (not attr.is_seq) and attr.group is not None, f"Cannot read symbol when requested attr does not have group {attr}"
        symbol_name = mq.popleft()
        assert symbol_name in self.symbols, f"Symbol {symbol_name} is not in symbols set"
        symbol = self.symbols[symbol_name]
        assert attr.group == symbol.group, f"Unexpected symbol group {symbol.group}. Correct {attr.group}"
        attr_vals = {}
        for a in symbol.attrs:
            if not a.has_values: #note that we ignore this assuming that input follows the trained schema
                attr_vals[a.name] = [] if a.is_seq else None 
            elif (not a.is_seq) and a.group is None:
                attr_val = self._decode_constant_arg(a, mq)
                attr_vals[a.name] = attr_val
            elif not a.is_seq:
                attr_symbol, attr_args = self._decode_symbol_arg(a, mq, constructor = constructor) 
                attr_vals[a.name] = constructor((attr_symbol, attr_args))
            else: #list 
                attr_val_list = self._decode_list_arg(a, mq, constructor = constructor)
                attr_vals[a.name] = attr_val_list
        return (symbol, attr_vals)

    def _decode_list_arg(self, attr: SymbolAttr, mq:deque, constructor = lambda x: x):
        should_be_lst = mq.popleft() if len(mq) > 0 else None #we ignore LST symbol
        assert attr.is_seq and attr.group is not None and should_be_lst == LST, f"Cannot read sequence for {attr} and first symbol {should_be_lst}"
        # we start reading mq till moment when we bump into umatched NEND or eof - this will end the list 
        res = []

        one_attr = SymbolAttr("", is_seq=False, has_values=True, group = attr.group)
        while len(mq) > 0:
            if (cur := mq.popleft()) == NEND: 
                break 
            mq.appendleft(cur)            
            symbol, attrs = self._decode_symbol_arg(one_attr, mq, constructor = constructor)
            res.append(constructor((symbol, attrs)))

        return res

    def unparse(self, message: 'list[str]', constructor = lambda x: x):
        mq = deque(message)
        symbol, attrs = self._decode_symbol_arg(start_symbol, mq, constructor = constructor)
        return constructor((symbol, attrs))

    def build_module(self, x):
        ''' Use this as constructor for unparse '''
        return x[0].type(**x[1])

    def _generate_constant_arg(self, attr: SymbolAttr, parent: Symbol, names = {}):
        assert attr.group is None and (not attr.is_seq), f"Cannot generate const {attr}"
        if not attr.has_values:
            return [] if attr.is_seq else None
        elif parent.type == ast.Constant:
            # we pick type - source of runtime error 
            #TODO: this also should be in config
            types = { int: lambda: np.random.randint(-100, 100), float: lambda: np.random.random(), bool: lambda: np.random.random() > 0.5, str: lambda: np.random.choice(['my', 'vocab']) }
            selected_type = np.random.choice(list(types.keys()))
            attr_val = types[selected_type]()
            return attr_val
        else:
            types2 = { int: lambda: 0, bool: lambda: False, str: lambda: "[NAME]"  }
            attr_val = "[LIT]"
            if attr.type in types2:
                attr_val = types2[attr.type]()
            return attr_val        

    def _generate_list_arg(self, attr: SymbolAttr, names = {}, constructor = lambda x: x):
        assert attr.is_seq and attr.group is not None, f"Cannot generate sequence for {attr}"
        # we start reading mq till moment when we bump into umatched NEND or eof - this will end the list 
        res = []

        one_attr = SymbolAttr("", is_seq=False, has_values=True, group = attr.group)
        n = np.random.binomial(5, 0.2) #TODO: note that these params should correspond to statistics collected from training or different for each node type
        for _ in range(n):
            symbol, attrs = self._generate_symbol_arg(one_attr, names, constructor = constructor)
            res.append(constructor((symbol, attrs)))

        return res

    def _generate_symbol_arg(self, attr: SymbolAttr, names = {}, constructor = lambda x: x):
        assert (not attr.is_seq) and attr.group is not None, f"Cannot generate symbol for attr {attr}"
        assert attr.group in self.groups, f"Symbol group was not found in groups for {attr}"
        symbol_name = np.random.choice(list(self.groups[attr.group])) #pick random symbol TODO: rand generator as arg
        symbol = self.symbols[symbol_name]
        attr_vals = {}
        for a in symbol.attrs:
            if not a.has_values: #note that we ignore this assuming that input follows the trained schema
                attr_vals[a.name] = [] if a.is_seq else None 
            elif (not a.is_seq) and a.group is None:
                attr_val = self._generate_constant_arg(a, symbol, names)
                attr_vals[a.name] = attr_val
            elif not a.is_seq:
                attr_symbol, attr_args = self._generate_symbol_arg(a, names, constructor = constructor) 
                attr_vals[a.name] = constructor((attr_symbol, attr_args))
            else: #list 
                attr_val_list = self._generate_list_arg(a, names, constructor = constructor)
                attr_vals[a.name] = attr_val_list
        return (symbol, attr_vals)

    def generate_program(self, start_symbol = start_symbol, constructor = lambda x: x):
        ''' Synthesize random program based on schema extracted during collect_metadata'''
        symbol, args = self._generate_symbol_arg(start_symbol, {}, constructor=constructor)
        return constructor((symbol, args))


# ast.dump(ast.parse("'test'"))


# v.collect_metadata(tasts[0])
# v.build_message(tasts[0])
# v.messages
# v.list_lengths

# test1 = ast.parse("def f(): return 5")

# v.collect_metadata(test1)
# m = v.build_message(test1, [])
# r = v.unparse(messages[147], constructor = v.build_module)
# messages[147]
# v.symbols


def gen_test():
    v = GrammarCollector()
    for t in tasts:
        v.collect_metadata(t)    
    r = v.generate_program(start_symbol, constructor=v.build_module)        
    print(astunparse.unparse(r))
    #NOTES on gen: better statistics collection is necessary in GrammarCollector 

def test0():
    v = GrammarCollector()
    for t in tasts:
        v.collect_metadata(t)    
    messages = [v.build_message(t, []) for t in tasts]
    print(messages[0])

def test1(): 
    v = GrammarCollector()
    for t in tasts:
        v.collect_metadata(t)    
    messages = [v.build_message(t, []) for t in tasts]
    for i, m in enumerate(messages):
        try:
            r = v.unparse(m, constructor=v.build_module)
            txt = astunparse.unparse(r)
            print(txt)
            print()
        except Exception as e:
            print(e)
            print()
            print(i)
            break

def test2():
    v = GrammarCollector()
    for t in tasts:
        v.collect_metadata(t)    
    messages = [v.build_message(t, []) for t in tasts]    
    for o, m in zip(tasts, messages):
        r = v.unparse(m, constructor=v.build_module)
        ot = ast.dump(o)
        rt = ast.dump(r)
        if ot != rt:
            print(ot)
            print(rt)
            print()