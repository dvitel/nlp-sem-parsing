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
    symbol_name: str
    is_seq: bool
    has_values: bool
    group: Optional[type] = None #grammar type from ast.AST
    type: Optional[type] = None
    #defines Symbols that are used as in child fields from training set
    # Example: { [ClassDef]: {body: {[FuncDef]}}} - we do not consider any other stmts for body if class contained only function definitions
    possible_symbols: 'set[str]' = field(default_factory = set)

@dataclass
class Symbol:
    name: str
    type: type
    group: Optional[type] = None # derived classes from ast.AST
    attrs: 'list[SymbolAttr]' = field(default_factory = list)

start_symbol = SymbolAttr("", "", is_seq = False, has_values = True, group = ast.mod, possible_symbols = {'[Module]'})

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
        self.non_ast_types = {}
        self.symbols = {LST: self.LST, NEND: self.NEND}
        self.recursive_groups = {ast.stmt, ast.expr} #set of groups which could create recursive loops
    
    def get_name(self, v): 
        return f"[{v.__name__ if type(v) == type else str(v)}]"

    def get_group(self, v) -> Optional[type]:
        if not isinstance(v, ast.AST):
            return None
        cur = v.__class__
        g = cur.__base__ if cur.__base__ != ast.AST else cur
        return g

    def collect_metadata(self, node, parent_symbol = None, parent_symbol_attr = None):
        if isinstance(node, ast.AST):
            symbol_name = self.get_name(node.__class__)

            if parent_symbol_attr is not None: 
                parent_symbol_attr.possible_symbols.add(symbol_name)

            if symbol_name not in self.symbols:
                self.symbols[symbol_name] = Symbol(symbol_name, node.__class__)
            symbol = self.symbols[symbol_name]

            symbol.group = self.get_group(node)
            self.groups.setdefault(symbol.group, set()).add(symbol_name)

            # attr_children = self.children.setdefault(symbol_name, {})

            for attr_pos, (attr_name, attr_val) in enumerate(ast.iter_fields(node)):
                if attr_pos >= len(symbol.attrs):
                    symbol.attrs.append(SymbolAttr(attr_name, symbol_name, False, False))
                attr = symbol.attrs[attr_pos]
                # children = attr_children.setdefault(attr.name, {})
                if type(attr_val) == list:
                    attr.is_seq = True
                    attr.has_values = attr.has_values or (len(attr_val) > 0)
                    if attr.group is None and len(attr_val) > 0:
                        attr.group = self.get_group(attr_val[0])
                elif attr_val is not None or (symbol.type == ast.Constant and attr.name == "value"):
                    attr.has_values = True
                    attr.group = self.get_group(attr_val)
                    if attr.group is None and not isinstance(node, ast.Constant):
                        attr.type = type(attr_val)
                self.collect_metadata(attr_val, parent_symbol = symbol, parent_symbol_attr = attr)
        elif type(node) == list:
            for n in node:
                self.collect_metadata(n, parent_symbol_attr = parent_symbol_attr)
        elif node is not None or (parent_symbol is not None and parent_symbol.type == ast.Constant):
            node_type = type(node)
            literal_type = self.get_name(node_type)            
            self.symbols[literal_type] = self.non_ast_types[literal_type] = Symbol(literal_type, node_type)            
    
    def build_message(self, node, message, qattr: SymbolAttr = start_symbol):
        if node is None and (len(message) == 0 or message[-1] != "[Constant]"):
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
            ch_attr = SymbolAttr("", qattr.symbol_name, is_seq=False, has_values=True, group = qattr.group)
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
            if node is not None:
                message.append(str(node))
            else:
                message.append("")
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
        elif chunk[0] in attr.possible_symbols: #only case for [int], [str] .. and other Constant node funcs
            if chunk[0] == "[bool]":
                return "".join(chunk[1:]) != "False"
            if chunk[0] == "[NoneType]":
                return None
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

        one_attr = SymbolAttr("", attr.symbol_name, is_seq=False, has_values=True, group = attr.group)
        while len(mq) > 0:
            if (cur := mq.popleft()) == NEND: 
                break 
            mq.appendleft(cur)            
            symbol, attrs = self._decode_symbol_arg(one_attr, mq, constructor = constructor)
            res.append(constructor((symbol, attrs)))

        return res

    def unparse(self, message: 'list[str]', start_symbol = start_symbol, constructor = lambda x: x):
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

        one_attr = SymbolAttr("", attr.symbol_name, is_seq=False, has_values=True, group = attr.group)
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

# x = 'class Hex(SpellCard):§    def __init__(self):§        super().__init__("Hex", 3, CHARACTER_CLASS.SHAMAN, CARD_RARITY.FREE, target_func=hearthbreaker.targeting.find_minion_spell_target)§§    def use(self, player, game):§        super().use(player, game)§§        frog = hearthbreaker.cards.minions.neutral.Frog()§        minion = frog.create_minion(None)§        minion.card = frog§        self.target.replace(minion)§'
# x = x.replace("§", "\n")
# t = ast.parse(x)
# v = GrammarCollector()
# for t in tasts:
#     v.collect_metadata(t)    

# # v.collect_metadata(t)
# print(v.build_message(t, []))

# v = GrammarCollector()
# for t in tasts:
#     v.collect_metadata(t)    

# [s for (n, s) in v.symbols.items() if n == '[Constant]']

# # p = 'class IronbarkProtector(MinionCard):§    def __init__(self):§        super().__init__("Ironbark Protector", 8, CHARACTER_CLASS.DRUID, CARD_RARITY.COMMON)§§    def create_minion(self, player):§        return Minion(8, 8, taunt=True)§'
# p = 'class KnifeJuggler(MinionCard):§    def __init__(self):§        super().__init__("Knife Juggler", 2, CHARACTER_CLASS.ALL, CARD_RARITY.RARE)§§    def create_minion(self, player):§        return Minion(3, 2, effects=[Effect(AfterAdded(), ActionTag(Damage(1), CharacterSelector(players=EnemyPlayer(), picker=RandomPicker(), condition=None)))])§'
# p = p.replace("§", "\n")
# t = ast.parse(p)
# v.build_message(t, [])
# v.symbols['[FunctionDef]']
    # ]       ]", "
    # ([^\["])\[         $1", "[
# x = ["[FunctionDef]", "__init__", "[NEND]", "[arguments]", "[LST]", "[arg]", "self", "[NEND]", "[NEND]", "[LST]", "[Expr]", "[Call]", "[Attribute]", "[Call]", "[Name]", "super", "[NEND]", "[Load]", "[LST]", "[NEND]", "[LST]", "[NEND]", "__init__", "[NEND]", "[Load]", "[LST]", "[Constant]", "[str]", "Gnomish Inventor", "[NEND]", "[Constant]", "[int]", "4", "[NEND]", "[Attribute]", "[Name]", "CHARACTER_CLASS", "[NEND]", "[Load]", "ALL", "[NEND]", "[Load]", "[Attribute]", "[Name]", "CARD_RARITY", "[NEND]", "[Load]", "COMMON", "[NEND]", "[Load]", "[NEND]", "[LST]", "[keyword]", "battlecry", "[NEND]", "[Call]", "[Name]", "Battlecry", "[NEND]", "[Load]", "[LST]", "[Call]", "[Name]", "Draw", "[NEND]", "[Load]", "[LST]", "[NEND]", "[LST]", "[NEND]", "[Call]", "[Name]", "PlayerSelector", "[NEND]", "[Load]", "[LST]", "[NEND]", "[LST]", "[NEND]", "[NEND]", "[LST]", "[NEND]", "[NEND]"]
# x = ["[FunctionDef]", "[NEND]", "[arguments]", "[LST]", "[NEND]", "!", "[NEND]", "[Return]", "[List]", "[LST]", "[BoolOp]", "[Load]", "[NEND]", "[Return]", "[Call]", "[Call]", "[List]", "[LST]", "[Call]", "[Name]", "Minion", "[NEND]", "[Load]", "[LST]", "[Constant]", "[int]", "2", "[NEND]", "[Constant]", "[int]", "4", "[NEND]", "[NEND]", "[LST]", "[NEND]", "[NEND]", "[Load]", "[LST]"]
# x = ["[Module]", "[LST]", "[ClassDef]", "[CLS0]", "[NEND]", "[LST]", "[Name]", "MinionCard", "[NEND]", "[Load]", "[NEND]", "[LST]", "[FunctionDef]", "__init__", "[NEND]", "[arguments]", "[LST]", "[arg]", "self", "[NEND]", "[NEND]", "[LST]", "[Expr]", "[Call]", "[Attribute]", "[Call]", "[Name]", "super", "[NEND]", "[Load]", "[LST]", "[NEND]", "[LST]", "[NEND]", "__init__", "[NEND]", "[Load]", "[LST]", "[str]", "Boulderfist Ogre", "[NEND]", "[Num]", "6", "[NEND]", "[Attribute]", "[Name]", "CHARACTER_CLASS", "[NEND]", "[Load]", "ALL", "[NEND]", "[Load]", "[Attribute]", "[Name]", "CARD_RARITY", "[NEND]", "[Load]", "FREE", "[NEND]", "[Load]", "[NEND]", "[LST]", "[NEND]", "[NEND]", "[FunctionDef]", "create_minion", "[NEND]", "[arguments]", "[LST]", "[arg]", "self", "[NEND]", "[arg]", "[v0]", "[NEND]", "[NEND]", "[LST]", "[Return]", "[Call]", "[Name]", "Minion", "[NEND]", "[Load]", "[LST]", "[Num]", "6", "[NEND]", "[Num]", "7", "[NEND]", "[NEND]", "[LST]", "[NEND]", "[NEND]", "[NEND]", "[NEND]"]  #"[keyword]", "[NEND]", "[Call]", "[Call]"]
# x = ['[Module]', '[LST]', '[ClassDef]', 'B', 'oulder', 'f', 'ist', 'O', 'gre', '[NEND]', '[LST]', '[Name]', 'Min', 'ion', 'Card', '[NEND]', '[Load]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[Call]', '[Call]', '[a]']
# x = ['[Module]', '[LST]', '[ClassDef]', 'Spell', 'breaker', '[NEND]', '[LST]', '[Name]', 'Min', 'ion', 'Card', '[NEND]', '[Load]', '[NEND]', '[LST]', '[FunctionDef]', '__', 'init', '__', '[NEND]', '[arguments]', '[LST]', '[arg]', 'self', '[NEND]', '[NEND]', '[LST]', '[Expr]', '[Call]', '[Attribute]', '[Call]', '[Name]', 'super', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '__', 'init', '__', '[NEND]', '[Load]', '[LST]', '[Constant]', '[str]', 'Spell', 'breaker', '[NEND]', '[Constant]', '[int]', '4', '[NEND]', '[Attribute]', '[Name]', 'CHAR', 'AC', 'TER', '_', 'CLASS', '[NEND]', '[Load]', 'ALL', '[NEND]', '[Load]', '[Attribute]', '[Name]', 'C', 'ARD', '_', 'R', 'AR', 'ITY', '[NEND]', '[Load]', 'COM', 'MON', '[NEND]', '[Load]', '[NEND]', '[LST]', '[keyword]', 'battle', 'ion', '_', 'type', '[NEND]', '[Attribute]', '[Name]', 'MIN', 'ION', '_', 'TYPE', '[NEND]', '[Load]', 'N', 'ONE', '[NEND]', '[Load]', '[keyword]', 'battle', 'cry', '[NEND]', '[Call]', '[Name]', 'Battle', 'cry', '[NEND]', '[Load]', '[LST]', '[Call]', '[Name]', 'Sil', 'ence', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[Call]', '[Name]', 'Min', 'ion', 'Select', 'or', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[keyword]', 'players', '[NEND]', '[Call]', '[Name]', 'Both', 'Player', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[keyword]', 'p', 'icker', '[NEND]', '[Call]', '[Name]', 'User', 'P', 'icker', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[FunctionDef]', '[NEND]', '[arguments]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[NEND]', '[Expr]', '[Name]', '[NEND]', '[Load]', '[NEND]', 'player', '[NEND]', '[NEND]', '[NEND]', '[Return]', '[NEND]', '[Name]', '[NEND]', 'Min', 'ion', '[NEND]', '[Constant]', '[NEND]', 'a', '[NEND]', '[NEND]', '4', '[NEND]', '[Constant]', '[NEND]', '3', '3', '[NEND]', '[NEND]', '[NEND]', 'red', '[NEND]', '[NEND]', '[NEND]', '[NEND]', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'A', 'In', 'In', 'In', 'In', 'In', 'In', 'A', 'A', 'In', 'A', 'In', 'In', 'A', 'A', 'A', 'A', 'A', 'A', 'In', 'A', 'A', 'A', 'A', 'In', 'In', 'A', 'A', 'A', 'In', 'In', 'In', 'In', 'In', 'A', 'A', 'A', 'A', 'In', 'In', 'In', 'A', 'In', 'A', 'A', 'In', 'In', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'In', 'A', 'In', 'A', 'A', 'A', 'A', 'In', 'A', 'A', 'A', 'A', 'In', 'A', 'A', 'A', 'In', 'A', 'A', 'A', 'A', 'A', 'A', 'In', 'A', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'A', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In', 'In']
# x = ['[Module]', '[LST]', '[ClassDef]', 'Kn', 'ife', 'J', 'ugg', 'ler', '[NEND]', '[LST]', '[Name]', 'Min', 'ion', 'Card', '[NEND]', '[Load]', '[NEND]', '[LST]', '[FunctionDef]', '__', 'init', '__', '[NEND]', '[arguments]', '[LST]', '[arg]', 'self', '[NEND]', '[NEND]', '[LST]', '[Expr]', '[Call]', '[Attribute]', '[Call]', '[Name]', 'super', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '__', 'init', '__', '[NEND]', '[Load]', '[LST]', '[Constant]', '[str]', 'Kn', 'ife', ' Jugg', 'ler', '[NEND]', '[Constant]', '[int]', '2', '[NEND]', '[Attribute]', '[Name]', 'CHAR', 'AC', 'TER', '_', 'CLASS', '[NEND]', '[Load]', 'ALL', '[NEND]', '[Load]', '[Attribute]', '[Name]', 'C', 'ARD', '_', 'R', 'AR', 'ITY', '[NEND]', '[Load]', 'R', 'ARE', '[NEND]', '[Load]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[FunctionDef]', 'create', '_', 'min', 'ion', '[NEND]', '[arguments]', '[LST]', '[arg]', 'self', '[NEND]', '[arg]', 'player', '[NEND]', '[NEND]', '[LST]', '[Return]', '[Call]', '[Name]', 'Min', 'ion', '[NEND]', '[Load]', '[LST]', '[Constant]', '[int]', '3', '[NEND]', '[Constant]', '[int]', '2', '[NEND]', '[NEND]', '[LST]', '[keyword]', 'effects', '[NEND]', '[List]', '[LST]', '[Call]', '[Name]', 'Effect', '[NEND]', '[Load]', '[LST]', '[Call]', '[Name]', 'Min', 'Effect', '[NEND]', '[Load]', '[LST]', '[Call]', '[Call]', '[Call]', '[Call]', '[Name]', 'Action', 'Tag', '[NEND]', '[Load]', '[LST]', '[Call]', '[Name]', 'Damage', '[NEND]', '[Load]', '[LST]', '[Constant]', '[int]', '1', '[NEND]', '[NEND]', '[LST]', '[NEND]', '[Call]', '[Name]', 'Character', 'Select', 'or', '[NEND]', '[Load]', '[LST]', '[Call]', '[Call]', '[Call]', '[Name]', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[keyword]', 'p', 'icker', '[NEND]', '[Call]', '[Name]', 'Random', 'P', 'icker', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[Load]', '[NEND]', '[Assign]', '[LST]', '[NEND]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[ListComp]', '[Call]', '[Call]', '[Call]', '[Call]', '[ListComp]', '[ListComp]', '[ListComp]', '[Call]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[ListComp]', '[Call]', '[ListComp]', '[Call]', '[Call]', '[ListComp]', '[Call]', '[ListComp]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]']
# x = ['[Module]', '[LST]', '[ClassDef]', 'B', 'oulder', 'f', 'ist', 'O', 'gre', '[NEND]', '[LST]', '[Name]', 'Min', 'ion', 'Card', '[NEND]', '[Load]', '[NEND]', '[LST]', '[FunctionDef]', '__', 'init', '__', '[NEND]', '[arguments]', '[LST]', '[arg]', 'self', '[NEND]', '[NEND]', '[LST]', '[Expr]', '[Call]', '[Attribute]', '[Call]', '[Name]', 'super', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '__', 'init', '__', '[NEND]', '[Load]', '[LST]', '[Constant]', '[str]', 'B', 'oulder', 'f', 'ist', ' Ogre', '[NEND]', '[Constant]', '[int]', '6', '[NEND]', '[Attribute]', '[Name]', 'CHAR', 'AC', 'TER', '_', 'CLASS', '[NEND]', '[Load]', 'ALL', '[NEND]', '[Load]', '[Attribute]', '[Name]', 'C', 'ARD', '_', 'R', 'AR', 'ITY', '[NEND]', '[Load]', 'FREE', '[NEND]', '[Load]', '[Constant]', '[bool]', 'True', '[NEND]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[ImportFrom]', '[Attribute]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[Expr]', '[Name]', '[NEND]', '[Load]', '[NEND]', '[Return]', '[Call]', '[Name]', 'Min', 'ion', '[NEND]', '[Load]', '[LST]', '[Constant]', '[int]', '6', '[NEND]', '[Constant]', '[int]', '7', '[NEND]', '[NEND]', '[LST]', '[keyword]', '[Call]', '[NEND]', '[Call]', '[Name]', 'A', 'A', 'A', 'A', 'A', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[NEND]']
# x = ['[Module]', '[LST]', '[ClassDef]', 'V', 'apor', 'ize', '[NEND]', '[LST]', '[Name]', 'Secret', 'Card', '[NEND]', '[Load]', '[NEND]', '[LST]', '[FunctionDef]', '__', 'init', '__', '[NEND]', '[arguments]', '[LST]', '[arg]', 'self', '[NEND]', '[NEND]', '[LST]', '[Expr]', '[Call]', '[Attribute]', '[Call]', '[Name]', 'super', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '__', 'init', '__', '[NEND]', '[Load]', '[LST]', '[Constant]', '[str]', 'V', 'apor', 'ize', '[NEND]', '[Constant]', '[int]', '3', '[NEND]', '[Attribute]', '[Name]', 'CHAR', 'AC', 'TER', '_', 'CLASS', '[NEND]', '[Load]', 'M', 'AGE', '[NEND]', '[Load]', '[Attribute]', '[Name]', 'C', 'ARD', '_', 'R', 'AR', 'ITY', '[NEND]', '[Load]', 'R', 'ARE', '[NEND]', '[Load]', '[NEND]', '[LST]', '[NEND]', '[FunctionDef]', '[Call]', '_', 'reve', 'al', '[NEND]', '[arguments]', '[LST]', '[arg]', 'self', '[NEND]', '[arg]', 'att', 'acker', '[NEND]', '[arg]', 'target', '[NEND]', '[NEND]', '[LST]', '[If]', '[BoolOp]', '[And]', '[LST]', '[Compare]', '[Call]', '[Name]', '[NEND]', '[Load]', '[LST]', '[Call]', '[Name]', '[And]', '[Attribute]', '[Attribute]', '[Attribute]', 'self', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[Attribute]', '[Constant]', '[int]', '0', '0', '[NEND]', '[Attribute]', 'target', 'acker', '[NEND]', '[Load]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[IsNot]', '[NEND]', '[LST]', '[NEND]', '[Attribute]', '[Attribute]', '[Attribute]', '[Compare]', '[Name]', '[Attribute]', '[Attribute]', 'att', 'acker', '[NEND]', '[Load]', '[LST]', '[Lt]', '[NEND]', '[LST]', '[NEND]', '[And]', '[And]', '[Call]', '[Attribute]', '[And]', '[Attribute]', 'att', '[NEND]', '[Load]', '[And]', 'add', '[NEND]', '[Load]', ' 3', ')', 'self', '[NEND]', '[Load]', '[NEND]', '[LST]', '[Expr]', '[Call]', '[Call]', '[Attribute]', '[Name]', '[Attribute]', 'att', 'acker', '[NEND]', '[Load]', 'remove', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[LST]', '[Expr]', '[Call]', '[Call]', '[Attribute]', '[Call]', '[Name]', 'super', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', 'reve', 'al', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[LST]', '[Constant]', '[NoneType]', '[NEND]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[NEND]', '[Expr]', '[Name]', '[NEND]', '[Load]', '[NEND]', '[NEND]', '[NEND]', '[And]', '[And]', '[Expr]', '[Call]', '[Attribute]', '[Attribute]', '[Attribute]', '[Attribute]', 'player', '[NEND]', 'game', ' a', 'game', 'onent', '[NEND]', '_', '.', 'bind', '[NEND]', ' on', ' on', '[Expr]', ' on', '_', 'character', '_', 'attack', '_', '[Attribute]', '[Attribute]', '[Attribute]', 'self', '[NEND]', '[Attribute]', '[Attribute]', '_', 'reve', 'al', '_', '[Attribute]', '[Attribute]', '[Attribute]', ' 3', '[Attribute]', '[Attribute]', '_', 'de', 'activate', '[NEND]', '[And]', '[Or]', '[Expr]', '_', 'self', '[NEND]', '_', '_', 'player', '[NEND]', '[And]', '[And]', '[Expr]', '[Call]', '[Attribute]', '[Attribute]', '[Attribute]', '_', 'player', '[NEND]', 'opp', 'opp', 'opp', 'onent', '[NEND]', 'un', 'un', 'un', 'bind', '[NEND]', '_', '[Or]', '[And]', '[str]', '_', 'character', '_', 'attack', '_', '[Attribute]', '[Attribute]', ',', 'self', '[NEND]']
# x = ['[Module]', '[LST]', '[ClassDef]', 'V', 'apor', 'ize', '[NEND]', '[LST]', '[Name]', 'Secret', 'Card', '[NEND]', '[Load]', '[NEND]', '[LST]', '[FunctionDef]', '__', 'init', '__', '[NEND]', '[arguments]', '[LST]', '[arg]', 'self', '[NEND]', '[NEND]', '[LST]', '[Expr]', '[Call]', '[Attribute]', '[Call]', '[Name]', 'super', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '__', 'init', '__', '[NEND]', '[Load]', '[LST]', '[Constant]', '[str]', 'V', 'apor', 'ize', '[NEND]', '[Constant]', '[int]', '3', '[NEND]', '[Attribute]', '[Name]', 'CHAR', 'AC', 'TER', '_', 'CLASS', '[NEND]', '[Load]', 'M', 'AGE', '[NEND]', '[Load]', '[Attribute]', '[Name]', 'C', 'ARD', '_', 'R', 'AR', 'ITY', '[NEND]', '[Load]', 'R', 'ARE', '[NEND]', '[Load]', '[NEND]', '[LST]', '[NEND]', '[FunctionDef]', '[Call]', '_', 'reve', 'al', '[NEND]', '[arguments]', '[LST]', '[arg]', 'self', '[NEND]', '[arg]', 'att', 'acker', '[NEND]', '[arg]', 'target', '[NEND]', '[NEND]', '[LST]', '[If]', '[BoolOp]', '[And]', '[LST]', '[Compare]', '[Call]', '[Name]', '[NEND]', '[Load]', '[LST]', '[Call]', '[Name]', '[And]', '[Attribute]', '[Attribute]', '[Attribute]', 'self', '[NEND]']
# x = ['[Module]', '[LST]', '[ClassDef]', 'Black', 'wing', 'Cor', 'rupt', 'or', '[NEND]', '[LST]', '[Name]', 'Min', 'ion', 'Card', '[NEND]', '[Load]', '[NEND]', '[LST]', '[FunctionDef]', '__', 'init', '__', '[NEND]', '[arguments]', '[LST]', '[arg]', 'self', '[NEND]', '[NEND]', '[LST]', '[Expr]', '[Call]', '[Attribute]', '[Call]', '[Name]', 'super', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '__', 'init', '__', '[NEND]', '[Load]', '[LST]', '[Constant]', '[str]', 'Black', 'wing', ' Cor', 'rupt', 'or', '[NEND]', '[Constant]', '[int]', '5', '[NEND]', '[Attribute]', '[Name]', 'CHAR', 'AC', 'TER', '_', 'CLASS', '[NEND]', '[Load]', 'ALL', '[NEND]', '[Load]', '[Attribute]', '[Name]', 'C', 'ARD', '_', 'R', 'AR', 'ITY', '[NEND]', '[Load]', 'COM', 'MON', '[NEND]', '[Load]', '[NEND]', '[LST]', '[keyword]', 'battle', 'cry', '[NEND]', '[Call]', '[Name]', 'Battle', 'cry', '[NEND]', '[Load]', '[LST]', '[Call]', '[Name]', 'Damage', '[NEND]', '[Load]', '[LST]', '[Constant]', '[int]', '3', '[NEND]', '[NEND]', '[LST]', '[NEND]', '[Call]', '[Name]', 'Character', 'Select', 'or', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[keyword]', 'players', '[NEND]', '[Call]', '[Name]', 'Both', 'Player', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[keyword]', 'p', 'icker', '[NEND]', '[Call]', '[Name]', 'User', 'P', 'icker', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[NEND]', '[NEND]', '[ImportFrom]', '.', '.', ' 5', '[NEND]', '[LST]', '[NEND]', '.', '.', '.', '[NEND]', '[NEND]', 'Card', 'Select', 'or', '[Lt]', '[keyword]', '[And]', '[keyword]', '[keyword]', '[keyword]', '[keyword]', 'condition', '[NEND]', '[And]', '[Attribute]', '[Attribute]', 'Is', 'Type', '[NEND]', '[Attribute]', '[Attribute]', '[Attribute]', '[Attribute]', '[Attribute]', 'MIN', 'ION', '_', 'TYPE', '[NEND]', '[Attribute]', '[Attribute]', 'D', 'RAG', 'ON', '_', '[keyword]', '[Attribute]', '[keyword]', '[keyword]', '[keyword]', '[keyword]', '[Call]', '[keyword]', '[Call]', '[Call]', '[Call]', '[keyword]', 'value', '[NEND]', '[keyword]', '[Or]', '[Or]', '[Or]', '[NEND]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', '[Call]', 'create', '_', 'min', 'ion', '[NEND]', '[Call]', '[List]', 'self', 'c', 'self', '[NEND]', 'player', 'c', 'player', '[NEND]', '[And]', '[And]', '[Return]', '[Call]', 'backs', '[NEND]', 'Min', 'ion', '[NEND]', '[And]', '[And]', ' 5', '[int]', '[And]', ' 5', '[NEND]', '[Constant]', '[int]', '[And]', '[And]', '[NEND]', '[And]', '[Sub]', ' 3', '[Call]', '[Call]', '[NEND]', '[NEND]']
# x = ['[Module]', '[LST]', '[ClassDef]', 'Ban', 'anas', '[NEND]', '[LST]', '[Name]', 'Spell', 'Card', '[NEND]', '[Load]', '[NEND]', '[LST]', '[FunctionDef]', '__', 'init', '__', '[NEND]', '[arguments]', '[LST]', '[arg]', 'self', '[NEND]', '[NEND]', '[LST]', '[Expr]', '[Call]', '[Attribute]', '[Call]', '[Name]', 'super', '[NEND]', '[Load]', '[LST]', '[NEND]', '[LST]', '[NEND]', '__', 'init', '__', '[NEND]', '[Load]', '[LST]', '[Constant]', '[str]', 'Ban', 'anas', '[NEND]', '[Constant]', '[int]', '1', '[NEND]', '[Attribute]', '[Name]', 'CHAR', 'AC', 'TER', '_', 'CLASS', '[NEND]', '[Load]', 'ALL', '[NEND]', '[Load]', '[Attribute]', '[Name]', 'C', 'ARD', '_', 'R', 'AR', 'ITY', '[NEND]', '[Load]', 'COM', 'MON', '[NEND]', '[Load]', '[Constant]', '[bool]', 'False', '[NEND]', '[Attribute]', '[Name]', ' to', 'he', 'arth', 'breaker', '[NEND]', '[Load]', 'target', 'ing', '[NEND]', '[Load]', '[NEND]', '[LST]', '[NEND]', '[NEND]', '[ImportFrom]', 'spell', '_', 'target', '[NEND]', '[LST]', '[alias]', ' 5', ' T', ' T', '[NEND]', '[NEND]', 'player', '[NEND]', '[Expr]', '[Name]', ' "', 'player', '[NEND]', '[Load]', '[NEND]', '[Expr]', '[Name]', '.', 'b', 'b', ' 3', '[NEND]', '[Load]', '[NEND]', '[Attribute]', '[Attribute]', 'super', '[NEND]', 'use', '[NEND]', '[Add]', 'use', '[Sub]', 'use', 'use', '[NEND]', '[arg]', '[And]', '[And]', '[Attribute]', 'player', '[NEND]', '[Or]', '[Or]', '[Or]', 'game', '[NEND]', '[Lambda]', '[Lambda]', '[Lambda]', '[Expr]', '[Add]', '[Call]', '[Attribute]', '[Attribute]', '[Attribute]', '[Attribute]', 'self', '[NEND]', 'target', '[Attribute]', 'target', '[NEND]', '[Add]', '[Attribute]', 'change', '_', 'attack', '_', '[Sub]', '.', '[Sub]', '[Sub]', ' day', '[And]', '.', '[Sub]', '[And]', '[Sub]', '[Sub]', '[Call]', '[Attribute]', '[Attribute]', '[Attribute]', '[Attribute]', 'self', '[NEND]', 'target', '[Attribute]', 'target', '[NEND]', 'add', '[Attribute]', 'change', 'ase', '_', 'health', '[NEND]', '[Call]', '.', '[Call]', '[int]', ' the', '[And]', '[NEND]', '[And]', '[And]', '[And]', '[And]', '[And]', '[NEND]', '[NEND]']
# res = v.unparse(x, start_symbol = start_symbol, constructor=v.build_module)
# astunparse.unparse(res)


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
        # try:
        # print(m)        
        r = v.unparse(m, constructor=v.build_module)
        txt = astunparse.unparse(r)
        print(txt)
        print()
        # except Exception as e:
        #     print(e)
        #     print()
        #     print(i)
        #     break

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