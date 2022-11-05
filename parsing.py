''' Module for target sentence preprocessing '''

def prolog_expr_preprocess(s: str) -> str:
    return s.replace('\\+','not ').replace('[', '(').replace(']', ')').replace("'.'", '.')[:-1]

def parse_sexpr(s: str):
    ''' converts string to tree - assuming Uniterpreted functions of form f(...)! '''
    s = prolog_expr_preprocess(s)
    symbs = {'(', ')', ','}
    def parse_name(acc, i):
        i0 = i
        while i < len(s) and s[i] not in symbs:
            i += 1
        if i < len(s) and s[i] == '(':
            f = [s[i0:i]]
            acc.append(f)
            i += 1
            i = parse_params(f, i) #after this f contains all args
            assert s[i] == ')', f"Not at ) {i} for: {s}"
            i += 1 #passing )
        else: #end 
            acc.append(s[i0:i])
        return i
    def parse_params(acc, i):
        while i < len(s) and s[i] != ')':
            i = parse_name(acc, i)
            if s[i] == ',':
                i += 1
        return i
    acc = []
    i = parse_name(acc, 0)
    assert i == len(s), f"Not at end {i} for: {s}"
    return acc 


def parse2(s: str):
    ''' converts string to tree - assuming Uniterpreted functions of form f(...)! '''
    s = prolog_expr_preprocess(s)
    symbs = {'(', ')', ','}
    def parse_name(acc, i):
        i0 = i
        while i < len(s) and s[i] not in symbs:
            i += 1
        if i < len(s) and s[i] == '(':
            name = s[i0:i].strip()            
            if name.startswith("not "):
                name = name.split(' ')[-1]
                f = [name]
                acc.append(["not", f])
            else:
                f = [name]
                acc.append(f)
            i += 1
            i = parse_params(f, i) #after this f contains all args
            if f[0] == "not": #make not to have only one arg
                f[1:] = [['', *f[1:]]] 
            # if f[0] == "":
            #     f[0] = "AND"
            #     f[1:] = [len(f) - 1, f[1:]] 
            # f[0] = f[0] if f[0] == "AND" or f[0] == "NOT" else f"{f[0]}:{str(len(f) - 1)}"
            assert s[i] == ')', f"Not at ) {i} for: {s}"
            i += 1 
        else:
            acc.append(s[i0:i].strip("'"))
        return i
    def parse_params(acc, i):
        while i < len(s) and s[i] != ')':
            i = parse_name(acc, i)
            if s[i] == ',':
                i += 1
        return i
    acc = []
    i = parse_name(acc, 0)
    assert i == len(s), f"Not at end {i} for: {s}"
    def split_list(l):
        if type(l) != list:
            return l
        elif l[0] == '':
            if not any(type(x) == list for x in l):
                return l[1:]
            root = subt = []
            for i, el in enumerate(l):
                if i == 0:
                    continue
                subt.append('and')                
                subt.append(split_list(el))
                if i == len(l) - 2:
                    subt.append(split_list(l[-1]))
                    break
                else:                
                    new_subt = []
                    subt.append(new_subt)
                    subt = new_subt
            return root
        else:
            return [split_list(el) for el in l]                
    res = split_list(acc)
    return res[0]

# [_, query, ast] = parse2("parse([how,many,rivers,do,not,traverse,the,state,with,the,capital,albany,?], answer(A,count(B,(river(B),\+ (traverse(B,C),state(C),loc(D,C),capital(D),const(D,cityid('albany',_)))),A))).")
# print(res)

# ['parse', ['how', 'many', 'rivers', 'do', 'not', 'traverse', 'the', 'state', 'with', 'the', 'capital', 'albany', '?'], ['answer', 'A', ['count', 'B', ['and', ['river', 'B'], ['not', ['and', ['traverse', 'B', 'C'], ['and', ['state', 'C'], ['and', ['loc', 'D', 'C'], ['and', ['capital', 'D'], ['const', 'D', ['cityid', 'albany', '_']]]]]]]], 'A']]]

def s_expr_to_str(t, lpar="(", rpar=")"):
    if type(t) != list:
        return t
    if len(t) > 1:
        sep = " " if t[0] != '' else ''
        args = " ".join(s_expr_to_str(x, lpar = lpar, rpar = rpar) for x in t[1:])
        sep_args = sep + args 
    else:
        sep_args = ""
    return lpar + t[0] + sep_args + rpar

def add_arity(t, sep=":", symbol_arities = {}):
    if type(t) != list:
        return t
    arity = str(len(t) - 1)
    symbol_arities.setdefault(arity, []).append(t[0])
    return [t[0] + sep + arity, *[add_arity(ch, sep = sep, symbol_arities = symbol_arities) for ch in t[1:]]] 

def plain_print(t):
    tokens = []
    def pprint_inner(t):
        for ch in t:
            if type(ch) != list:
                tokens.append(str(ch))
            else:
                pprint_inner(ch)
    pprint_inner(t)
    return " ".join(tokens)

# print(plain_print(add_arity(ast)))
