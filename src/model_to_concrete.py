import sys
import lark
# A language based on a Lark example from:
# https://github.com/lark-parser/lark/wiki/Examples
'(define-fun _a_3_ () Real (to_real 0))'
GRAMMAR = """
start: "(" "define-fun" var "()" "Real" expr ")"

// Defining expression Nonterminal 
expr:  "(" "/" num num ")"      -> div
    | "("  "to_real" expr ")"   -> to_real
    | "(" "-" expr ")"          -> neg
    | num                       -> num

var: /[a-zA-Z_][a-zA-Z0-9_]*/
num: NUMBER
    | "-" NUMBER -> neg

COMMENT: /\/\/.*/
%import common.NUMBER
%import common.WORD
%import common.WS
%import common.CNAME
%ignore WS
%ignore COMMENT
""".strip()


def get_parser():
    parser = lark.Lark(GRAMMAR, propagate_positions=True, parser='lalr')
    return parser


def get_expr(expr):
    if type(expr) == lark.Token:
        return float(str(expr))
    elif expr.data == "num":
        return get_expr(expr.children[0])
    elif expr.data == "neg":
        return -1*get_expr(expr.children[0])
    elif expr.data == "div":
        return get_expr(expr.children[0])/get_expr(expr.children[1])
    elif expr.data == "to_real":
        return get_expr(expr.children[0])
    else:
        raise Exception("Not Implemented")


def get_concrete(model_file, symbolic_poly):

    parser = get_parser()

    model = open(model_file, "r").readlines()

    subs = dict()
    for line in model:
        line = line.strip()
        if line.startswith("(define"):

            tree = parser.parse(line)
            var_name = str(tree.children[0].children[0])
            expr_tree = tree.children[1]
            subs[var_name] = get_expr(expr_tree)
    symbolics = open(symbolic_poly, 'r').read().split('\n')
    concretes = []
    for symbolic in symbolics:
        concrete = symbolic
        s0, smain = concrete.split('gas = ')
        split_concrete = list(map(str.strip, smain.split("+")))
        for k in subs:
            if subs[k] == 0:
                new_concrete = []
                for j in split_concrete:
                    if j.startswith(k):
                        pass
                    else:
                        new_concrete.append(j)

                split_concrete = new_concrete

        concrete = " + ".join(split_concrete)
        for k in subs:
            concrete = concrete.replace(k, "{:.4f}".format(subs[k]))
        concretes.append(s0+concrete)
    return '\n'.join(concretes)


if __name__ == '__main__':

    model_file = sys.argv[1]
    symbolic_poly = sys.argv[2]
    concrete = get_concrete(model_file, symbolic_poly)
    open(symbolic_poly + ".solve", "w").write(concrete + "\n")
