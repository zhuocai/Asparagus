from utils import is_number

import random
import sympy as sp
import copy
from polynomial import Polynomial
import time
from PolynomialTS import PolynomialTS
from z3 import *
import os
import time

import main_parser
import pandas as pd
import argparse
from model_to_concrete import get_concrete


random.seed(0)

debug = True
tree = None


bound = 100

condition_pairs = []
variables_index = None
all_unknown_coefficients = []
unknown_coefficients_index = 0
all_real_vars = dict()

sp_unit = 1
unit_poly = Polynomial({sp_unit: 1})

args = None
###########################################################
#                   POSITIVSTELLENSATZ                    #
###########################################################

QP = []

cond_index_to_monomials = {}
all_ls_name = []
M_index = 0
M_dict = {}
inv_M_dict = {}


def all_vars_in_list_of_poly(lst):
    variables = set()
    for f in lst:
        for mono in f.monomial_dict:
            if type(mono) == int:
                continue
            variables = variables.union(mono.free_symbols)
    # return the dictionary of variable name: sympy variable
    return {str(x): x for x in variables}


# creates all monomials of degree smaller than k as a vector
def monomial_vector(real_vars, k):
    all_mono = {sp_unit}
    cur_mono = {sp_unit}
    for i in range(k):
        new_mono = set()
        for v in real_vars:
            var = real_vars[v]
            for m in cur_mono:
                new_mono.add(m*var)
        all_mono = all_mono.union(new_mono)
        cur_mono = new_mono

    ans = list(map(lambda x: [Polynomial({x: 1})], all_mono))
    # print(f"monomials of order {k} are {ans}")
    return ans


l_cnt = 0
h_complexity_code_to_randomness = {
    2: 0.1,
    3: 0.2,
    4: 0.3,
    5: 0.4,
    6: 0.5,
    7: 0.6,
    8: 0.7,
    9: 0.8,
    10: 0.9
}


def get_l_name(cond_index, with_handelman, index):
    t = 1 if with_handelman else 0
    return "_l_{}_{}_{}".format(t, cond_index, index)


def split_l_name(l_name):
    return l_name.split("_")[1:]


def generate_new_lower_triangular_matrix(n, cond_index, with_handelman, l_names, positive_l_names, l_cnt):
    """
    Generates a lower traingular matrix (via Cholskey Decomposition)
    h_complexity_code = 0: Use full matrix
    h_complexity_code = 1: Use only diagonal entries
    """
    ans = []
    for i in range(n):
        row = []
        for j in range(n):
            if j <= i:
                l_cnt[0] += 1
                fresh_var_name = get_l_name(
                    cond_index, with_handelman, l_cnt[0])
                l_names.append(fresh_var_name)
                fresh_var = Polynomial({sp_unit: sp.Symbol(fresh_var_name)})
                row += [fresh_var]
                if i == j:
                    positive_l_names.append(fresh_var_name)
            else:
                row += [Polynomial({})]
        ans += [row]
    m = ans
    return m


def transpose_poly_matrix(mat):
    dimr = len(mat)
    dimc = len(mat[0])
    result = [[None for i in range(dimr)] for j in range(dimc)]
    for i in range(dimr):
        for j in range(dimc):
            result[j][i] = mat[i][j]
    return result


def multiply_poly_matrices(mat1, mat2):
    """
    Assume matrices are 2D as list of lists, and both matrices are nonempty
    """
    dim1r = len(mat1)
    dim1c = len(mat1[0])
    dim2r = len(mat2)
    dim2c = len(mat2[0])
    if (dim1c != dim2r):
        raise Exception("Matrices dimension mismatch!")
    result = [[Polynomial({}) for i in range(dim2c)] for j in range(dim1r)]
    for i in range(dim1r):
        for j in range(dim2c):
            for k in range(dim1c):
                result[i][j] += mat1[i][k]*mat2[k][j]
    return result


def handelman_monoid(lst, d):
    if args.print_verbose:
        print("Generating Handleman Monoid...")
    all_mono = {unit_poly}
    cur_mono = {unit_poly}
    for i in range(1, d+1):
        new_mono = set()
        for poly in lst:
            for m in cur_mono:
                new_m = m*poly
                new_mono.add(new_m)
        all_mono = all_mono.union(new_mono)
        cur_mono = new_mono
    return list(all_mono)


def generate_base_QP_dict(condition_pairs, h_degree):
    """
    Assuming condition_pair = (alpha, beta, real_vars: Dictionary)
    """
    QP_dict = dict()
    QP_with_Handelman_dict = dict()

    for cond_index in range(len(condition_pairs)):
        condition_pair = condition_pairs[cond_index]
        if args.print_verbose:
            print(f"Handling constraint pair: {cond_index}...")
        QP_dict[cond_index] = get_QP_for_condition_pair(
            condition_pair, cond_index, False, h_degree)
        if args.apply_handelman:
            QP_with_Handelman_dict[cond_index] = get_QP_for_condition_pair(
                condition_pair, cond_index, True, h_degree)
        else:
            QP_with_Handelman_dict = dict()

    return QP_dict, QP_with_Handelman_dict


def get_QP_for_condition_pair(condition_pair, cond_index, with_handelman, h_degree):
    l_names = []
    positive_l_names = []
    l_cnt = [0]
    (pre, post, real_vars) = condition_pair
    vars_in_condition_pair = all_vars_in_list_of_poly(pre+post)
    # find the largest total degree in the polynomials
    largest_degree = max(map(lambda x: x.total_degree, pre+post))
    largest_degree = max(largest_degree, post[0].total_degree)
    current_h_degree = 1 if largest_degree == 1 else h_degree
    # if largest_degree is 1, then apply farkas
    with_farkas = (current_h_degree == 1)
    if not with_farkas and with_handelman:
        # compute the monoid if it is not farkas
        pre = handelman_monoid(pre, 2)

    y = monomial_vector(vars_in_condition_pair, int(h_degree/2))
    dim = len(y)

    pre.append(Polynomial({sp_unit: 1}))
    lhs = post[0]
    rhs = Polynomial({})
    for f in pre:
        # print("Processing: ", f)
        if with_farkas or with_handelman:
            # both farkas and handelman use a single l var
            hM = generate_new_lower_triangular_matrix(
                1, cond_index, with_handelman, l_names, positive_l_names, l_cnt)
        else:
            L = generate_new_lower_triangular_matrix(
                dim, cond_index, with_handelman, l_names, positive_l_names, l_cnt)
            yL = multiply_poly_matrices(transpose_poly_matrix(y), L)
            hM = multiply_poly_matrices(yL, transpose_poly_matrix(yL))
        h = hM[0][0]
        rhs = rhs + (f * h)

    # add a sum of square of degree h_degree extra if handelman
    if with_handelman:
        L = generate_new_lower_triangular_matrix(
            dim, cond_index, with_handelman, l_names, positive_l_names, l_cnt)
        yL = multiply_poly_matrices(transpose_poly_matrix(y), L)
        hM = multiply_poly_matrices(yL, transpose_poly_matrix(yL))
        h0 = hM[0][0]
        rhs = rhs + h0

    zero_poly = lhs - rhs
    zero_conds, monomial_list_of_poly = zero_poly.get_eq_zero_conditions()

    return zero_conds, l_names, positive_l_names
    # useful_ls_for_cond_pair_for_monomial[cond_index] = {}
    # for c in range(len(zero_conds)):
    #     useful_ls_for_cond_pair_for_monomial[cond_index][monomial_list_of_poly[c]] = l_vars_in_expression(zero_conds[c])
    # print("LHS: {}\nRHS: {}".format(lhs, rhs))
    # print("Following constraints are generated:")
    # for i in list(lhs.monomial_dict.keys()) + list(rhs.monomial_dict.keys()):
    #     lcoff = lhs.monomial_dict[i] if i in lhs.monomial_dict else 0
    #     rcoff = rhs.monomial_dict[i] if i in rhs.monomial_dict else 0
    #     print("\t{}: {} = {}".format(i, lcoff, rcoff))

    # for c in range(len(zero_conds)):
    #     cond = zero_conds[c]
    #     M_dict[f"M_{M_index}"] = (cond_index, monomial_list_of_poly[c], cond)
    #     if c in zero_conds:
    #         raise Exception("Ambiguitiy encountered for inv_M_dict!")
    #     inv_M_dict[zero_conds[c]] = f"M_{M_index}"
    #     M_index += 1


def get_monomial_from_poly(y):
    lst = []
    for poly_lst in y:
        poly = poly_lst[0]
        lst.append(list(poly.monomial_dict.keys())[0])
    return lst


def l_vars_in_expression(cond):
    lst = []
    atoms = cond.atoms()
    for atom in atoms:
        if str(atom).startswith("l"):
            lst.append(atom)
    return list(map(str, lst))


def pow_to_mul(expr):
    """
    Convert integer powers in an expression to Muls, like a**2 => a*a.
    """
    pows = list(expr.atoms(sp.Pow))
    repl = zip(pows, [sp.Mul(*[b]*e, evaluate=False)
               for b, e in (i.as_base_exp() for i in pows)])
    dic = {a[0]: a[1] for a in repl}
    return expr.subs(dic)


def get_lst_lines_for_z3(zero_conds, cond_index, with_assert=True):
    m_cnt = 0
    lst_lines = []
    for i in zero_conds:
        if i == sp.sympify("True"):
            line = None
            continue
        elif i == sp.sympify("False"):
            line = f"s.add({i})\n"
            lst_lines = [line]
            break
        elif type(i) == sp.core.relational.GreaterThan:
            expr = pow_to_mul(i.lhs)
            line = f"s.add( ({expr}>={pow_to_mul(i.rhs)}))\n"
        elif type(i) == sp.core.relational.Equality:
            if with_assert:
                line = f"s.assert_and_track(({pow_to_mul(i.lhs)} == {pow_to_mul(i.rhs)}), 'M_{cond_index}_{m_cnt}')\n"
            else:
                line = f"s.add(({pow_to_mul(i.lhs)} == {pow_to_mul(i.rhs)}))\n"
            m_cnt += 1
        else:
            raise Exception("Unknown constraint type in QP")
        if line:
            lst_lines.append(line)
    return lst_lines


def solve_with_abstraction_refinement(QP_dict, QP_with_Handelman_dict, with_abstraction=True):
    # print(inv_M_dict)
    with_handelman_dict = {
        cond_index: apply_handelman for cond_index in QP_dict}
    if args.print_verbose:
        print("-"*10)

    if with_abstraction:
        if args.print_verbose:
            print("Solving with Abstraction Refinement Heuristics...")
    else:
        if args.print_verbose:
            print("Solving with Z3 without Heuristics...")

    if args.print_verbose:
        print("Initiated solver...")
    for i in all_unknown_coefficients:
        line = f"{i} = Real('{i}')"
        if args.print_verbose:
            print(line)
        exec(line)
    # pdb.set_trace()
    for cond_index in QP_dict:
        l_names = QP_dict[cond_index][1]
        if apply_handelman:
            Hl_names = QP_with_Handelman_dict[cond_index][1]
            for l in Hl_names:
                line = f"{l} = Real('{l}')"
                exec(line)
        for l in l_names:
            line = f"{l} = Real('{l}')"
            exec(line)

    while True:
        s = Solver()
        s.set(unsat_core=True)
        all_ls = 0
        set_ls_zero = set()
        with sp.core.parameters.evaluate(0):
            for cond_index in QP_dict:
                do_exit = False
                with_handelman = with_handelman_dict[cond_index]
                if with_handelman:
                    zero_conds, l_names, positive_l_names = QP_with_Handelman_dict[cond_index]
                else:
                    zero_conds, l_names, positive_l_names = QP_dict[cond_index]
                all_ls += len(l_names)
                if with_abstraction:
                    for l in l_names:
                        if random.random() <= dropping_probability:
                            set_ls_zero.add(l)
                lst_lines = get_lst_lines_for_z3(zero_conds, cond_index)

                for l in lst_lines:
                    exec(l)
                for l in positive_l_names:
                    line = f"s.add({l} >= 0)\n"
                    exec(line)

        # for a in all_unknown_coefficients:
        #     if random.random() <= dropping_probability:
        #         set_ls_zero.add(str(a))
        #         all_ls += 1

        # now we have the base solver instance
        while True:
            s.push()
            for l in set_ls_zero:
                l = f"s.assert_and_track({l} == 0, '{str(l)}')"
                exec(l)
            # pushed all the l's that are zero
            if args.print_verbose:
                print("Total number of ls being used is {} out of {}".format(
                    all_ls - len(set_ls_zero), all_ls))
                print("Checking the satisfiability...")
            res = s.check()
            if args.print_verbose:
                print("#"*100)
            if (str(res) == "sat"):
                model = s.model()
                input_prog = file_name + ".symbolic"
                output_prog = file_name + ".synth"
                if args.print_verbose:
                    print("Model:", model)
                # update_symbolic_program(input_prog, output_prog, model)
                # print("Synthesized program saved at: ", output_prog)
                return s
            else:
                # print(s)
                core = s.unsat_core()
                if args.print_verbose:
                    print(f"UNSAT core found: {core[:10]}...")
                should_check = True
                unsat = True
                for M in core:
                    if not should_check:
                        break
                    if "l" in str(M) or 'a' in str(M):
                        l_name = str(M)
                        if args.print_verbose:
                            print(f"Relaxing {l_name}...")
                        set_ls_zero.remove(l_name)
                        should_check = False
                        unsat = False
                        break
                refresh = False
                if unsat:
                    for M in core:
                        if "M" in str(M):
                            cond_index, m_cnt = list(
                                map(int, str(M).split("_")[1:]))

                        if args.print_verbose:
                            print(
                                "Constraint Pair: {} - Monomial Number: {} ".format(cond_index, m_cnt))
                        if with_handelman_dict[cond_index]:
                            with_handelman_dict[cond_index] = False
                            refresh = True
                            break
                    if not refresh:
                        raise Exception("Final Result: UNSAT")
                if refresh:
                    if args.print_verbose:
                        print("Reinitializing the solver...")
                        print("Using Handelman for {} constraint pairs...".format(
                            sum(with_handelman_dict)))
                    break
            s.pop()

    return None


def print_qp_for_mathsat(QP_dict, smt_output_filename):

    for i in all_unknown_coefficients:
        line = f"{i} = Real('{i}')"
        exec(line)
    for cond_index in QP_dict:
        l_names = QP_dict[cond_index][1]
        for l in l_names:
            line = f"{l} = Real('{l}')"
            exec(line)

    if args.print_verbose:
        print("#"*100)
        print("Saving the .smt2 for the constraint pair at: ", smt_output_filename)
    s = Solver()
    s.set(unsat_core=True)
    all_ls = 0
    set_ls_zero = set()
    with sp.core.parameters.evaluate(0):
        for cond_index in QP_dict:
            do_exit = False
            zero_conds, l_names, positive_l_names = QP_dict[cond_index]
            all_ls += len(l_names)
            lst_lines = get_lst_lines_for_z3(
                zero_conds, cond_index, with_assert=False)
            for l in lst_lines:
                exec(l)
            for l in positive_l_names:
                line = f"s.add({l} >= 0)\n"
                exec(line)
    # now save the constraints in smt format in a file
    f = open(smt_output_filename, "w")
    f.write("(set-option :produce-models true)")
    f.write(s.to_smt2())
    f.write("(get-model)")

    # args.smt_output = ''
    # args.smt_output += "(set-option :produce-models true)"
    # args.smt_output += str(s.to_smt2())
    # args.smt_output += "(get-model)"


def print_qp_for_ampl():  # does not substitute p
    f = open("ampl-run-{}.mod".format(file_name), "w")
    for i in all_unknown_coefficients:
        f.write(f"{i} = Real('{i}')\n")
    for i in real_vars:
        f.write(f"{i} = Real('{i}')\n")
    for i in range(l_cnt+1):
        f.write(f"l_{i} = Real('l_{i}')\n")
    M_cnt = 0

    f.write("minimize z: 1;\n\n")
    counter_assert = 0
    lst_lines = []
    with sp.core.parameters.evaluate(0):
        for i in QP:
            if i == sp.sympify("True"):
                line = None
                continue
            elif i == sp.sympify("False"):
                # QP is unsat, so break
                line = "s.t. M_0: 0 = 1\n"
                lst_lines = [line]
                break
            elif type(i) == sp.core.relational.GreaterThan:
                line = f"s.t. M_{counter_assert}: {i};\n".replace("**", "^")
                counter_assert += 1
            elif type(i) == sp.core.relational.Equality:
                line = f"s.t. M_{counter_assert}: {i.lhs} = {i.rhs};\n".replace(
                    "**", "^")
                counter_assert += 1
            else:
                raise Exception("Unknown constraint type in QP")
            if line:
                lst_lines.append(line)
    f.writelines(lst_lines)
    f.close()


def print_qp_for_julia():  # does not substitute p
    julia_file_name = file_name + "julia.mod"
    f = open(julia_file_name, "w")
    f.write("""using JuMP
using Ipopt
model = Model(Ipopt.Optimizer)
\n""")
    for i in all_unknown_coefficients:
        f.write(f"@variable(model, {i})\n")

    for i in real_vars:
        f.write(f"@variable(model, {-bound} <= {i} <= {bound})\n")
    for i in range(l_cnt+1):
        f.write(f"@variable(model, l_{i})\n")
    M_cnt = 0

    f.write("\n@NLobjective(model, Min, 1);\n")
    counter_assert = 0
    lst_lines = []
    with sp.core.parameters.evaluate(0):
        for i in QP:
            if i == sp.sympify("True"):
                line = None
                continue
            elif i == sp.sympify("False"):
                line = f"@NLconstraint(model, c0, 0 == 1\n"
                lst_lines = [line]
                break
            elif type(i) == sp.core.relational.GreaterThan:
                # expr = pow_to_mul(i.lhs)
                line = f"@NLconstraint(model, c{counter_assert}, {i})\n".replace(
                    "**", "^")
                counter_assert += 1
            elif type(i) == sp.core.relational.Equality:
                line = f"@NLconstraint(model, c{counter_assert}, {i.lhs} == {i.rhs})\n".replace(
                    "**", "^")
                counter_assert += 1
            else:
                raise Exception("Unknown constraint type in QP")
            if line:
                lst_lines.append(line)
    f.writelines(lst_lines)
    f.write("""print(model)
optimize!(model)
@show termination_status(model)
@show primal_status(model)
@show dual_status(model)
@show objective_value(model)
""")

    for i in all_unknown_coefficients:
        f.write(f"@show value({i})\n")
    f.close()


def print_mat(mat):
    for i in mat:
        print(i)

##### Abstraction Refinement ####


def update_symbolic_program(file_name, output_file, model):
    # Read in the file
    with open(file_name, 'r') as file:
        filedata = file.read()

    # Replace the target string
    for var in model:
        s_var = str(var)
        if s_var.startswith("_a"):  # symbolic variables start with _a
            filedata = filedata.replace(s_var, str(model[var]))

    # Write the file out again
    with open(output_file, 'w') as file:
        file.write(filedata)


def remove_bad_constraint(condition_pairs, var_manager, graph):
    new_constraint = []
    for index in range(len(condition_pairs)):
        i = condition_pairs[index]
        new_pre = []
        post = i[1][0]
        gas_var = sp.Symbol("GAS_0")
        if gas_var not in post.monomial_dict:
            real_vars = i[2]
            # print(real_vars)
            # pdb.set_trace()
            if "GAS" in real_vars:
                del real_vars["GAS"]
            elif "GAS_0" in real_vars:
                del real_vars["GAS_0"]

            for pre in i[0]:
                if gas_var not in pre.monomial_dict:
                    new_pre.append(pre)

            new_constraint.append([new_pre, [post], real_vars])
        else:

            new_constraint.append(i)
    return reduce_constraint_2(new_constraint)


def reduce_constraint_2(constraint):

    
    constraint = remove_zero_constraints(constraint)

    new_constraint = []
    # 1. deep copy and reduce
    #   gas_inv1>=0  ====> gas_inv2>=0
    # 2. remove unncessary purely constant inequality:
    for i, (pres, posts, real_vars) in enumerate(constraint):
        post = posts[0]
        if len(post.monomial_dict) == 1 and sp_unit in post.monomial_dict:
            coef = post.monomial_dict[sp_unit]
            if is_number(str(coef)):
                coef_numeric = float(str(coef))
                assert coef_numeric >= 0, "post is -1>=0, unsat"
                continue  # this constraint is always true
            else:
                pass
        false_in_pre = False

        new_pres = []
        for pre in pres:
            if len(pre.monomial_dict) == 1 and sp_unit in pre.monomial_dict and \
                    is_number(str(pre.monomial_dict[sp_unit])):
                if float(str(pre.monomial_dict[sp_unit])) < 0:
                    false_in_pre = True
                    if args.print_verbose:
                        print('false in pre, constraint delete')
                    break
                else:  # 1>=0, remove from pre

                    pass
            else:
                new_pres.append(copy.deepcopy(pre))
        if false_in_pre:
            continue  # this constraint is always true, can be deleted

        real_vars = copy.deepcopy(real_vars)
        new_post = copy.deepcopy(post)
        gas_var = sp.Symbol("GAS_0")
        real_vars = remove_vars_from_real_vars([gas_var], real_vars)

        new_pres2 = []
        if gas_var in post.monomial_dict:

            for pre in new_pres:
                if gas_var not in pre.monomial_dict:
                    new_pres2.append(pre)
                else:
                    assert gas_var in new_post.monomial_dict, "if gas is in pre, then gas is also in post"

                    new_post = post-pre
                    remove_vars_from_poly([gas_var], new_post)

            new_constraint.append([new_pres2, [new_post], real_vars])
        else:
            new_constraint.append([new_pres, [new_post], real_vars])
    
    # 3. find and remove unncessary single_cond
    #
    #
    constraint_len = len(new_constraint)
    new_constraints2 = []
    for i in range(constraint_len):
        pres, posts, real_vars = new_constraint[i]
        ignore_invariant = False
        # step1: find single-var cond:
        single_conds = {}
        single_var_bounds = {}
        ignore_index = set()
        for i, pre in enumerate(pres):
            is_single_cond, is_to_ignore, (single_var, var_bound) = check_single_cond(
                pre)
            if is_single_cond:
                if is_to_ignore == 1:  # this pre-cond is unncessary
                    # the var will be removed base on single_var_bounds, after considering all pres.
                    ignore_index.add(i)
                elif is_to_ignore == -1:  # this constraint always holds because of false in pre
                    ignore_invariant = True
                    break

                single_conds[i] = single_var
                if single_var not in single_var_bounds:
                    single_var_bounds[single_var] = (None, None)
                single_var_bounds[single_var] = update_single_var_bounds(
                    single_var_bounds[single_var], var_bound)

       

         # remove the single-var pre if the var is constrainted to be [None, None], which has no information -> remove the pre

        for var in single_var_bounds:
            left, right = single_var_bounds[var]
            if left is not None and right is not None and right < left:
                ignore_invariant = True
                break
            elif left is None and right is None:
                for i in single_conds:
                    if single_conds[i] == var:
                        ignore_index.add(i)

        # step2: check necessity
        is_single_var_used = {i: False for i in single_var_bounds}
        for var in single_var_bounds:
            if is_single_var_used[var]:
                continue
            for j, ineq in enumerate(pres + posts):
                if j in single_conds or j in ignore_index:
                    continue
                for mono in ineq.monomial_dict:
                    if str(var) in str(mono):  # b in a*b
                        is_single_var_used[var] = True
                        break
        remove_vars = [
            var for var in is_single_var_used if not is_single_var_used[var]]
        for i in single_conds:
            if not is_single_var_used[single_conds[i]]:
                ignore_index.add(i)
        new_pres = []
        for i, pre in enumerate(pres):
            if i in ignore_index:
                pass
            elif i not in single_conds:
                new_pres.append(remove_vars_from_poly(remove_vars, pre))
            else:
                new_pres.append(remove_vars_from_poly(remove_vars, pre))

        real_vars = remove_vars_from_real_vars(remove_vars, real_vars)
        new_constraints2.append([new_pres, posts, real_vars])

    return process_constraints3(new_constraints2)


def print_constraints(constraint, index=None):
    if not args.print_verbose:
        return
    if index is None:
        index = range(len(constraint))
    for i in index:
        (pres, posts, real_vars) = constraint[i]
        print("======== Constraint %d ==========" % i)
        print("\t real vars = ", real_vars)
        for pre in pres:
            print('\t pre =', pre)
        print('\t\t =====> ', posts[0])
        print('\n')


def remove_vars_from_poly(remove_vars, poly):
    for var in remove_vars:
        if var in poly.monomial_dict:
            del poly.monomial_dict[var]
    return poly


def remove_vars_from_real_vars(remove_vars, real_vars):
    for var in remove_vars:
        var_string = str(var)
        if var in real_vars:
            del real_vars[var]
        if sp.Symbol(var_string+'_0') in real_vars:
            del real_vars[sp.Symbol(var_string+'_0')]
        if var_string.endswith('_0') and sp.Symbol(var_string[:-2]) in real_vars:
            del real_vars[sp.Symbol(var_string[:-2])]
    return real_vars


def update_single_var_bounds(bound1, bound2):
    if bound1[0] is None:
        left = bound2[0]
    elif bound2[0] is None:
        left = bound1[0]
    else:
        left = max(bound1[0], bound2[0])
    if bound1[1] is None:
        right = bound2[1]
    elif bound2[1] is None:
        right = bound1[1]
    else:
        right = min(bound1[1], bound2[1])
    return (left, right)


def check_single_cond(pre: Polynomial):
    default_output = False, 0, (None, None)
    if len(pre.monomial_dict) > 2:
        return default_output
    if len(pre.monomial_dict) == 1:
        key = list(pre.monomial_dict.keys())[0]
        # assert key!=sp_unit, "purely constant inquality should be removed"
        if key == sp_unit:
            return default_output
        coef = str(pre.monomial_dict[key])
        if is_number(coef):
            coef = float(coef)
            if coef > 0:
                return True, 0, (key, [0, None])
            elif coef < 0:
                return True, 0, (key, [None, 0])
            else:
                return True, 1, (key, [None, None])
        else:
            return default_output
    # len == 2
    if sp_unit not in pre.monomial_dict:
        return default_output
    another_var = None
    for k in pre.monomial_dict:
        if k != sp_unit:
            assert another_var is None, 'two other vars'
            another_var = k
    assert another_var is not None, 'empty other var'

    if is_number(str(pre.monomial_dict[another_var])) and is_number(str(pre.monomial_dict[sp_unit])):
        coef_var = float(str(pre.monomial_dict[another_var]))
        coef_cons = float(str(pre.monomial_dict[sp_unit]))
        if coef_var == 0:
            if coef_cons >= 0:
                return True, 1, (another_var, [None, None])
            else:
                return True, -1, (another_var, [None, None])
        elif coef_var > 0:
            return True, 0, (another_var, [-coef_cons/coef_var, None])
        else:
            return True, 0, (another_var, [None, -coef_cons/coef_var])
    else:
        return default_output


def remove_zero_constraints(constraint):
    new_constraints = []
    for i, (pres, posts, real_vars) in enumerate(constraint):
        new_pres = [remove_zero_terms_in_poly(pre) for pre in pres]
        new_posts = [remove_zero_terms_in_poly(post) for post in posts]

        if len(new_posts[0].monomial_dict) >= 1:
            new_constraints.append([new_pres, new_posts, real_vars])
    return new_constraints


def process_constraints3(constraint):
    new_constraints = []
    for i, (pres, posts, real_vars) in enumerate(constraint):
        new_pres = [remove_zero_terms_in_poly(pre) for pre in pres]
        new_posts = [remove_zero_terms_in_poly(post) for post in posts]
        active_mono_set = set()
        for poly in new_pres + new_posts:
            for mono in poly.monomial_dict:
                active_mono_set.add(str(mono))
        new_real_vars = dict()
        for k, v in real_vars.items():
            if k in active_mono_set:
                new_real_vars[k] = v
        new_constraints.append((new_pres, new_posts, new_real_vars))
    new_constraints = remove_empty_pres_in_constraints(new_constraints)
    if args.print_verbose:
        print("\n\n========== Final Constraints ==========")
    if args.print_verbose:
        print_constraints(new_constraints)
    return new_constraints


def remove_empty_pres_in_constraints(constraints):
    new_constraints = []
    for i, (pres, posts, real_vars) in enumerate(constraints):
        new_pres = []
        for pre in pres:
            if len(pre.monomial_dict) >= 1:
                new_pres.append(pre)
        new_constraints.append((new_pres, posts, real_vars))
    return new_constraints


def varname_remove_0(name):
    if name.endswith('_0'):
        return name[:-2]
    else:
        return name


def remove_zero_terms_in_poly(poly: Polynomial):
    new_poly = copy.deepcopy(poly)
    for mono, coef in new_poly.monomial_dict.items():
        if is_number(coef):
            coef_val = float(str(coef))
            if coef_val == 0:
                del poly.monomial_dict[mono]

    return poly


def graph_main(root_dir, file_name, contract_name, function_name, args):
    global all_unknown_coefficients

    graph = None
    args.task_name = '%s_%s_%s%s' % (file_name, contract_name,
                                     function_name.split(
                                         '(')[0] if '(' in function_name else function_name,
                                     '' if args.use_const_gas else str(args.use_const_cnt))
    if not args.apply_handelman:  # linear case
        smtfile = args.task_name + "-putinar-h_{}.smt2".format(args.h_degree)

    else:  # quadratic case
        smtfile = args.task_name + "-handelman-h_{}.smt2".format(args.h_degree)

    if os.path.exists(smtfile):
        print('smt exist')
    else:
        start = time.time()
        graph, global_unknowns, nodes, blocks, var_manager, block_name_to_start_vertex, \
            block_name_to_invariant, block_name_to_gas_function,  block_name_to_pre_condition, \
            start_block_name = main_parser.get_polynomial_ts(
                root_dir, file_name, contract_name, function_name, args)
        graph.print()

        all_unknown_coefficients = global_unknowns
        if args.print_verbose:
            print("Globals: ")
            print(global_unknowns)
        CP = graph.get_constraint_pairs()
        # create new condition pairs for each condition in post
        temp_condition_pair = []
        all_real_vars = dict()
        for (pre, post, real_vars) in CP:
            all_real_vars.update(real_vars)
            for lhs in post:
                temp_condition_pair.append((pre, [lhs], real_vars))
        condition_pairs = temp_condition_pair

        condition_pairs = remove_bad_constraint(
            condition_pairs, var_manager, graph)

        QP_dict, QP_with_Handelman_dict = generate_base_QP_dict(
            condition_pairs, args.h_degree)
        end = time.time()
        print('Generate Polynomial Transition System Elapsed Time: %.4f seconds' % (
            end-start))
        if not args.apply_handelman:  # linear case
            print_qp_for_mathsat(QP_dict, smtfile)
        else:  # quadratic case
            print_qp_for_mathsat(QP_with_Handelman_dict, smtfile)

    start = time.time()
    print("Solving with mathsat")
    modelfile = args.task_name + ".mathsat.model"
    symbolicfile = args.task_name + ".gas"
    os.system("mathsat {filename} > {savefile}".format(
        filename=smtfile, savefile=modelfile))
    print_result = get_concrete(modelfile, symbolicfile)
    if '_a_' in print_result:  # mathsat does not solve
        print(open(modelfile, 'r').read())
        raise Exception("Mathsat UNSAT")
    end = time.time()
    print('Mathsat Solver Elapsed Time: %.4f seconds' % (end-start))
    print(print_result)
    return print_result, graph


def get_concrete_polynomial(poly, solver):
    poly_str_terms = poly.to_string_nonzero()
    model = solver.model()
    new_terms = []
    for term in poly_str_terms:
        is_zero = False
        for var in model:
            s_var = str(var)
            if s_var.startswith("_a"):  # symbolic variables start with _a
                if s_var in term and str(model[var]) == '0':
                    is_zero = True
                    break
                term = term.replace(s_var, str(model[var]))

        if not is_zero:
            new_terms.append(term)
    return ' + '.join(new_terms)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Synthesizer Argument")
    # path to the folder that contains "xxx.sol", "yyy.rbr", "xxx.meta" files
    arg_parser.add_argument('--root_dir', type=str)
    arg_parser.add_argument('--file_name', type=str)    # "xxx.sol"
    # "yyy": name of a smart contract in the solidity file. If "yyy" is a smart contract, then "yyy.rbr" exists
    arg_parser.add_argument('--contract_name', type=str)
    # "zzz": name of a function in a smart contract. The start location (block) in rbr is specified in "xxx.meta" file.
    arg_parser.add_argument('--function_name', type=str)
    # output file that records the synthesized gas bound.
    arg_parser.add_argument('--output_file', type=str, default='syn.out')

    arg_parser.add_argument('--use_quadratic', action='store_true')
    # If set, use h_degree = 2 and degree=2 gas invariants.
    # Else, use h_degree = 1 and degree=1 gas invariants.
    arg_parser.add_argument('--use_const_cnt', type=int, default=0)
    # This is to bound the number of invariant templates for cutset nodes.
    # If use_const_cnt==0, the number of templates is set as cutset_cnt, the number of cutset nodes reachable from the current cutset node (including itself).
    # Else, it is further bounded by the value of use_const_cnt: min(use_const_cnt, cutset_cnt).
    arg_parser.add_argument('--use_const_gas', action='store_true')
    # If use_const_gas, the gas polynomial must be the form of (const0 + const1 * returndatasize).
    # Else the gas polynomial can dependend on other parameters.
    arg_parser.add_argument('--bound_external_call', action='store_true')
    # If bound_external_call, bound the gas cost of every external call to 21000, which is the suggested gasLimit of Ethereum.
    # This option is required when Asparagus cannot synthesize a bound that treats the gas of each external call as a parameter.
    arg_parser.add_argument('--process_and', action='store_true')
    # Normally, an and computation generates a new fresh variable. y = and(x1, x2) => y = new_fresh_{new_unique_id}
    # If process_and, a specific class of and operations will preserve more information.
    # For example in y = and(x, 31), will be processed as y = new_fresh_{new_unique_id} with new constraints: new_fresh_{new_unique_id}>=0 AND new_fresh_{new_unique_id}<=31.
    arg_parser.add_argument('--is_nestedloop', action='store_true')
    # NestedLoop is an illustration example of quadratic gas bound.
    # Quadratic cases are significantly more complicated than linear cases, so we manually provide some additional information:
    # - specify the concrete invariants
    # - bound the set of variables used in gas polynomials
    # We believe that getting this information is another problem othogonal to program synthesis.
    arg_parser.add_argument('--print_verbose', action='store_true')
    # If print_verbose is set True, immediate processing information will be printed to terminal.
    args = arg_parser.parse_args()

    t1 = time.time()
    if True:

        if args.use_quadratic:
            args.h_degree = 2
            args.apply_handelman = True
        else:
            args.h_degree = 1
            args.apply_handelman = False

        if args.apply_handelman and args.print_verbose:
            print("Using handelman...")

        result = {
            'start_time': int(t1),
            'end_time': None,
            'function': args.function_name,
            "contract": args.contract_name,
            "file": args.file_name,
            "gas_estimation": None,
            "error": None,
            "failed": True,
            "timeout": False
        }

        gas_estimation, graph = graph_main(
            args.root_dir, args.file_name, args.contract_name, args.function_name, args)

        result["gas_estimation"] = gas_estimation
        result["failed"] = False
        result["end_time"] = int(time.time())
        pd.DataFrame([result]).to_csv(
            args.output_file,
            mode='a',
            header=not os.path.exists(args.output_file),
            index=False
        )

    t2 = time.time()
    print('Time Used:', t2 - t1)
