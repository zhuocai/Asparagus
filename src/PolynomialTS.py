import copy
import sympy as sp
from polynomial import Polynomial
import sys
import copy
from sympy.matrices import Matrix
import time
import os
from collections import deque

all_unknown_coefficients = []
unknown_coefficients_index = 0


def get_symbolic_poly(symbolic_poly, real_vars):
    global all_unknown_coefficients, unknown_coefficients_index
    if len(symbolic_poly.children) == 2:
        variables = get_vars_name(symbolic_poly.children[0])
        degree = int(symbolic_poly.children[1].children[0].value)
    else:
        variables = []
        degree = int(symbolic_poly.children[0].children[0].value)

    # check all variables are in real_vars
    sympy_vars = []
    for var in variables:
        if var not in real_vars:
            raise Exception(
                "Undefined variable used in symbolic_poly:" + str(symbolic_poly))
        sympy_vars.append(real_vars[var])

    # sympy_vars = list(map(lambda x: real_vars[x], variables))
    curr_monomial_list = [sp_unit]
    all_monomials = set([sp_unit])
    # sympy_vars = list(map(lambda x: real_vars[x], variables))
    curr_degree = 1
    while curr_degree <= degree:
        new_monomial_list = list()
        for var in sympy_vars:
            for monomial in curr_monomial_list:
                new_monomial_list.append(monomial*var)
        all_monomials = all_monomials.union(set(new_monomial_list))
        curr_monomial_list = new_monomial_list
        curr_degree += 1

    monomial_vector = list(all_monomials)
    monomial_dict = {}
    coefficient_list = []
    # create the monomial_dict for the Polynomial
    for mono in monomial_vector:
        unknown_coefficient_name = "_a_" + \
            str(unknown_coefficients_index) + "_"
        sym = sp.Symbol(unknown_coefficient_name)
        unknown_coefficients_index += 1
        coefficient_list.append(sym)
        monomial_dict[mono] = sym

    all_unknown_coefficients.extend(coefficient_list)
    polynomial = Polynomial(monomial_dict)
    return polynomial, polynomial.to_string()


def apply_substitute(lst, substitute_dict):
    return list(map(lambda x: x.subs_complex(substitute_dict), lst))


def get_bound_poly(real_vars, bound):
    sp_bound = Polynomial({sp_unit: bound})
    monomial_dict = {}
    count = 0
    for x in real_vars:
        var = real_vars[x]
        monomial_dict[var*var] = 1
        count += 1
    poly = Polynomial(monomial_dict)
    p_count = Polynomial({sp_unit: count})

    return (sp_bound*sp_bound*p_count - poly)


def get_substition_dict(real_vars, variables_index):
    substitute_dict = {}
    for var in variables_index:
        var_name = var + "_" + str(variables_index[var])
        if var_name not in real_vars:
            real_vars[var_name] = sp.Symbol(var_name)
        substitute_dict[var] = real_vars[var_name]
    return substitute_dict


def get_negation(lst_poly_geq):
    # we overapproximate negation
    return list(map(lambda x: x*Polynomial({sp_unit: -1}), lst_poly_geq))


def graph_preprocessor(tree, file_name):
    """
    The function takes a program tree as input and returns the symbolic
    """
    global real_vars, function_vars
    declarations = tree.children[0]
    precond = tree.children[1]
    stmts = tree.children[2]
    postcond = tree.children[3].children[0]
    # to store each line of the symbolic polynomial program
    symbolic_polynomial_program = []

    # get variable list
    real_vars, function_vars = get_declarations(
        declarations, symbolic_polynomial_program)
    variables_index = {x: 0 for x in real_vars}

    if len(precond.children) == 0:
        _preconditions, precondition_line = [Polynomial({sp_unit: 1})], ""
    else:
        # it is a list, we only store the lhs of the inequalities, rhs is >= 0 by default
        _preconditions, precondition_line = get_assertion(precond, real_vars)
    symbolic_polynomial_program.append("@pre({});".format(precondition_line))

    graph = PolynomialTS(real_vars, file_name)
    root = graph.add_vertex(text=precondition_line)
    graph.set_initial_loc(root, _preconditions)
    # Handle Statements
    parent_vertex = root
    for statement in stmts.children:
        parent_vertex = add_statement_for_graph(
            graph, parent_vertex, statement, symbolic_polynomial_program)
    # Handle Postcondition
    _postconditions, postconditions_line = get_assertion(postcond, real_vars)

    terminal = graph.add_vertex(text=postconditions_line)
    graph.add_edge(parent_vertex, terminal, [], [])
    graph.set_final_loc(terminal, _postconditions)
    symbolic_polynomial_program.append(
        "@post({});".format(postconditions_line))
    # we will add a compact polynomial inequality when we apply stellensatze (if required)
    return graph, symbolic_polynomial_program, all_unknown_coefficients


class PolynomialTS:
    def __init__(self, variables_dict, file_name, var_manager):
        # maps each variable name type(str) to the sp_symbol
        self.variables = variables_dict
        self.var_manager = var_manager
        self.vertices = set()
        self.vertex_text = dict()
        self.edges = dict()
        # list v's such that every cycle passses through at least one v from cutset
        self.cutset = set()
        # v -> list of P(V) such that P(V) \geq 0 whenever control reaches v
        self.invariants = dict()
        self.pre_conditions = dict()
        self.counter = 0
        self.file_name = file_name

    def add_vertex(self, text=""):
        v = self.counter
        self.counter += 1
        self.vertices.add(v)
        self.vertex_text[v] = text
        return v

    def add_edge(self, v1, v2, update, guard, text=""):
        # update = (var, P(V)) for var = P(V) (P(V) is Polynomial type) or None if no update
        # guard  = P(V) which stands for P(V) \geq 0
        if len(update) != 2 and len(update) != 0:
            raise Exception("Invalid update provided: ", update, len(update))
        if v1 not in self.vertices or v2 not in self.vertices:
            raise Exception(
                "Invalid vertex provided for edge: {} - {}".format(v1, v2))
        if (v1, v2) in self.edges:
            raise Exception("Edge already exists")
        self.edges[(v1, v2)] = (update, guard, text)

    def add_cutset(self, v, invariant, pre_condition=[]):
        if v not in self.vertices:
            raise Exception("Vertex {} does not exists.".format(v))
        self.cutset.add(v)
        self.invariants[v] = invariant
        self.pre_conditions[v] = pre_condition

    def set_initial_loc(self, v, invariant, pre_condition=[]):
        if v not in self.vertices:
            raise Exception("Vertex {} does not exists.".format(v))
        self.initial_loc = v
        self.invariants[v] = invariant
        self.pre_conditions[v] = pre_condition

    def set_final_loc(self, v, invariant, pre_condition=[]):
        if v not in self.vertices:
            raise Exception("Vertex {} does not exists.".format(v))
        self.final_loc = v
        self.invariants[v] = invariant
        self.pre_conditions[v] = pre_condition

    def print(self):
        if not self.var_manager.args.print_verbose:
            return
        print("\n===== Polynomial Transition System =====")
        print("Initial Location: ", self.initial_loc)
        print("Final Location: ", self.final_loc)
        print("Vertices:")
        for v in self.vertices:
            if v in self.invariants or v in self.pre_conditions:
                print("==== vertice %s ====" % v)
            if v in self.invariants:
                print("     === Invariant: ", self.invariants[v])
            if v in self.pre_conditions:
                print("     === Other Pre-Conditions: ",
                      self.pre_conditions[v])
        print()
        # print("Edges: ")
        # for e in self.edges:
        # 	print(e, self.edges[e])

    def plot(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        edges = self.edges.keys()
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G)
        plt.figure()
        nx.draw(
            G, pos, edge_color='black', width=1, linewidths=2,
            node_size=500, node_color='pink', alpha=0.9,
            labels={node: node for node in G.nodes()}
        )
        edge_labels = dict()
        for edge in self.edges:
            edge_labels[edge] = self.edges[edge][2][:10]  # the text
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_color='red',
            font_size='8'
        )
        # get positions
        pos = nx.spring_layout(G)
        # shift position a little bit
        shift = [0, 0]
        shifted_pos = {node: node_pos +
                       shift for node, node_pos in pos.items()}
        # Just some text to print in addition to node ids
        labels = {}
        for v in self.vertex_text:
            labels[v] = self.vertex_text[v]

        # nx.draw_networkx_labels(G, shifted_pos, labels=labels, horizontalalignment="left", font_size='8')

        plt.axis('off')
        plt.savefig(self.file_name + "_graph.pdf", block=True)

    def get_constraint_pairs(self):
        CP = []
        for v in self.cutset:
            CP.extend(self.get_constraint_pairs_for_paths_between(
                self.initial_loc, v))
            CP.extend(self.get_constraint_pairs_for_paths_between(
                v, self.final_loc))

        for v1 in self.cutset:
            for v2 in self.cutset:
                CP.extend(self.get_constraint_pairs_for_paths_between(v1, v2))

        # search for path directly from initial to final location
        CP.extend(self.get_constraint_pairs_for_paths_between(
            self.initial_loc, self.final_loc))

        return CP

    def get_constraint_pairs_for_paths_between(self, v1, v2):
        """
        Returns: A list of constraint pairs for each path between v1 and v2
        """
        lst = []
        paths = self.get_paths_between(v1, v2)
        if self.var_manager.args.print_verbose:
            print("\n======== Path of [%d]=>[%d] = %d" % (v1, v2, len(paths)))
        for i, path in enumerate(paths):
            if self.var_manager.args.print_verbose:
                print("\n##### Path: [%d]=>[%d], %d/%d" %
                      (v1, v2, i, len(paths)))
            variables_index = {x: 0 for x in self.variables}
            alpha, beta, real_vars = [], [], copy.copy(self.variables)
            substitute_dict = get_substition_dict(real_vars, variables_index)
            final_values_of_vars = {x: Polynomial(
                {substitute_dict[x]: 1}) for x in self.variables}
            alpha += apply_substitute(self.invariants[v1],
                                      final_values_of_vars)
            alpha += apply_substitute(
                self.pre_conditions[v1], final_values_of_vars)
            old_vertex = v1

            for vert in path[1:]:
                new_vertex = vert
                update, guard, text = self.edges[(old_vertex, new_vertex)]
                guard_subs = apply_substitute(guard, final_values_of_vars)
                alpha.extend(guard_subs)
                if update:
                    (var, poly) = update
                    poly_subs = poly.subs_complex(final_values_of_vars)
                    final_values_of_vars[str(var)] = poly_subs
                old_vertex = new_vertex
            if self.var_manager.args.print_verbose:
                print("\n\tTransitions:\n\t")
            for k, v in final_values_of_vars.items():
                if len(v.monomial_dict.keys()) == 1:
                    term = list(v.monomial_dict.keys())[0]

                    if k in str(term) and v.monomial_dict[term] == 1:
                        continue
                if self.var_manager.args.print_verbose:
                    print('\t\t ', k, '=', v.to_string())

            if self.var_manager.args.print_verbose:
                print("\n\tPre-Guards:\n\t\t",
                      alpha[len(self.invariants[v1]):])
            beta += apply_substitute(self.invariants[v2], final_values_of_vars)
            lst.append((alpha, beta, real_vars))
        return lst

    def get_paths_between_backtrack(self, v1, v2):
        solutions = []
        current_solution = []
        candidate_vertices = deque()
        candidate_vertices.append((1, v1))
        while len(candidate_vertices) >= 1:
            depth, curr_v = candidate_vertices.pop()
            if len(current_solution) >= depth:
                current_solution = current_solution[:depth-1] + [curr_v]
            else:
                current_solution.append(curr_v)

            if curr_v is None:
                # not a solution
                current_solution.pop()
            elif curr_v == v2 and depth > 1:
                # a solution is found
                solutions.append(copy.deepcopy(current_solution))
                current_solution.pop()
            elif depth > 1 and curr_v in self.cutset:
                # not a solution
                current_solution.pop()
            else:
                # expand the node
                for next_v in self.get_neighbors(curr_v):
                    candidate_vertices.append((depth+1, next_v))
        #print('path_of %s => %s = ' % (v1, v2), solutions)
        return solutions

    def get_paths_between(self, v1, v2, visited=None):
        # assume no self loop, i.e., every while block has at least one statement
        return self.get_paths_between_backtrack(v1, v2)

    def get_neighbors(self, v):
        n = []
        for _v in self.vertices:
            if (v, _v) in self.edges:
                n.append(_v)
        return n
