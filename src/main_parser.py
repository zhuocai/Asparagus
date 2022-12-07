"""
rbr, meta => polynomial transition systems
"""
from utils import is_number
import os
import json
import opcodes
from opcodes import GCOST

import numpy as np
import sympy as sp
import copy

from polynomial import Polynomial
from PolynomialTS import *
import cutset

sp_unit = 1

GLOBAL_UNKNOWN_INDEX = 0


def process_rbr_strs(rbrs):

    str_blocks = []

    current_block_name = None
    current_block = None

    for single in rbrs:
        if current_block_name is None:
            # originally outside a block
            if single.startswith('block') or single.startswith('jump'):
                l_pos = single.find('(')
                r_pos = single.rfind(')')
                block_name = single[:l_pos]
                header = single[l_pos+1:r_pos]
                current_block_name = block_name
                current_block = {'name': block_name,
                                 'header': header, 'lines': [], 'next_block': None}
        else:  # current_block_name is not None
            if single == '\n':
                str_blocks.append(current_block)
                current_block_name = None
                current_block = None
            else:
                current_block['lines'].append(
                    single.replace('\t', '').replace('\n', ''))
                if single.startswith('\tcall('):
                    next_block = single[len('\tcall('):]
                    next_block = next_block.split('(')[0]
                    assert current_block['next_block'] is None

                    current_block['next_block'] = next_block

    # from list to map: jumpXXX might appear twice
    block_maps = {}
    for block in str_blocks:

        if block['name'] not in block_maps:
            block_maps[block['name']] = [block]
        else:
            block_maps[block['name']].append(block)
    return block_maps  # block_name => []


def collect_all_vars(block_map, start_block_name):
    # collect variables in block headers

    vars = set()
    visited_block_names = set()
    blocks_to_visit = [start_block_name]

    while len(blocks_to_visit) >= 1:
        block_name = blocks_to_visit.pop()
        visited_block_names.add(block_name)
        blocks = block_map[block_name]
        for block in blocks:
            b_vars = block['header'].replace(' ', '').split(',')
            for v in b_vars:
                if len(v) > 0:
                    vars.add(v)
            next_block = block['next_block']
            if next_block is not None and next_block not in visited_block_names:
                blocks_to_visit.append(next_block)

    vars_list = sorted(list(vars))
    return vars_list, visited_block_names


def var_to_sp_format(var_string):
    return var_string.replace('(', '__').replace(')', '__')


class VarsManager:
    # manages all variable and their sp_format strings, sympy symbols

    def __init__(self, all_vars, args):
        self.old_all_vars = all_vars
        self.all_vars = copy.deepcopy(all_vars)
        self.sp_vars = dict()
        for var in all_vars:
            self.sp_vars[var] = sp.Symbol(var_to_sp_format(var))
        self.is_var_non_neg = dict()
        for var in all_vars:
            self.is_var_non_neg[var] = False

        self.unique_num = 0

        self.other_pre_conds = dict()  # for and(31, xx)
        self.args = args

    def get_unique_name(self):
        self.unique_num += 1
        return "laksjdhfg"+str(self.unique_num)

    def get_unique_fun_name(self, fun_name):
        return "ExternalCall"+str(fun_name)

    def create_new_var(self, var=None):
        old_var = copy.deepcopy(var)
        if var is None or 'fresh()' in var:  
            var = self.get_unique_name()

        assert not is_number(var), "var should not be numeric digits"

        if var not in self.all_vars:
            self.all_vars.append(var)
            self.sp_vars[var] = sp.Symbol(var_to_sp_format(var))
            self.is_var_non_neg[var] = False
            if 'fresh(' in var and var != 'fresh()':
                self.is_var_non_neg[var] = True
        return var, self.sp_vars[var]

    def var_to_sp(self, var):
        if var in self.all_vars:
            return self.sp_vars[var]
        assert False, "var to sp, var not exist"

    def contains_var(self, var):
        return var in self.all_vars

    def var_to_sp_create_if_not_exist(self, var):
        if var in self.all_vars:
            return self.sp_vars[var]

        var_name, var_sp = self.create_new_var(var)

        return var_sp

    def var_to_poly_create_if_not_exist(self, var):
        if is_number(var):
            return Polynomial({sp_unit: int(var)})
        var_sp = self.var_to_sp_create_if_not_exist(var)
        return Polynomial({var_sp: 1})

    def finalize(self):
        self.all_vars_backup = copy.deepcopy(self.all_vars)
        self.sp_vars_backup = copy.deepcopy(self.sp_vars)
        self.is_var_non_neg_backup = copy.deepcopy(self.is_var_non_neg)
        self.all_vars = [var_to_sp_format(v) for v in self.all_vars]
        self.sp_vars = {var_to_sp_format(
            k): v for k, v in self.sp_vars.items()}
        self.non_neg_vars_name_sp_format = {var_to_sp_format(
            k) for k, v in self.is_var_non_neg.items() if v}
        if self.args.print_verbose:
            print('VAR Manager, finalize, is_var_non_neg =  ',
                  self.non_neg_vars_name_sp_format)


def new_zero_polynomial():
    return Polynomial({sp_unit: 0})


# Step 4: Process each single block
def process_block(block, var_manager: VarsManager, external_name=None, args=None):
    local_vars = block['header']
    local_vars_list = local_vars.replace(' ', '').split(
        ',') if len(local_vars) > 0 else []

    active_vars = []
    for v in local_vars_list:
        exclude_set = ['calldataload', 'caller', 'callvalue']
        if args.use_const_cnt <= 1 or args.use_quadratic:
            exclude_set.append('calldatasize')
            exclude_set.append('extcodesize')
            for v_ in local_vars_list:
                if v_.startswith('l__l') or v_.startswith('l('):  # here is v is () format
                    exclude_set.append(v_)

        if v not in exclude_set:
            active_vars.append(var_to_sp_format(v))

    # process block lines:
    instrs = []
    end_instr = None
    i = 0
    while i < len(block['lines']):
        line = block['lines'][i]
        if line.startswith('call'):
            end_instr = [block['lines'][i:]]
            break
        j = i
        while j < len(block['lines']) and not block['lines'][j].startswith('nop('):
            j += 1
        instrs.append(block['lines'][i:j+1])
        i = j+1

    if block['name'].startswith('block'):
        # block has transition, no condition
        # jump has only condition, no transition
        # transition = LinearTransition(all_vars=all_vars)

        gas = 0
        # s__num__
        stack_top = -1
        for v in local_vars:
            if v.startswith('s('):  # s(5)
                stack_top = max(stack_top, int(v[2:-1]))

        trans_list = []
        gas_list = []
        guard_list = []

        geq_list = []
        g_list = []

        last_left_operand = None

        for instr_cnt, instr in enumerate(instrs):
            # instr : ['s(6) = s(2)', 's(2) = s(5)', 's(5) = s(6)', 'nop(SWAP3)']
            op_ins = instr[-1][4:-1]

            gas = None
            
            base_gas = opcodes.get_ins_cost(op_ins)
            extra_gas = new_zero_polynomial()
            if op_ins == 'EXP':
                extra_gas = Polynomial(
                    {sp_unit: opcodes.GCOST['Gexpbyte'] * (1 + 1)})
            elif op_ins in ('LOG0', 'LOG1', 'LOG2', 'LOG3', 'LOG4'):
                # assert False, 'LOG:'+str(instr)
                assert instr_cnt >= 1, "SWAP should come before LOGX"
                prev_instr = instrs[instr_cnt-1]
                assert 'nop(SWAP' in prev_instr[-1], "SWAP should come before LOGX"
                # first is s(n+1) = s(m)
                ll_v = prev_instr[0].replace(' ', '').split('=')[0]
                stack_top = int(ll_v[2:-1])-1
                assert stack_top >= 1
                v_name = 's(%d)' % (stack_top-1)
                extra_gas = Polynomial({var_manager.var_to_sp_create_if_not_exist(
                    v_name): opcodes.GCOST['Glogdata']})
            elif op_ins in ('SHA3', 'KECCAK256'):
                # s(n-1) = sha(s(n), s(n-1))
                tmp = instr[0]
                if op_ins == 'SHA3':
                    tmp2 = tmp[tmp.find('sha3(s(')+len('sha3(s('):]
                else:
                    tmp2 = tmp[tmp.find(
                        'keccak256(s(')+len('keccak256(s('):]
                stack_top = int(tmp2[:tmp2.find(')')])

                if stack_top >= 1:
                    v_name = 's(%d)' % (stack_top-1)
                    v_sp = var_manager.var_to_sp_create_if_not_exist(
                        v_name)
                    extra_gas = Polynomial({v_sp: opcodes.GCOST['Gsha3word']/32,
                                            sp_unit: opcodes.GCOST['Gsha3word']})
                else:
                    if args.print_verbose:
                        print('ERROR in stack size:', instr)
            # ('CALLDATACOPY', 'CODECOPY', 'RETURNDATACOPY'): # WCOPY
            elif op_ins in opcodes.Wcopy:
                assert stack_top >= 2
                v_name = 's(%d)' % (stack_top-2)
                extra_gas = Polynomial({var_manager.var_to_sp_create_if_not_exist(
                    v_name): opcodes.GCOST['Gcopy']/32,
                    sp_unit: opcodes.GCOST['Gcopy']})
                if args.print_verbose:
                    print("COPY usage:", instr)
                if args.print_verbose:
                    print(' extra_gas on: ', v_name)
            elif op_ins in ('EXTCODECOPY'):
                assert False, 'EXTCODECOPY: '+str(instr)
                # does not exist in gastap_dataset
                assert stack_top >= 3
                v_name = 's(%d)' % (stack_top-3)
                extra_gas = Polynomial({var_manager.var_to_sp_create_if_not_exist(
                    v_name): opcodes.GCOST['Gcopy']/32,
                    sp_unit: opcodes.GCOST['Gcopy']})
            # ('CALL', 'CALLCODE', 'DELEGATECALL', 'STATICCALL'):
            elif op_ins in opcodes.Wcall:
                v_name = "s(%d)" % stack_top
                extra_gas = Polynomial({var_manager.var_to_sp_create_if_not_exist(
                    v_name): 1,
                    sp_unit: -GCOST['Gcallstipend']})
                # callvalue is subtracted from Callvalue, and included in the forwarded gas.
                geq_list.append(extra_gas)

            elif op_ins in ('CREATE2'):
                assert False, op_ins+': '+str(instr)
                # does not exist in gastap_dataset
            gas = Polynomial({sp_unit: base_gas}) + extra_gas

            
            trans_list.append(None)
            gas_list.append(gas)
            guard_list.append([])
            if op_ins in opcodes.opcodes:
                stack_top += opcodes.opcodes[op_ins][2] - \
                    opcodes.opcodes[op_ins][1]
            else:
                if args.print_verbose:
                    print(
                        op_ins, 'not in opcodes.opcodes, so that I can not update stack_top')

            for ass in instr[:-1]:

                if op_ins in ("ADD", 'SUB', 'MUL'):
                    # ass = " s(n-1) = s(n) + s(n-1)"
                    n_ins = ass.replace(' ', '').split('=')
                    left = n_ins[0]
                    sign = {'ADD': '+', 'SUB': '-', 'MUL': '*'}[op_ins]
                    r1, r2 = n_ins[1].split(sign)
                    r1_v = var_manager.var_to_sp_create_if_not_exist(r1)
                    r2_v = var_manager.var_to_sp_create_if_not_exist(r2)
                    left_v = var_manager.var_to_sp_create_if_not_exist(left)

                    r1_poly = Polynomial({r1_v: 1})
                    r2_poly = Polynomial({r2_v: 1})
                    if op_ins == 'ADD':
                        transition = (left_v, r1_poly + r2_poly)
                    elif op_ins == 'SUB':
                        transition = (left_v, r1_poly - r2_poly)
                    elif op_ins == 'MUL':
                        transition = (left_v, r1_poly * r2_poly)
                    trans_list.append(transition)
                    gas_list.append(None)
                    guard_list.append([])

                elif op_ins in ('DIV', 'SDIV', 'MOD', 'SMOD',
                                'ADDMOD', 'MULMOD', 'EXP', 'SIGNEXTEND',
                                'LT', 'GT', 'SLT', 'SGT', 'EQ', 'ISZERO',
                                'AND', 'OR', 'XOR', 'NOT', 'BYTE', 'SHA3',
                                'SHL', 'SHR', 'SAR',
                                ) or op_ins.startswith('LOG'):
                    # left = unknown
                    n_ins = ass.replace(' ', '').split('=')
                    left = n_ins[0]
                    left_v = var_manager.var_to_sp_create_if_not_exist(left)
                    guards_step = []
                    if op_ins == 'ISZERO' and instr_cnt >= 1 and instr_cnt+1 < len(instrs) and \
                            instrs[instr_cnt+1][-1][4:-1] == 'MUL' and \
                            instrs[instr_cnt-1][-1][4:-1] == 'DUP2':
                        r_poly = Polynomial({sp_unit: 0})
                    elif args.use_const_cnt >= 2 and op_ins == 'AND':  
                        # use process_and when use_const_cnt>=2 option
                        r_v, r_v_sp = var_manager.create_new_var()
                        r_poly = Polynomial({r_v_sp: 1})
                        try:
                            if instrs[instr_cnt-1][1] == 'nop(PUSH1)':
                                v_ass = instrs[instr_cnt-1][0]
                                ass_val = int(v_ass.replace(
                                    ' ', '').split('=')[1])
                                if ass_val < 258:
                                    poly_to_adds = [Polynomial({sp_unit: ass_val, r_v_sp: -1}),
                                                    Polynomial({r_v_sp: 1})]

                                    if r_v not in var_manager.other_pre_conds:
                                        var_manager.other_pre_conds[r_v] = poly_to_adds
                                    else:
                                        var_manager.other_pre_conds[r_v] += poly_to_adds
                                    if r_v not in active_vars:  
                                        active_vars.append(r_v)

                        except:
                            pass
                    else:
                        r_v, r_v_sp = var_manager.create_new_var()
                        r_poly = Polynomial({r_v_sp: 1})
                    transition = (left_v, r_poly)
                    trans_list.append(transition)
                    gas_list.append(None)
                    guard_list.append(guards_step)

                elif op_ins in ('MSTORE', 'MSTORE8'):
                    n_ins = ass.replace(' ', '').split('=')
                    left = n_ins[0]
                    left_v = var_manager.var_to_sp_create_if_not_exist(left)
                    right = n_ins[1]
                    r_poly = var_manager.var_to_poly_create_if_not_exist(right)

                    if left.startswith('l('):  # known position
                        transition = (left_v, r_poly)
                        trans_list.append(transition)
                        gas_list.append(None)
                        guard_list.append([])
                    else:  # refresh and break
                        vs = var_manager.all_vars.copy()
                        for v in vs:
                            v_sp = var_manager.var_to_sp(v)
                            if v.startswith('l('):
                                new_v, new_v_sp = var_manager.create_new_var()
                                #var_manager.is_var_non_neg[new_v] = True
                                trans_list.append(
                                    (v_sp, Polynomial({new_v_sp: 1})))
                                gas_list.append(None)
                                guard_list.append([])
                        break
                elif op_ins in ('SSTORE'):
                    n_ins = ass.replace(' ', '').split('=')
                    left = n_ins[0]
                    left_v = var_manager.var_to_sp_create_if_not_exist(left)
                    right = n_ins[1]
                    r_poly = var_manager.var_to_poly_create_if_not_exist(right)

                    if left.startswith('gs('):
                        transition = (left_v, r_poly)
                        trans_list.append(transition)
                        gas_list.append(None)
                        guard_list.append([])
                    else:  # fresh and break

                        vs = var_manager.all_vars.copy()
                        for v in vs:
                            v_sp = var_manager.var_to_sp(v)
                            if v.startswith('g('):
                                trans_list.append(
                                    (v_sp, Polynomial({var_manager.create_new_var()[1]: 1})))
                                gas_list.append(None)
                                guard_list.append([])
                        break
                elif op_ins in ('CALL',):
                    n_ins = ass.replace(' ', '').split('=')
                    left = n_ins[0]
                    left_v = var_manager.var_to_sp_create_if_not_exist(left)
                    right = n_ins[1]
                    r_poly = var_manager.var_to_poly_create_if_not_exist(right)
                    transition = (left_v, r_poly)
                    trans_list.append(transition)
                    gas_list.append(None)
                    guard_list.append([])

                    vs = var_manager.all_vars.copy()
                    for v in vs:
                        v_sp = var_manager.var_to_sp(v)
                        if v.startswith('g(') or v.startswith('l('):
                            trans_list.append(
                                (v_sp, Polynomial({var_manager.create_new_var()[1]: 1})))
                            gas_list.append(None)
                            guard_list.append([])

                elif op_ins in ('ADDRESS', 'BALANCE', 'CALLER', 'ORIGIN',
                                'CALLVALUE', 'CALLDATALOAD', 'CALLDATASIZE',
                                'CODESIZE', 'RETURNDATASIZE',
                                'GASPRICE', 'EXTCODESIZE', 'BLOCKHASH',
                                'COINBASE', 'TIMESTAMP', 'NUMBER',
                                'DIFFICULTY', 'GASLIMIT', 'PC', 'MSIZE',
                                'GAS', 'CREATE', "CREATE2",
                                'CHAINID', 'SELFBALANCE', 'EXTCODEHASH', 'MLOAD',
                                'SLOAD',) or op_ins.startswith('PUSH') \
                        or op_ins.startswith('DUP') or op_ins.startswith('SWAP'):
                    # left = right
                    # right might be constant number

                    n_ins = ass.replace(' ', '').split('=')

                    left = n_ins[0]
                    left_v = var_manager.var_to_sp_create_if_not_exist(left)

                    right = n_ins[1]
                    if op_ins == 'GAS' and instr_cnt + 1 < len(instrs) \
                            and instrs[instr_cnt+1][-1][4:-1] == 'CALL':
                        if external_name is None:
                            external_name = var_manager.get_unique_name()
                        right, right_sp = var_manager.create_new_var(
                            var_manager.get_unique_fun_name(external_name))
                        var_manager.is_var_non_neg[right] = True
                        r_poly = var_manager.var_to_poly_create_if_not_exist(
                            right)
                    else:
                        r_poly = var_manager.var_to_poly_create_if_not_exist(
                            right)
                    if op_ins in ("RETURNDATASIZE", "CALLDATASIZE", "EXTCODESIZE"):
                        var_manager.is_var_non_neg[right] = True
                    transition = (left_v, r_poly)
                    trans_list.append(transition)
                    gas_list.append(None)
                    guard_list.append([])
                    last_left_operand = left_v

                    if left.startswith('s(') and left.endswith(')'):

                        stack_top = int(left[2:-1])

                elif op_ins in ('CALLDATACOPY', 'CODECOPY',
                                'RETURNDATACOPY', 'EXTCODECOPY',  'CALLCODE',
                                'DELEGATECALL', 'STATICCALL',
                                'RETURN', 'REVERT', 'SUICIDE', 'POP',):
                    if args.print_verbose:
                        print(op_ins, " should not have assignment => ", n_ins)
                    pass
                elif op_ins in ('STOP', 'PATTERN'):
                    pass
                else:
                    if args.print_verbose:
                        print('unknown opcode:', op_ins)

        if end_instr is not None:
            for op_ins in end_instr[0][1:]:  
                gas_c = opcodes.get_ins_cost(op_ins[4:-1])
                gas_list.append(Polynomial({sp_unit: gas_c}))
                trans_list.append(None)
                guard_list.append([])
        return trans_list, (geq_list, g_list), gas_list, guard_list, active_vars
    else:  # JUMP block
        # [p: Polynomial]
        # [>]
        # -3x + 5 >= 0, -3x >= -5, 3x <= 5
        # "condition: 'neq(s(2), 0)'"
        # return None, block['lines'][0], 0
        line = block['lines'][0]
        l_pos = line.find('(')
        r_pos = line.rfind(')')
        relation = line[:l_pos]
        op1, op2 = line[l_pos + 1:r_pos].replace(' ', '').split(',')
        op1_poly = var_manager.var_to_poly_create_if_not_exist(op1)
        op2_poly = var_manager.var_to_poly_create_if_not_exist(op2)
        geq_list = []
        g_list = []
        if relation == 'lt':
            geq_list.append(op2_poly-op1_poly - Polynomial({sp_unit: 1}))
        elif relation == 'geq':
            geq_list.append(op1_poly - op2_poly)
        elif relation == 'leq':
            geq_list.append(op2_poly-op1_poly)
        elif relation == 'gt':
            geq_list.append(op1_poly-op2_poly - Polynomial({sp_unit: 1}))
        elif relation == 'eq':
            geq_list.append(op1_poly - op2_poly)
            geq_list.append(op2_poly - op1_poly)
        elif relation == 'neq':
            pass
            """cannot represent OR: a>0 OR a<0"""

        return None, (geq_list, g_list), None, None, active_vars


def preprocessor(fun_map, spec, rbrs, function_name, call_fun, args=None):

    # RBR file to block of strings
    rbr_strs_blocks = process_rbr_strs(rbrs)
    fun = ''
    for f in fun_map:
        if function_name in f:
            fun = f
    assert fun != '', "function_name %s does not exist in fun_map %s" % (
        function_name, fun_map)

    start_block_name = 'block' + str(fun_map[fun][0])
    print("Start Block:", start_block_name)
    all_vars, related_blocks = collect_all_vars(
        rbr_strs_blocks, start_block_name)
    related_blocks = sorted(list(related_blocks))
    all_vars = sorted(list(all_vars))
    var_manager = VarsManager(all_vars, args=args)
    non_negative_init_vars = []
    non_negative_always_vars = []

    def remove_underscore(name: str):
        if name.startswith('_'):
            return name[1:]
        else:
            return name
    if fun in spec: 
        init_state = spec[fun]

        for ag, _type in init_state['argument'].items():
            ag = remove_underscore(ag)
            if ag in all_vars:
                "add ag >= 0"
                if _type.endswith('[]'):
                    non_negative_always_vars.append(var_manager.var_to_sp(ag))
                elif _type.startswith('uint'):
                    non_negative_init_vars.append(var_manager.var_to_sp(ag))
                elif 'string' in _type:
                    non_negative_always_vars.append(var_manager.var_to_sp(ag))
                elif _type.startswith('bytes'):
                    if ag == '':  
                        ag = "bytes"
                    non_negative_always_vars.append(var_manager.var_to_sp(ag))
                else:
                    assert False, "type is not uint or string: %s" % _type
                    non_negative_always_vars.append(var_manager.var_to_sp(ag))
            else:
                if args.print_verbose:
                    print("init_state argument %s is unused in vars" % ag)

        for cv, _type in init_state['contract'].items():
            ag = 'g(%s)' % remove_underscore(cv)
            if ag in all_vars:
                "add ag >= 0"
                if _type.endswith('[]'):
                    non_negative_always_vars.append(var_manager.var_to_sp(ag))
                elif _type.startswith('uint'):
                    non_negative_init_vars.append(var_manager.var_to_sp(ag))
                elif 'string' in _type:
                    non_negative_always_vars.append(var_manager.var_to_sp(ag))
                elif _type.startswith('bytes'):
                    if ag == '':  
                        ag = "bytes"
                    non_negative_always_vars.append(var_manager.var_to_sp(ag))
                else:
                    assert False, "type is not uint or string: %s" % _type
                    non_negative_always_vars.append(var_manager.var_to_sp(ag))

    # process each single block:

    nodes = related_blocks  # None block is not added
    blocks = []

    if args.print_verbose:
        print('\n========= RELATED blocks (Connection) =========')

    block_graph = {}
    for b in related_blocks:
        if args.print_verbose:
            print('[', b, ']')
        block_graph[b] = rbr_strs_blocks[b]
        for body in rbr_strs_blocks[b]:
            if args.print_verbose:
                print('\t==>', body['next_block'])
    if args.print_verbose:
        print()
    

    for block_name in related_blocks:
        for block in rbr_strs_blocks[block_name]:
            external_name = None
            if block_name in call_fun:
                external_name = call_fun[block_name]
            trans_list, jump_condition, gas_list, guard_list, active_vars = process_block(
                block, var_manager, external_name, args)

            blocks.append({
                'name': block_name,
                'trans_list': trans_list,
                'gas_list': gas_list,
                'guard_list': guard_list,
                'next_block': block['next_block'],
                'jump_condition': jump_condition,
                'active_vars': active_vars})
    if args.print_verbose:
        print()
        print("====")
    var_manager.finalize()
    if args.print_verbose:
        print('ALL_VARS:', var_manager.all_vars)
        print('non_neg vars:', var_manager.non_neg_vars_name_sp_format)
        print()
    for block in blocks:
        active_vars = block['active_vars']

        for v in var_manager.all_vars:
            if 'ExternalCall' in v and v not in active_vars:
                active_vars.append(v)  # var in sp string format

        for v in var_manager.non_neg_vars_name_sp_format:
            if v not in active_vars:
                if not args.use_quadratic or v not in ['calldatasize', 'extcodesize']:
                    active_vars.append(v)
        for v in var_manager.other_pre_conds:
            active_vars.append(v)

    for v in var_manager.all_vars:
        if ((v.startswith('g__') and v.endswith('__')) or v == 'gas'
                    or (args.use_const_cnt >= 2 and v.startswith('l__l') and v.endswith('__'))
                ) and var_manager.sp_vars[v] not in non_negative_always_vars:  # array length >=0 added here
            non_negative_always_vars.append(var_manager.sp_vars[v])

    return (non_negative_init_vars, non_negative_always_vars, nodes, blocks, start_block_name, var_manager)


def get_polynomial_ts(root_dir, file_name, contract_name, function_name, args):
    """
    Input: name: The name of the example
    Output: 
    pts: PolynomialTS
    global_uknowns : List of all unknows
    nodes: list of name of blocks
    blocks: blocks
    var_manager: VarManager
    block_name_to_start_vertex: dictionary maps block name to the start node in pts
    block_name_to_invariant: dictionary maps block name to list of invariants >= 0
    block_name_to_gas_function: dictionary maps block name to a single polynomial function for gas
    """

    """
    fun_map: {'func3(uint256)': (68, 75)} func3() starts at block68
    spec: {'func3(uint256)': {'argument': {'k': '>=0'}, 'contract': {'j': '>=0'}}}
        in func3, argument k>=0 (uint), contract variable j>=0 (uint)
    """

    task_path = os.path.join(root_dir, args.task_name)
    meta_path = os.path.join(root_dir, file_name.replace(".sol", ".meta"))
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    fun_map = meta[contract_name]['function_block_mapping']
    spec = meta[contract_name]['initial_state']
    call_fun = meta[contract_name]['block_CALL_mapping']

    with open(os.path.join(root_dir, contract_name + '.rbr'), 'r') as f:
        rbrs = f.readlines()

    non_negative_init_vars, non_negative_always_vars, nodes, blocks, \
        start_block_name, var_manager = preprocessor(
            fun_map, spec, rbrs, function_name, call_fun, args=args)

    pts = PolynomialTS(var_manager.sp_vars, task_path, var_manager)
    block_name_to_start_vertex = dict()

    for block_name in nodes:
        v = pts.add_vertex(text=block_name)
        block_name_to_start_vertex[block_name] = v

    # cutset
    nodes_ = list(nodes) + ['DEAD']
    cutset_graph = cutset.CutsetGraph(
        nodes=nodes_, start_node=start_block_name, end_node='DEAD')
    block_name2active_vars = {}
    for block in blocks:
        from_name = block['name']
        block_name2active_vars[from_name] = block['active_vars']
        to_name = block['next_block']
        if to_name is None:
            to_name = 'DEAD'
        cutset_graph.add_edge(from_name, to_name)
    cutset_nodes = cutset_graph.get_cut_set()

    cutset_nodes = [(name.replace('jump', 'block'), cnt)
                    for name, cnt in cutset_nodes]
    if args.print_verbose:
        print('CUTSET_NODES=', cutset_nodes)

    v = pts.add_vertex(text="DEAD")
    block_name_to_start_vertex["DEAD"] = v

    GAS_VAR_NAME = "GAS"
    assert GAS_VAR_NAME not in var_manager.sp_vars
    GAS_VAR = var_manager.var_to_sp_create_if_not_exist(GAS_VAR_NAME)

    for i in range(len(blocks)):
        block = blocks[i]
        if "jump" in block["name"]:
            next_block = block["next_block"]
            v = block_name_to_start_vertex[block["name"]]
            next_block_vertex = block_name_to_start_vertex[next_block]
            geq_list, g_list = block["jump_condition"]
            condition = geq_list + g_list
            pts.add_edge(v, next_block_vertex, [],
                         condition, text=block["name"])
            continue

        start_vertex = block_name_to_start_vertex[block["name"]]
        trans_list = block["trans_list"]
        gas_list = block["gas_list"]
        guard_list = block['guard_list']
        v = start_vertex
        for t in range(len(trans_list)):
            trans = trans_list[t]
            gas = gas_list[t]
            guards = guard_list[t]
            if len(guards) >= 1:
                v_ = pts.add_vertex()
                pts.add_edge(v, v_, [], guards, text="trans_guard")
                v = v_
            if gas:
                v_ = pts.add_vertex()
                gas_var_poly = Polynomial({GAS_VAR: 1})
                pts.add_edge(v, v_, (GAS_VAR, gas_var_poly - gas),
                             [], text="G--")
                v = v_

            if trans:
                v_ = pts.add_vertex()
                pts.add_edge(v, v_, trans, [], text=str(trans))
                v = v_

        next_block = block["next_block"]
        if next_block is None:
            next_block = "DEAD"
        next_block_vertex = block_name_to_start_vertex[next_block]

        pts.add_edge(v, next_block_vertex, [], [])

    if args.is_nestedloop:
        block155_invariant = [
            Polynomial({var_manager.sp_vars['s__2__']:1,
                        var_manager.sp_vars['s__6__']:-1}),
            Polynomial({var_manager.sp_vars['s__3__']:1,
                        var_manager.sp_vars['s__7__']:-1}),
            Polynomial({var_manager.sp_vars['s__6__']:1}),
            Polynomial({var_manager.sp_vars['s__7__']:1}),
            # Polynomial({var_manager.sp_vars['s__2__']:1}),
            # Polynomial({var_manager.sp_vars['s__3__']:1}),

        ]
        block167_invariant = [
            Polynomial({var_manager.sp_vars['s__2__']:1,
                        var_manager.sp_vars['s__6__']:-1,
                        sp_unit:-1}),
            Polynomial({var_manager.sp_vars['s__3__']:1,
                        var_manager.sp_vars['s__7__']:-1}),
            Polynomial({var_manager.sp_vars['s__6__']:1}),
            Polynomial({var_manager.sp_vars['s__7__']:1}),

            # Polynomial({var_manager.sp_vars['s__2__']:1}),
            # Polynomial({var_manager.sp_vars['s__3__']:1}),
        ]
        nested_loop_invariant = {
            'block155': block155_invariant, 'block167': block167_invariant}

    # Create invariants and gas functions
    block_name_to_invariant = dict()
    block_name_to_gas_function = dict()
    block_name_to_pre_condition = dict()
    variables = var_manager.all_vars

    # gas polynomial does not have GAS_VAR as a variables
    variables.remove(GAS_VAR_NAME)
    global_unknowns = list()
    gas_var_poly = Polynomial({GAS_VAR: 1})

    gas_exprs = []

    for node_name, cnt in cutset_nodes:
        if args.print_verbose:
            print("\n===CutsetNode -", node_name)
        node = block_name_to_start_vertex[node_name]
        variables = block_name2active_vars[node_name]
        if args.is_nestedloop:
            invariant = nested_loop_invariant[node_name]
        else: 
            inv_num = cnt
            if args.use_quadratic:
            # for quadratic cases, we need more invariants
                inv_num = cnt*2
            if args.use_const_cnt > 0:
                inv_num = min(inv_num, args.use_const_cnt)
            invariant = get_invariant(variables, var_manager, global_unknowns, [1] * inv_num)
        pre_condition = []

        for k, v in var_manager.sp_vars.items():
            if v in non_negative_always_vars or k in var_manager.non_neg_vars_name_sp_format:
                pre_condition.append(Polynomial({v: 1}))
            if args.bound_external_call and k.startswith('ExternalCall'):
                pre_condition.append(Polynomial({v: -1, sp_unit: 21000}))

        for k, v in var_manager.other_pre_conds.items():
            pre_condition += v

        if args.use_quadratic:
            # gas_variables will be limited
            gas_variables = variables
            if args.is_nestedloop:
                gas_variables = ['s__2__', 's__3__', 's__6__', 's__7__']
            gas_polynomial = get_invariant(
                gas_variables, var_manager, global_unknowns, [2, ])[0]
        else:
            gas_polynomial = get_invariant(variables, var_manager, global_unknowns, [1, ])[
                0] 
        # gas_exprs.append(node_name + ": gas = " +
        #                  str(gas_polynomial.as_expr()))
        gas_invariant = [gas_var_poly - gas_polynomial]
        block_name_to_invariant[node_name] = invariant
        block_name_to_gas_function[node_name] = gas_polynomial
        block_name_to_pre_condition[node_name] = pre_condition
        pts.add_cutset(node, invariant + gas_invariant, pre_condition)

    start_invariant = list()
    start_pre_condition = []
    for k, v in var_manager.sp_vars.items():
        if v in non_negative_init_vars or v in non_negative_always_vars or k in var_manager.non_neg_vars_name_sp_format:
            start_pre_condition.append(Polynomial({v: 1}))

    start_vertex = block_name_to_start_vertex[start_block_name]
    variables = block_name2active_vars[start_block_name]

    def is_stack(v):  # also exclude temporary laks..
        if 'laksjdhfg' in v:
            return True
        if not v.startswith('s__') or not v.endswith('__'):
            return False
        try:
            v_int = int(v[3:-2])
        except:
            return False
        return True

    gas_variables = [v for v in variables if not is_stack(v)]

    if args.use_const_cnt <= 1:  # remove fresh, memory var, cds, eds
        gas_variables = [var for var in gas_variables if 'fresh' not in var and 'l__l' not in var and var not in (
            'calldatasize', 'extcodesize')]
    else:
        gas_variables = gas_variables
    if args.use_const_gas:
        small_variables = [v for v in gas_variables if v == 'returndatasize']
        gas_polynomial = get_invariant(
            small_variables, var_manager, global_unknowns, [1, ])[0]
    elif args.use_quadratic:
        gas_polynomial = get_invariant(
            gas_variables, var_manager, global_unknowns, [2, ])[0]
    else:
        gas_polynomial = get_invariant(
            gas_variables, var_manager, global_unknowns, [1, ])[0]
    gas_invariant = [gas_var_poly - gas_polynomial]
    if args.print_verbose:
        print("\nStart Vertex=====\n\tGAS polynomial: ", gas_polynomial)

    with open(args.task_name + ".gas", "w") as f:   ## save the start gas formula to .gas file
        new_gas_expr = "gas = " + str(gas_polynomial.as_expr())
        f.write(new_gas_expr)

    block_name_to_invariant[start_block_name] = start_invariant
    block_name_to_gas_function[start_block_name] = gas_polynomial
    block_name_to_pre_condition[start_block_name] = start_pre_condition
    pts.set_initial_loc(start_vertex, start_invariant +
                        gas_invariant, start_pre_condition)

    end_vertex = block_name_to_start_vertex["DEAD"]
    pts.set_final_loc(end_vertex, [gas_var_poly], [])

    return pts, global_unknowns, nodes, blocks, var_manager, block_name_to_start_vertex, \
        block_name_to_invariant, block_name_to_gas_function, block_name_to_pre_condition, \
        start_block_name


def get_symbolic_polynomial(variables, degree, var_manager, global_unknowns):
    """
    variables : list of variable names
    degree : degree of the symbolic polynomial
    var_manager : VarManager 
    """
    global GLOBAL_UNKNOWN_INDEX
    sympy_vars = list()
    for var in variables:
        if not var_manager.contains_var(var):
            raise Exception("Undefined variable passed:" + str(variables))
        sympy_vars.append(var_manager.var_to_sp(var))

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
        unknown_coefficient_name = "_a_" + str(GLOBAL_UNKNOWN_INDEX) + "_"
        sym = sp.Symbol(unknown_coefficient_name)
        GLOBAL_UNKNOWN_INDEX += 1
        coefficient_list.append(sym)
        monomial_dict[mono] = sym

    global_unknowns.extend(coefficient_list)
    polynomial = Polynomial(monomial_dict)
    return polynomial


def get_invariant(variables, var_manager, global_unknowns, template):
    """
    Template is list of degrees of polynomials that will be used in invariant.
    E.g., template = [2,1,1] means polynomial of degree 2, 1 and 1 will be used
    """
    invariant = list()
    for degree in template:
        new_inv = get_symbolic_polynomial(
            variables, degree, var_manager, global_unknowns)
        invariant.append(new_inv)
    return invariant
