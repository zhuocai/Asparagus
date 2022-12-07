# based on ETH yellow paper of 2020-05-29
# Same as GASTAP

# list of all opcodes except the PUSHi and DUPi
# opcodes[name] has a list of [value (index), no. of items removed from stack, no. of items added to stack]
opcodes = {
    "STOP": [0x00, 0, 0],
    "ADD": [0x01, 2, 1],
    "MUL": [0x02, 2, 1],
    "SUB": [0x03, 2, 1],
    "DIV": [0x04, 2, 1],
    "SDIV": [0x05, 2, 1],
    "MOD": [0x06, 2, 1],
    "SMOD": [0x07, 2, 1],
    "ADDMOD": [0x08, 3, 1],
    "MULMOD": [0x09, 3, 1],
    "EXP": [0x0a, 2, 1],
    "SIGNEXTEND": [0x0b, 2, 1],

    "LT": [0x10, 2, 1],
    "GT": [0x11, 2, 1],
    "SLT": [0x12, 2, 1],
    "SGT": [0x13, 2, 1],
    "EQ": [0x14, 2, 1],
    "ISZERO": [0x15, 1, 1],
    "AND": [0x16, 2, 1],
    "OR": [0x17, 2, 1],
    "XOR": [0x18, 2, 1],
    "NOT": [0x19, 1, 1],
    "BYTE": [0x1a, 2, 1],
    "SHL": [0x1b, 2, 1],
    "SHR": [0x1c, 2, 1],
    "SAR": [0x1d, 2, 1],

    "SHA3": [0x20, 2, 1],
    # "KECCAK256": [0x20, 2, 1], in 2022, not in 2020

    "ADDRESS": [0x30, 0, 1],
    "BALANCE": [0x31, 1, 1],
    "ORIGIN": [0x32, 0, 1],
    "CALLER": [0x33, 0, 1],
    "CALLVALUE": [0x34, 0, 1],
    "CALLDATALOAD": [0x35, 1, 1],
    "CALLDATASIZE": [0x36, 0, 1],
    "CALLDATACOPY": [0x37, 3, 0],
    "CODESIZE": [0x38, 0, 1],
    "CODECOPY": [0x39, 3, 0],
    "GASPRICE": [0x3a, 0, 1],
    "EXTCODESIZE": [0x3b, 1, 1],
    "EXTCODECOPY": [0x3c, 4, 0],
    # "MCOPY": [0x3d, 3, 0], in 2022, not in 2020
    "RETURNDATASIZE": [0x3d, 0, 1],
    "RETURNDATACOPY": [0x3e, 3, 0],
    "EXTCODEHASH": [0x3f, 1, 1],

    "BLOCKHASH": [0x40, 1, 1],
    "COINBASE": [0x41, 0, 1],
    "TIMESTAMP": [0x42, 0, 1],
    "NUMBER": [0x43, 0, 1],
    "DIFFICULTY": [0x44, 0, 1],
    "GASLIMIT": [0x45, 0, 1],
    "CHAINID": [0x46, 0, 1],  # not in 2020 ?
    "SELFBALANCE": [0x47, 0, 1],  # not in 2020 ?


    "POP": [0x50, 1, 0],
    "MLOAD": [0x51, 1, 1],
    "MSTORE": [0x52, 2, 0],
    "MSTORE8": [0x53, 2, 0],
    "SLOAD": [0x54, 1, 1],
    "SSTORE": [0x55, 2, 0],
    "JUMP": [0x56, 1, 0],
    "JUMPI": [0x57, 2, 0],
    "PC": [0x58, 0, 1],
    "MSIZE": [0x59, 0, 1],
    "GAS": [0x5a, 0, 1],
    "JUMPDEST": [0x5b, 0, 0],
    # "SLOADEXT": [0x5c, 2, 1], # not in 2020
    # "SSTOREEXT": [0x5d, 3, 0], # not in 2020
    # "SLOADBYTESEXT": [0x5c, 4, 0], # not in 2020
    # "SSTOREBYTESEXT": [0x5d, 4, 0], # not in 2020

    'PUSH1': [0x60, 0, 1],
    'PUSH2': [0x61, 0, 1],
    'PUSH3': [0x62, 0, 1],
    'PUSH4': [0x63, 0, 1],
    'PUSH5': [0x64, 0, 1],
    'PUSH6': [0x65, 0, 1],
    'PUSH7': [0x66, 0, 1],
    'PUSH8': [0x67, 0, 1],
    'PUSH9': [0x68, 0, 1],
    'PUSH10': [0x69, 0, 1],
    'PUSH11': [0x6A, 0, 1],
    'PUSH12': [0x6B, 0, 1],
    'PUSH13': [0x6C, 0, 1],
    'PUSH14': [0x6D, 0, 1],
    'PUSH15': [0x6E, 0, 1],
    'PUSH16': [0x6F, 0, 1],
    'PUSH17': [0x70, 0, 1],
    'PUSH18': [0x71, 0, 1],
    'PUSH19': [0x72, 0, 1],
    'PUSH20': [0x73, 0, 1],
    'PUSH21': [0x74, 0, 1],
    'PUSH22': [0x75, 0, 1],
    'PUSH23': [0x76, 0, 1],
    'PUSH24': [0x77, 0, 1],
    'PUSH25': [0x78, 0, 1],
    'PUSH26': [0x79, 0, 1],
    'PUSH27': [0x7A, 0, 1],
    'PUSH28': [0x7B, 0, 1],
    'PUSH29': [0x7C, 0, 1],
    'PUSH30': [0x7D, 0, 1],
    'PUSH31': [0x7E, 0, 1],
    'PUSH32': [0x7F, 0, 1],

    'DUP1': [0x80, 1, 2],
    'DUP2': [0x81, 1, 2],
    'DUP3': [0x82, 1, 2],
    'DUP4': [0x83, 1, 2],
    'DUP5': [0x84, 1, 2],
    'DUP6': [0x85, 1, 2],
    'DUP7': [0x86, 1, 2],
    'DUP8': [0x87, 1, 2],
    'DUP9': [0x88, 1, 2],
    'DUP10': [0x89, 1, 2],
    'DUP11': [0x8A, 1, 2],
    'DUP12': [0x8B, 1, 2],
    'DUP13': [0x8C, 1, 2],
    'DUP14': [0x8D, 1, 2],
    'DUP15': [0x8E, 1, 2],
    'DUP16': [0x8F, 1, 2],

    'SWAP1': [0x90, 0, 0],
    'SWAP2': [0x91, 0, 0],
    'SWAP3': [0x92, 0, 0],
    'SWAP4': [0x93, 0, 0],
    'SWAP5': [0x94, 0, 0],
    'SWAP6': [0x95, 0, 0],
    'SWAP7': [0x96, 0, 0],
    'SWAP8': [0x97, 0, 0],
    'SWAP9': [0x98, 0, 0],
    'SWAP10': [0x99, 0, 0],
    'SWAP11': [0x9A, 0, 0],
    'SWAP12': [0x9B, 0, 0],
    'SWAP13': [0x9C, 0, 0],
    'SWAP14': [0x9D, 0, 0],
    'SWAP15': [0x9E, 0, 0],
    'SWAP16': [0x9F, 0, 0],

    "LOG0": [0xa0, 2, 0],
    "LOG1": [0xa1, 3, 0],
    "LOG2": [0xa2, 4, 0],
    "LOG3": [0xa3, 5, 0],
    "LOG4": [0xa4, 6, 0],

    "CREATE": [0xf0, 3, 1],
    "CALL": [0xf1, 7, 1],
    "CALLCODE": [0xf2, 7, 1],
    "RETURN": [0xf3, 2, 0],
    "DELEGATECALL": [0xf4, 6, 1],
    # "BREAKPOINT": [0xf5, 0, 0], not in 2020
    'CREATE2': [0xf5, 4, 1],
    # "RNGSEED": [0xf6, 1, 1], not in 2020
    # "SSIZEEXT": [0xf7, 2, 1], not in 2020
    # "SLOADBYTES": [0xf8, 3, 0], not in 2020
    # "SSTOREBYTES": [0xf9, 3, 0], not in 2020
    "STATICCALL": [0xfa, 6, 1],
    # "SSIZE": [0xfa, 1, 1], not in 2020
    # "STATEROOT": [0xfb, 1, 1], not in 2020
    # "TXEXECGAS": [0xfc, 0, 1], not in 2020
    "REVERT": [0xfd, 2, 0],
    # "CALLSTATIC": [0xfd, 7, 1], not in 2020
    "ASSERTFAIL": [0xfe, 0, 0],
    "INVALID": [0xfe, 0, 0],
    # "SUICIDE": [0xff, 1, 0], not in 2020
    "SELFDESTRUCT": [0xff, 1, 0],
    "---END---": [0x00, 0, 0],  # just for fun, by EthIR


}

# TO BE UPDATED IF ETHEREUM VM CHANGES their fee structure

GCOST = {
    "Gzero": 0,
    "Gjumpdest": 1,
    "Gbase": 2,
    "Gverylow": 3,
    "Glow": 5,
    "Gmid": 8,
    "Ghigh": 10,
    # In 2022, not in 2020: "Gwarmaccess": 100,
    # In 2022, not in 2020: "Gaccesslistaddress": 2400,
    # In 2022, not in 2020: "Gaccessliststorage": 1900,
    # In 2022, not in 2020: "Gcoldaccountaccess": 2600,
    # In 2022, not in 2020: "Gcoldsload": 2100,

    "Gextcode": 700,  # Not in 2022, in 2020
    "Gextcodehash": 400,  # Not in 2022, in 2020
    "Gbalance": 400,  # Not in 2022, in 2020
    "Gsload": 200,  # Not in 2022, in 2020

    "Gsset": 20000,
    "Gsreset": 5000,  # In 2022, it is 2900
    'Rsclear': 15000,  # In 2020,

    "Rselfdestruct": 24000,
    "Gselfdestruct": 5000,
    "Gcreate": 32000,
    "Gcodedeposit": 200,

    "Gcall": 700,  # Not in 2022, in 2020. Paid for a CALL operation.
    # Paid for a non-zero value transfer as part of the CALL operation.
    "Gcallvalue": 9000,
    "Gcallstipend": 2300,  # A stipend for the called constract, subtracted from Gcallvalue
    "Gnewaccount": 25000,  # CALL or SELFDESTRUCT that creates an account
    "Gexp": 10,
    "Gexpbyte": 50,
    "Gmemory": 3,
    "Gtxcreate": 32000,
    "Gtxdatazero": 4,  # paid for every zero byte of data or code for a transaction
    "Gtxdatanonzero": 68,  # In 2022: =16, in 2020: =68
    # paid for every non-zero byte of data or code for a transaction

    "Gtransaction": 21000,
    "Glog": 375,
    "Glogdata": 8,
    "Glogtopic": 375,
    "Gsha3": 30,
    # "Gkeccak256": 30, In 2022, not in 2020 (sha3).
    "Gsha3word": 6,
    # "Gkeccak256word": 6, In 2022, not in 2020 (sha3word).
    "Gcopy": 3,
    "Gblockhash": 20,
    # Not in 2022, In 2020: for exp-over-modulo precompiled contract.
    "Gquaddivisor": 20

}

# 2022: Wzero = ("STOP", "RETURN", "REVERT", "ASSERTFAIL")
Wzero = ("STOP", "RETURN", "REVERT", "ASSERTFAIL")


# 2022-same: Wbase = ("ADDRESS", "ORIGIN", "CALLER", "CALLVALUE", "CALLDATASIZE",
#          "CODESIZE", "GASPRICE", "COINBASE", "TIMESTAMP", "NUMBER",
#          "DIFFICULTY", "GASLIMIT", "POP", "PC", "MSIZE", "GAS","CHAINID",
#          "RETURNDATASIZE")

Wbase = ("ADDRESS", "ORIGIN", "CALLER", "CALLVALUE", "CALLDATASIZE",
         "CODESIZE", "GASPRICE", "COINBASE", "TIMESTAMP", "NUMBER",
         "DIFFICULTY", "GASLIMIT", "POP", "PC", "MSIZE", "GAS", "CHAINID",
         "RETURNDATASIZE")

Wverylow = ("ADD", "SUB", "NOT", "LT", "GT", "SLT", "SGT", "EQ",
            "ISZERO", "AND", "OR", "XOR", "BYTE", "CALLDATALOAD",
            "MLOAD", "MSTORE", "MSTORE8", "PUSH", "DUP", "SWAP",
            "SHL", "SHR", "SAR")

# 2022: Wlow = ("MUL", "DIV", "SDIV", "MOD", "SMOD", "SIGNEXTEND","SELFBALANCE")
Wlow = ("MUL", "DIV", "SDIV", "MOD", "SMOD", "SIGNEXTEND")

Wmid = ("ADDMOD", "MULMOD", "JUMP")

Whigh = ("JUMPI")

Wcopy = ("CALLDATACOPY", "CODECOPY", "RETURNDATACOPY")
# 2022: Wcall = ("CALL", "CALLCODE", "DELEGATECALL", "STATICCALL")
Wcall = ("CALL", "CALLCODE", "DELEGATECALL", "STATICCALL")


# Wextaccount = ("BALANCE", "EXTCODESIZE", "EXTCODEHASH") in 2022, not in 2020

def get_opcode(opcode):
    if opcode in opcodes:
        return opcodes[opcode]

    # check PUSHi
    for i in range(32):
        if opcode == 'PUSH' + str(i + 1):
            return [hex(0x60 + i), 0, 1]

    # check DUPi
    for i in range(16):
        if opcode == 'DUP' + str(i + 1):
            return [hex(0x80 + i), i + 1, i + 2]

    # check SWAPi
    for i in range(16):
        if opcode == 'SWAP' + str(i + 1):
            return [hex(0x90 + i), i + 2, i + 2]
    raise ValueError('Bad Opcode ' + opcode)


def get_ins_cost(opcode):
    if opcode == "PATTERN":  # Nov 24
        actual_opcodes = ["PUSH1", "DUP2", "PUSH1", "AND", "ISZERO",
                          "PUSH2", "MUL", "SUB", "AND", "PUSH1", "SWAP1", "DIV"]
        return sum(get_ins_cost(opc) for opc in actual_opcodes)

    elif opcode == "SSTORE":
        # 2022: return max(GCOST['Gwarmaccess'], GCOST['Gsset'], GCOST['Gsreset']) + max(0, GCOST['Gcoldsload'])
        return max(GCOST['Gsset'], GCOST['Gsreset'])
    elif opcode == "EXP":  # further dependency
        return GCOST["Gexp"]
    elif opcode in Wcopy:  # further dependency
        return GCOST['Gverylow']
    elif opcode == "EXTCODECOPY":  # further dependency for EXTCODECOPY
        return max(GCOST['Gextcode'])
    # not in 2022
    # elif opcode in Wextaccount:
    #     return max(GCOST['Gwarmaccess'], GCOST['Gcoldaccountaccess'])
    elif opcode in ("LOG0", "LOG1", "LOG2", "LOG3", "LOG4"):  # further dependency
        num_topics = int(opcode[3:])
        return GCOST["Glog"] + num_topics * GCOST["Glogtopic"]

    # Not in 2020
    # elif opcode in Wcall: # further dependency, directly on s(0) = gas forwarded
    #     ##
    #     ## specified gas + extra gas
    #     ## extra_gas = Access(<=Gcoldaccountaccess=2600) + XFER(<=Gcallvalue=9000) + Newaccount (assume not, 0)
    #     ##
    #     return max(GCOST['Gwarmaccess'], GCOST['Gcoldaccountaccess']) + \
    #         max(0, GCOST['Gcallvalue']) # + max(0, GCOST['Gnewaccount'])

    elif opcode in Wcall:
        # assume always transfer value, cannot create new accounts.
        return GCOST['Gcall'] + GCOST['Gcallvalue']

    # Not in 2020
    # elif opcode in ("SUICIDE", 'SELFDESTRUCT'):
    #     return GCOST['Gselfdestruct'] + max(0, GCOST['Gcoldaccountaccess']) + max(0, GCOST['Gnewaccount'])
    elif opcode in ("SELFDESTRUCT", ):
        # cannot handle newaccount, assume always create a new account.
        return GCOST['Gselfdestruct'] + max(0, GCOST['Gnewaccount'])

    elif opcode in ("CREATE", "CREATE2"):  # further dependency for CREATE2
        return GCOST['Gcreate']
    elif opcode in ("SHA3",):  # further dependency
        return GCOST["Gsha3"]
    elif opcode == "JUMPDEST":
        return GCOST["Gjumpdest"]
    elif opcode == "SLOAD":
        return GCOST["Gsload"]  # in 2020
        # return max(GCOST['Gwarmaccess'], GCOST['Gcoldsload']) in 2022

    elif opcode in Wzero:
        return GCOST["Gzero"]
    elif opcode in Wbase:
        return GCOST["Gbase"]
    elif opcode in Wverylow or opcode.startswith("PUSH") or opcode.startswith("DUP") or opcode.startswith("SWAP"):
        return GCOST["Gverylow"]
    elif opcode in Wlow:
        return GCOST["Glow"]
    elif opcode in Wmid:
        return GCOST["Gmid"]
    elif opcode in Whigh:
        return GCOST["Ghigh"]
    elif opcode in ('EXTCODESIZE', ):
        return GCOST["Gextcode"]
    elif opcode in ('EXTCODEHASH', ):
        return GCOST['Gextcodehash']
    elif opcode == "BALANCE":
        return GCOST["Gbalance"]
    elif opcode == "BLOCKHASH":
        return GCOST["Gblockhash"]

    return 0
