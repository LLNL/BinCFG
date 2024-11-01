import re
import os
import traceback
from enum import Enum
from bincfg.normalization.base_tokenizer import Tokens


def java_check_assembly_rules(token_list, newline_tup):
    """Checks the given tokenized java line is valid
    
    Java bytecode is a lot easier than x86...

    Most instructions are just the opcode, some have one or two operands (either an actual value like a jump address,
    or often an index into the constant pool or local variable). Very rarely, there are instructions that have 
    (arbitrarily) many operands, and even one that is an instruction prefix. The full list of 'types' of tokens is:

        * opcodes - appear at the beginning of the instruction. Can be a string of alpha-numeric characters along with '_'
        * local variable index - a 1-byte index (or 2-byte in the case of a 'wide' prefixed instruction) into the local 
          variable array of the current frame
        * constant pool index - a 2-byte index into the constant pool
        * immediate value - signed 1-byte or 2-byte immediate value
        * branch address - signed 2-byte or 4-byte branch offset from the current address
        * table match - signed 4-byte immediate value that acts as a possible match for values in lookup tables
        * array type - unsigned 1-byte value designating a valid primitive array type
        * instruction prefix - the 'wide' instruction acts as an instruction prefix designating the use of 'wide' (2-byte
          short) index values into the local variable array
    """
    # This function won't work if the newline tuple isn't a Tokens.NEWLINE token or None
    if newline_tup is not None and newline_tup[0] != Tokens.NEWLINE:
        raise ValueError("Cannot check the assembly tokens with a newline_tup that is not a Tokens.NEWLINE: %s" % newline_tup)
    
    # Get each line and check that
    curr_line = []
    for token_type, token in token_list:
        if token_type in [Tokens.SPACING, Tokens.INSTRUCTION_ADDRESS, Tokens.DISASSEMBLER_INFO]:
            continue
        elif token_type in [Tokens.NEWLINE]:
            _check_java_line(curr_line)
            curr_line = []
        else:
            curr_line.append((token_type, token))
    
    if len(curr_line) > 0:
        _check_java_line(curr_line)
    
    return token_list


def _check_java_line(line):
    """Checks a given java line is valid"""
    _check_valid_structure(line)
    opcode, operands = _check_operands(line)
    _check_specials(opcode, operands)


def _check_specials(opcode, operands):
    """Does some special checks on specific opcodes and their values"""
    try:
        if opcode == 'lookupswitch':
            _, npairs, *pairs = operands
            pairs = list(zip(pairs[::2], pairs[1::2]))
            if npairs != len(pairs):
                raise ValueError("npairs != len(pairs): %d != %d" % (npairs, len(pairs)))
            if pairs != sorted(pairs, key=lambda p: p[0]):
                raise ValueError("pairs was not sorted: %s" % pairs)
            
        elif opcode == 'tableswitch':
            _, low, high, *jumps = operands
            low, high = int(low[1], 0), int(high[1], 0)
            if low > high:
                raise ValueError("tableswitch must have low <= high. Low: %d, high: %d" % (low, high))
            if high - low + 1 != len(jumps):
                raise ValueError("tablswitch must have (high - low + 1) jumps, should be %d jumps, found %d. High: %d, low: %d"
                                 % (high - low + 1, len(jumps), high, low))
    except Exception as e:
        raise ValueError("Error checking special constraints on opcode: %s, operands: %s.\nMessage: %s\nTraceback: %s" 
                         % (repr(opcode), operands, e, traceback.format_exc()))


def _check_operands(line):
    """Checks that the opcodes have the corrent number/types of operands
    
    Instructions should already be in the form: {prefix}?{opcode}{operands}*

    Returns the opcode and operands list
    """
    prefix_wide = False
    opcode, operands = None, []
    for token_type, token in line:
        if token_type in [Tokens.INSTRUCTION_PREFIX]:
            if token == 'wide':
                prefix_wide = True
            else:
                raise NotImplementedError("Unknown instruction prefix: %s in line: %s" % (repr(token), line))
            
        elif token_type in [Tokens.OPCODE]:
            opcode = token
        
        else:
            operands.append((token_type, token))
    
    # Check that the 'wide' prefix is on a valid opcode
    if prefix_wide:
        if opcode not in _JAVA_PREFIX_WIDE_OPCODES:
            raise ValueError("Instruction prefix 'wide' found on invalid opcode %s, only allowed on: %s, in line: %s" 
                             % (repr(opcode), _JAVA_PREFIX_WIDE_OPCODES, line))

    # Check the opcode is known and we can get the opspecs
    if opcode not in _JAVA_OPCODE_TYPES:
        raise ValueError("Unknown java opcode: %s in line: %s" % (repr(opcode), line))
    opspecs = _JAVA_OPCODE_TYPES[opcode]
    
    # Check the number of operands is correct. A 'pairs' should have at least one pair, hence the +1 (since the 'pair' type is present in opspecs)
    if _JavaOperandType.PAIRS in opspecs:
        if len(operands) < len(opspecs) + 1:
            raise ValueError("Did not find enough operands for opcode %s. Found: %d, expected minimum of: %d. Valid opspec: %s. In line: %s"
                            % (repr(opcode), len(operands), len(opspecs) + 1, opspecs, line))
    elif _JavaOperandType.JUMPS in opspecs:
        if len(operands) < len(opspecs):
            raise ValueError("Did not find enough operands for opcode %s. Found: %d, expected minimum of: %d. Valid opspec: %s. In line: %s"
                            % (repr(opcode), len(operands), len(opspecs), opspecs, line))
    elif len(operands) != len(opspecs):
        raise ValueError("Invalid number of operands for opcode %s. Found: %d, expected: %d. Valid opspec: %s. In line: %s"
                         % (repr(opcode), len(operands), len(opspecs), opspecs, line))

    # Check the operands match the opspec
    for i, opspec in enumerate(opspecs):
        try:
            if opspec == _JavaOperandType.LOCAL_VARIABLE:
                _fits_in_bits(operands[i], 16 if prefix_wide else 8, signed=False)
            elif opspec in [_JavaOperandType.CONSTANT_POOL, _JavaOperandType.CONSTANT_POOL_BYTE]:
                _fits_in_bits(operands[i], 16 if opspec == _JavaOperandType.CONSTANT_POOL else 8, signed=False, positive=True)
            elif opspec in [_JavaOperandType.BRANCH, _JavaOperandType.BRANCH_LONG]:
                _fits_in_bits(operands[i], 16 if opspec == _JavaOperandType.BRANCH else 32, signed=True)
            elif opspec == _JavaOperandType.ARRAY_TYPE:
                _fits_in_bits(operands[i], 8, signed=False)
                if int(operands[i][1], 0) not in _JAVA_ARRAY_TYPE_VALUES:
                    raise ValueError("Unknown array_type integer value: %d. Valid values: %s" % (int(operands[i][1], 0), _JAVA_ARRAY_TYPE_VALUES))
            elif opspec == _JavaOperandType.ZERO:
                if int(operands[i][1], 0) != 0:
                    raise ValueError("Non-zero value: %s" % int(operands[i][1], 0))
            elif opspec in [_JavaOperandType.BYTE, _JavaOperandType.BYTE_POSITIVE]:
                _fits_in_bits(operands[i], 16 if prefix_wide else 8, signed=True, positive=opspec == _JavaOperandType.BYTE_POSITIVE)
            elif opspec in [_JavaOperandType.SHORT]:
                _fits_in_bits(operands[i], 16, signed=True)
            elif opspec in [_JavaOperandType.LONG, _JavaOperandType.LONG_POSITIVE]:
                _fits_in_bits(operands[i], 32, signed=True, positive=opspec == _JavaOperandType.LONG_POSITIVE)
            elif opspec == _JavaOperandType.PAIRS:
                while i < len(operands):
                    if len(operands) == i + 1:
                        raise ValueError("Unmatched pair")
                    _fits_in_bits(operands[i], 32, signed=True)
                    _fits_in_bits(operands[i + 1], 32, signed=True)
                    i += 2
            elif opspec == _JavaOperandType.JUMPS:
                while i < len(operands):
                    _fits_in_bits(operands[i], 32, signed=True)
                    i += 1
            else:
                raise NotImplementedError("Unknown _JavaOperandType: %s" % opspec)
        except Exception as e:
            raise ValueError("Bad operand %s for opcode %s. Opspec: %s, in line: %s\nReason: %s\nTraceback: %s"
                             % (operands[i], opcode, opspec.name, line, e, traceback.format_exc()))
    
    return opcode, operands
    

def _fits_in_bits(operand, bits, signed=False, positive=False):
    """Returns True if the given value can be stored in the given number of bits, False otherwise
    
    NOTE: passing 'signed' means that either val or it's two's complement (if negative) can be expressed as a `bits`-bit
    UNSIGNED integer, or that abs(val - 2^64) can fit within a `bits`-bit UNSIGNED integer
    """
    token_type, val = operand
    if token_type != Tokens.IMMEDIATE:
        raise ValueError("Invalid token type: %s, should be: 'immediate'" % repr(token_type.name))
    val = int(val, 0)
    
    if signed:
        if (val < - (2 ** (bits - 1)) or val >= 2 ** bits) and not abs(val - 2 ** 64) < 2**32:
            raise ValueError("Value %d could not fit into a %d-bit signed integer" % (val, bits))
    else:
        if val < 0 or val >= 2 ** bits:
            raise ValueError("Value %d could not fit into a %d-bit unsigned integer" % (val, bits))
    
    if positive and val <= 0:
        raise ValueError("Value %d was not positive" % val)


def _check_valid_structure(line):
    """Checks that the instruction matches a valid instruction structure:
    
    {prefix}?{opcode}{immediate}*
    """
    string = ''
    for token_type, token in line:
        if token_type not in _JAVA_TOKEN_TYPE_TO_CHAR_CODE:
            raise ValueError("Invalid token type %s in instruction: %s" % (repr(token_type), line))
        string += _JAVA_TOKEN_TYPE_TO_CHAR_CODE[token_type]
    
    if not _JAVA_STRUCTURE_RE.fullmatch(string):
        raise ValueError("Invalid instruction structure: %s. Should match: %s" % (repr(string), repr(_JAVA_STRUCTURE_RE.pattern)))


# For checking a valid java instruction structure
_JAVA_TOKEN_TYPE_TO_CHAR_CODE = {
    Tokens.INSTRUCTION_PREFIX: 'p',
    Tokens.OPCODE: 'o',
    Tokens.IMMEDIATE: 'i',
}
_JAVA_STRUCTURE_RE = re.compile(r'p?oi*')


# For checking valid operands/operand types
# Values are case-sensitive string types in the java_isa.txt file
class _JavaOperandType(Enum):
    LOCAL_VARIABLE = 'l'        # An unsigned 1-byte index (2-byte if 'wide') into the local variable array
    CONSTANT_POOL = 'c'         # An unsigned non-zero 2-byte index into the constant pool
    CONSTANT_POOL_BYTE = 'cb'   # An unsigned non-zero 1-byte index into the constant pool
    BRANCH = 'b'                # A signed 2-byte branch offset
    BRANCH_LONG = 'bl'          # A signed 4-byte branch offset
    PAIRS = 'P...'              # An arbitrary non-zero amount of (match, offset) pairs, each signed 4-byte values for a lookupswitch
    JUMPS = 'J...'              # An arbitrary non-zero amount of signed 4-byte jump offsets for a tableswitch
    ARRAY_TYPE = 'a'            # An unsigned 1-byte value for the primitive array type. Can be: [4, 5, 6, 7, 8, 9, 10, 11]

    ZERO = 'Z'                  # A 1-byte value of 0
    BYTE = 'B'                  # A signed 1-byte value
    BYTE_POSITIVE = 'B+'        # A positive signed 1-byte value
    SHORT = 'S'                 # A signed 2-byte value
    LONG = 'L'                  # A signed 4-byte value
    LONG_POSITIVE = 'L+'        # A positive signed 4-byte value

_PARSE_JAVA_TYPE_RE = re.compile(r'([^ \t\n]*)(?:[ \t\n]*([^ \t\n]+))?')

def _parse_java_isa(path):
    """Parses the isa information from the given path
    
    Should be a text file with each line starting with a non-instruction-prefix object, either alone, or followed by some
    amount of whitespace then a comma-separated list of _JavaOperandType strings denoting the operand types in order
    NOTE: the 'wide' instruction prefix is not put in this list
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    
    ret = {}
    for l in [l.strip() for l in lines]:
        if l == '':
            continue

        match = _PARSE_JAVA_TYPE_RE.fullmatch(l)
        if match is None:
            raise ValueError("Could not parse java isa opcode/type information in line: %s" % repr(l))
        
        ret[match.groups()[0]] = []
        if match.groups()[1] is not None:
            op_types = match.groups()[1].strip()
            op_types = op_types[:-1] if op_types.endswith(',') else op_types

            if _JavaOperandType.PAIRS.name in op_types and not op_types.endswith(_JavaOperandType.PAIRS.name):
                raise ValueError("Any 'pairs' operand type must be at the end of an instruction: %s" % repr(l))
            
            for op_type in op_types.split(','):
                op_type = op_type.strip()
                if op_type not in _JavaOperandType._value2member_map_:
                    raise ValueError("Unknown java operand type %s in line: %s" % (repr(op_type), repr(l)))
                ret[match.groups()[0]].append(_JavaOperandType._value2member_map_[op_type])
    
    return ret

_JAVA_OPCODE_TYPES = _parse_java_isa(os.path.join(os.path.dirname(__file__), 'java_isa.txt'))


# The valid opcodes that the 'wide' instruction prefix can appear on
_JAVA_PREFIX_WIDE_OPCODES = ['iload', 'fload', 'aload', 'lload', 'dload', 'istore', 'fstore', 'astore', 'lstore', 'dstore', 'ret', 'iinc']


# The valid ARRAY_TYPE operand integer values
_JAVA_ARRAY_TYPE_VALUES = [4, 5, 6, 7, 8, 9, 10, 11]