from ..base_tokenizer import Tokens, BaseTokenizer, Architectures
from ..norm_utils import RE_SPACING, RE_NEWLINE, scan_for_token
import re


# The delimiter lookahead. Allows for lookahead to make sure there is some delimiter, and we don't greedily grab 
#   registers/values/etc in case their names interfere with anything else.
# Can be a spacing, newline, or end of string
X86_RE_DELIMITER = r'{spacing}|{newline}|[\[\]+*<>]|$'.format(spacing=RE_SPACING, newline=RE_NEWLINE)

# Dictionary mapping all x86_64 registers to their sizes in bytes
X86_REGISTER_SIZES = {
    k:v for a, v in [
        (['xmm%d' % i for i in range(16)], 16),                             # XMM 16-byte registers
        (['ymm%d' % i for i in range(16)], 32),                             # YMM 32-byte registers
        (['zmm%d' % i for i in range(32)], 64),                             # ZMM 64-byte registers

        (['st%d' % i for i in range(16)], 10),                              # ST 10-byte registers (rose interpretation)
        (['st(%d)' % i for i in range(16)], 10),                            # ST 10-byte registers (ghidra interpretation)
        (['st'], 10),                                                       # Plain 'st' is just 'st(0)'
        (['mm%d' % i for i in range(8)], 8),                                # MM 8-byte registers [lower 64 bits of ST]
        
        (['cw', 'fp_ip', 'fp_dp', 'fp_cs', 'sw', 'tw', 
            'fp_ds', 'fp_opc', 'cs', 'ss', 'ds', 'es', 'fs', 
            'gs', 'gdtr', 'idtr', 'ldtr', 'tr', 'msw'], 2),                 # Various special 2-byte registers
        (['fp_dp', 'fp_ip', 'mxcsr'], 4),                                   # Various special 4-byte registers

        (['%sr%d' % (k, v) for k in ['c', 'd'] for v in range(16)], 8),     # DR(0-15)/CR(0-15) 8-byte registers

        (['r%d' % v for v in range(8, 16)], 8),                             # R(8-15)  8-byte registers
        (['r%dd' % v for v in range(8, 16)], 4),                            # R(8-15)d 4-byte registers
        (['r%dw' % v for v in range(8, 16)], 2),                            # R(8-15)w 2-byte registers
        (['r%db' % v for v in range(8, 16)], 1),                            # R(8-15)b 1-byte registers

        (['rax', 'rbx', 'rcx', 'rdx'], 8),                                  # Named general 8-byte r-registers
        (['eax', 'ebx', 'ecx', 'edx'], 4),                                  # Named general 4-byte r-registers
        (['ax', 'bx', 'cx', 'dx'], 2),                                      # Named general 2-byte r-registers

        (['rflags', 'rip', 'rbp', 'rsi', 'rdi', 'rsp'], 8),                 # Various special 8-byte r-registers
        (['eflags', 'eip', 'ebp', 'esi', 'edi', 'esp'], 4),                 # Various special 4-byte e-registers
        (['flags', 'ip', 'bp', 'si', 'di', 'sp'], 2),                       # Various special 2-byte registers

        (['bpl', 'sil', 'dil', 'spl'], 1),                                  # Various special 1-byte registers
        (['%s%s' % (a, b) for a in 'abcd' for b in 'lh'], 1),               # Various general 1-byte registers (low and high)
    ]
    for k in a
}

# Dictionary mapping all base memory sizes to their size in bytes
# NOTE: Should find someone who can confirm sizes here, because I get different values when googling what 'ldouble' is...
X86_MEMORY_SIZES = {
    'byte': 1, 'word': 2, 'dword': 4, 'qword': 8, 'tword': 10, 'float': 4, 'tfloat': 10, 'double': 8, 'ldouble': 16,
    'xmmword': 16, 'ymmword': 32, 'zmmword': 64,
}

# Regex matches to registers. This is ~15% faster than brute-force matching all keys in REGISTER_SIZES
# Registers can be followed by a '*', ']', or delimiter
X86_RE_REG_RE_MATCH = r'[xyz]?mm[0-9]+|st\(?[0-9]*\)?|(?:[sb]p|[ds]i)l|(?:[sb]p|[ds]i)l|[re]?(?:flags|ip|[bs]p|[sd]i|[abcd]x)|[cd]r[0-9]+|r[0-9]+[dwb]?|[abcd][lh]|[cst]w|fp_(?:[id]p|[cd]s|opc)|[cdefgs]s|(?:[gil]d)?tr|msw|mxcsr'
X86_RE_ALL_REGISTERS = r'(?:{all_reg})(?={delim})'.format(all_reg='|'.join([X86_RE_REG_RE_MATCH]), delim=X86_RE_DELIMITER)

# x86_64 instruction prefixes, stolen from: http://web.mit.edu/rhel-doc/3/rhel-as-en-3/i386-prefixes.html.
# These must be followed by either an underscore (which will be captured), or a delimiter
# We split these up into codes + needing delimiter for some postprocessing that is done in the tokenizers
X86_RE_INSTRUCTION_PREFIX_CODES = r'(?:lock|rep(?:ne|nz|e|z)?)'
X86_RE_INSTRUCTION_PREFIX = r'{codes}(?={delim})'.format(delim=X86_RE_DELIMITER, codes=X86_RE_INSTRUCTION_PREFIX_CODES)

# For branch prediction. Not used by re for tokenization, but used in postprocessing of token stream
X86_RE_BRANCH_PREDICTION = r'p[tn]'

# Known memory sizes. Includes a 'v%d' in front of all of them to denote a vector of multiple of these. Optional 'ptr' after spacing
X86_RE_MEM_SIZES = r'(?:v[0-9]+)?(?:byte|[dqt]?word|t?float|l?double|[xyz]mmword)(?:{spacing}ptr)?(?={delim})'.format(delim=X86_RE_DELIMITER, spacing=RE_SPACING)

# The characters that are used to designate a possible opcode splitting (when the instruction prefix or branch prediction
#   tokens are prepended or appended to the opcode)
X86_OPCODE_SPLIT_CHARS = '.,_'

# We also want to capture the opcode splitting character
X86_OPCODE_PREFIX_SPLIT = re.compile(r'([{op_split}]+)'.format(op_split=X86_OPCODE_SPLIT_CHARS))

# Check for any opcode mnemonic. For now, we just get any and all characters that could correspond to an opcode, 
#   instruction prefix, or branch prediction. These will be sorted out later when parsing. Must be followed by delimiter
X86_RE_OPCODE = r'(?:[a-z][a-z0-9{op_split}]*)(?={delim})'.format(delim=X86_RE_DELIMITER, op_split=X86_OPCODE_SPLIT_CHARS)

# Match names to tokens, and define the order in which they should be matched
X86_DEFAULT_TOKENS = [
    (Tokens.INSTRUCTION_PREFIX, X86_RE_INSTRUCTION_PREFIX),  # Instructions prefixes (EG: lock, rep, etc.), possible underscore after them
    (Tokens.MEMORY_SIZE, X86_RE_MEM_SIZES),  # Known memory sizes. Must go before reg/opcode
    (Tokens.REGISTER, X86_RE_ALL_REGISTERS),  # All known registers accross all architectures
    (Tokens.OPCODE, X86_RE_OPCODE),  # Opcode. Should go last since it's the most general matching. 
]
"""Default list of (token_type, regex) token tuples to match to"""


class X86BaseTokenizer(BaseTokenizer):
    """A default class to tokenize x86 assembly line input

    This class matches the following tokens:

        * All of the default special tokens from parent (see :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer`)
        * Instruction prefix tokens (EG: 'lock', 'repe', etc.)
        * Memory sizes (EG: 'qword', 'byte ptr', 'xmmword', 'v2float', etc.)
        * Registers (see ``bincfg.normalization.x86.x86_tokenizer.X86_REGISTER_SIZES`` for the list of them)
        * Prepended or appended instruction prefixes and branch predictions. You can prepend or append either of those
          to opcodes while separating with either ',', '_', or '.'. EG: "lock_str", "jnz,pt". This will only apply to
          known prefixes and branch prediction strings; if it is unknown, it is considered a part of a larger opcode
          (EG: "vcmpneq_oqss").

          Known prefixes: ['lock', 'rep', 'repe', 'repz', 'repne', 'repnz']
          Known branch predictions: ['pt', 'pn']

    This will perform the following transformations to the incomming token stream:

        * Instruction prefixes and branch prediction tokens may be reordered to keep ordering consistent. NOTE: if you
          do not wish to have this behavior, pass 'reorder_tokens=False' as a kwarg to the `.tokenize()` call.

          Any instruction prefixes alone on their own line will be moved to the next line under the assumption that is
          the opcode they are affecting (no check is done to ensure the first token in the subsequent line is actually
          an opcode, however). Then, for each opcode, we get all surrounding instruction prefix and branch prediction
          tokens (ignoring spacing, grabbing all tokens until reaching a non-instruction prefix and non-branch prediction
          token). These are reordered such that all instruction prefixes come first, then branch predictions after those,
          and finally the opcode token. They will appear in the order that they were found in the token list. There will
          only be a single space ' ' acting as spacing inbetween them.


    Parameters
    ----------
    tokens: `Optional[List[Tuple[str, str]]]`
        the tokens to use. Should be a list of 2-tuples. Each tuple is a pair of (name, regex) where
        name is the string name of the token, and regex is a regular expression to find that token. These
        tuples should be ordered in the preferred order to search for tokens. If None, then this will default to 
        self.DEFAULT_TOKENS, which is ``bincfg.normalization.x86.x86_tokenizer.X86_DEFAULT_TOKENS``
    token_handlers: `Optional[Dict[str, Callable[[Dict[str, Any]], Union[None, str]]]]`
        optional dictionary mapping token type strings to functions to handle those token types when tokenizing. This is
        intended to be used when you wish to add entirely new token types not present in `bincfg.normalization.base_tokenizer.Tokens`.
        If you wish to change the behavior of handling an already-present token type, just override that token handler function.
        These will override the default token handlers.
    insert_special_tokens: `bool`
        by default, some special tokens will be inserted at the front of `tokens` (see the 'special tokens' listed above).
        If you wish to stop this from happening, you can set `insert_special_tokens` to False
    case_sensitive: `bool`
        If True, then it is assumed that all regular expressions will exactly match case. If False, then it is assumed
        that all regular expressions only handle lowercase strings, and all incoming instructions will be converted to lowercase
    """
    DEFAULT_TOKENS = X86_DEFAULT_TOKENS

    ARCHITECTURE = Architectures.X86
    """The architecture this tokenizer works on"""
    
    def token_opcode(self, state):
        """Handles the opcode tokens
        
        Overriden from BaseTokenizer. Splits off instruction prefixes and branch predictions from opcodes.

        Cases we must consider:

            - opcodes can be surrounded by any number of prefixes or branch predictions, delimited by ',', '.', or '_'
            - final results will be rearranged to be prefixes, then branch prediction, then opcodes, in the order they
              appear in the token
            - it is possible to not have any opcodes, just prefixes
            - all possible prefixes and branch predictions are known, and any extra tokens will be considered opcodes.
              If there is more than one opcode, then an error will be raised
            - all extra spacing directly attached to this token will be ignored

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            None: we fix everything in the line here, so nothing to return
        """
        prefixes = []
        branch_predictions = []
        opcodes = []

        prev_opcode, opcode_spacing = False, None

        for t in X86_OPCODE_PREFIX_SPLIT.split(state['token']):
            if len(t) == 0:
                continue

            # Check for spacing to ignore, or to set as prev_opcode spacing
            if t[0] in ',_.':
                if prev_opcode and opcode_spacing is None:
                    opcode_spacing = t
                continue

            # Check for prefixes and branch predictions
            elif re.fullmatch(X86_RE_INSTRUCTION_PREFIX_CODES, t):
                prefixes.append((Tokens.INSTRUCTION_PREFIX, t))
            elif re.fullmatch(X86_RE_BRANCH_PREDICTION, t):
                branch_predictions.append((Tokens.BRANCH_PREDICTION, t))
            
            # Finally, this must be an opcode. Allow for subsequent opcodes connected by spacing
            else:
                if prev_opcode and opcode_spacing is not None:
                    opcodes[-1] = (Tokens.OPCODE, opcodes[-1][1] + opcode_spacing + t)
                else:
                    opcodes.append((Tokens.OPCODE, t))
                prev_opcode = True
                continue
            
            prev_opcode = False
        
        # Add all the prefixes, branch predictions, and opcodes to the line and return None
        state['line'].extend(prefixes + branch_predictions + opcodes)

        return None
    
    def handle_sentence(self, state):
        """Handles a full sentence (everything passed to the tokenizer)
        
        Overridden from BaseTokenizer. This will reorder some tokens (see below). If you wish to stop this behavior,
        pass 'reorder_tokens=False' as a kwarg during the `.tokenize()` call.

        Assuming 'reorder_tokens=False' has not been passed, this will:

            - Move any lonely instruction prefixes (those that exist as the only non-spacing token in a line) to the
              next instruction. They will be placed immediately previous the first instruction prefix in the next line,
              or the opcode if no instruction prefixes are present. This is needed because ROSE seems to put instruction
              prefixes on their own line instead of next to the opcode they affect.
            - Rearrange any instruction prefixes, branch predictions, etc. in each line in the sentence.
        
        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            List[Tuple[str, str]]: the final list of tokens
        """
        # If we are not reordering, just return
        if 'reorder_tokens' in state['kwargs'] and not state['kwargs']['reorder_tokens']:
            return state['sentence']
        
        ret_line = []
        idx = 0
        while idx < len(state['sentence']):
            token_type, token = state['sentence'][idx]

            # Check if this is a 'lonely' instruction prefix. Must meet the following:
            #   1. An instruction prefix token
            #   2. Either at the beginning of sentence, or previous non-space token is newline
            #   3. Not at end of sentence, and subsequent non-space token is newline
            # If so, then all subsequent spacing and newline characters are removed until reaching a non-space and non-newline
            #   token, and a single space ' ' token is inserted inbetween the instruction prefix and that next token
            scan_kwargs = {'token_list': state['sentence'], 'type': Tokens.NEWLINE, 'stop_unmatched': True, 'ignore_type': [Tokens.SPACING]}
            if token_type in [Tokens.INSTRUCTION_PREFIX] and idx < len(state['sentence']) - 1 and \
                (idx == 0 or scan_for_token(start=idx-1, increment=-1, **scan_kwargs) is not None) and \
                scan_for_token(start=idx+1, **scan_kwargs) is not None:
                
                idx = scan_for_token(state['sentence'], type=r'.*', ignore_type=[Tokens.SPACING, Tokens.NEWLINE], start=idx+1)
                ret_line += [(token_type, token), (Tokens.SPACING, ' ')]
                continue
            
            # If this isn't an opcode, just append it to the ret_line
            if token_type not in [Tokens.OPCODE]:
                ret_line.append((token_type, token))
                idx += 1
                continue

            # Otherwise, we need to find all previous and subsequent useful tokens (ignoring spacing), and checking
            #   to make sure we aren't at the start or end of a line when scanning
            find_t = [Tokens.INSTRUCTION_PREFIX, Tokens.BRANCH_PREDICTION]  # Make these in the order they should appear in output
            scan_kwargs = {'token_list': state['sentence'], 'type': find_t, 'stop_unmatched': True, 'wrap': False, 'max_matches': None,
                           'ignore_type': [Tokens.SPACING], 'ret': ['index', 'type', 'token'], 'on_no_match': []}
            prev_tokens, subseq_tokens = scan_for_token(start=idx-1, increment=-1, **scan_kwargs), scan_for_token(start=idx+1, **scan_kwargs)

            # Reorder these tokens
            new_tokens = sorted(prev_tokens + subseq_tokens, key=lambda t: find_t.index(t) if t in find_t else 1000)

            # Remove all previous tokens, insert the reordered tokens, and skip ahead in sentence
            rl_idx = len(ret_line) - (idx - min([t[0] for t in prev_tokens] + [idx]))
            ret_line = ret_line[:rl_idx] + [(t[1], t[2]) for t in new_tokens] + [(token_type, token)]
            idx = max([t[0] for t in subseq_tokens] + [idx]) + 1
        
        return ret_line


X86_DEFAULT_TOKENIZER = X86BaseTokenizer()
