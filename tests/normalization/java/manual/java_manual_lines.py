from bincfg import Tokens, TokenMismatchError
from bincfg.normalization import *
from bincfg.normalization.base_normalizer import DEFAULT_IMMEDIATE_THRESHOLD
from bincfg.normalization.norm_utils import *


# Name, class, kwargs
JAVA_TEST_OBJECTS = [
    ('java_base_tokenizer', JavaBaseTokenizer, {}),
    ('java_base', JavaBaseNormalizer, {}),
    ('java_repl_imm', JavaReplaceImmediateNormalizer, {}),
]


JAVA_TEST_INPUTS = [

    {
        'input': 'iload 0',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'iload'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0'),
            (Tokens.NEWLINE, '\n')
        ],
        'java_base': ['iload 0'],
        'java_repl_imm': ['iload {immval}'],
    },

    {
        'input': 'aload_0',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'aload_0'), (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['aload_0'],
        'java_repl_imm': ['aload_0'],
    },

    {
        'input': 'iconst_m1',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'iconst_m1'), (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['iconst_m1'],
        'java_repl_imm': ['iconst_m1'],
    },

    {
        'input': 'newarray 0x0a',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'newarray'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x0a'), (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['newarray 10'],
        'java_repl_imm': ['newarray {immval}'],
    },

    {
        'input': 'invokespecial 0x0001<java/lang/Object::<init>>',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'invokespecial'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x0001'), 
            (Tokens.DISASSEMBLER_INFO, '<java/lang/Object::<init>>'), (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['invokespecial 1'],
        'java_repl_imm': ['invokespecial {func}'],
    },

    {
        'input': 'invokevirtual 0x0012<java/util/Scanner::nextInt>',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'invokevirtual'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x0012'), 
            (Tokens.DISASSEMBLER_INFO, '<java/util/Scanner::nextInt>'), (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['invokevirtual 18'],
        'java_repl_imm': ['invokevirtual {func}'],
    },

    {
        'input': 'invokedynamic 0x0116<278>, 0x00, ,  0x00',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'invokedynamic'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x0116'), 
            (Tokens.DISASSEMBLER_INFO, '<278>'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00'), (Tokens.SPACING, ', ,  '), 
            (Tokens.IMMEDIATE, '0x00'), (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['invokedynamic 278 0 0'],
        'java_repl_imm': ['invokedynamic {func} {immval} {immval}'],
    },

    {
        'input': 'getstatic 0x0009<java/lang/System::in>',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'getstatic'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x0009'), 
            (Tokens.DISASSEMBLER_INFO, '<java/lang/System::in>'), (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['getstatic 9'],
        'java_repl_imm': ['getstatic {immval}'],
    },

    {
        'input': 'if_icmpne 0x0006',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'if_icmpne'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x0006'), (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['if_icmpne 6'],
        'java_repl_imm': ['if_icmpne {jmpdst}'],
    },

    {
        'input': 'goto 0x0006',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'goto'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x0006'), (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['goto 6'],
        'java_repl_imm': ['goto {jmpdst}'],
    },

    {
        'input': 'multianewarray 0x0016, 0x02',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'multianewarray'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x0016'), 
            (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x02'), (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['multianewarray 22 2'],
        'java_repl_imm': ['multianewarray {immval} {immval}'],
    },

    {
        'input': 'lookupswitch 0x0000001e, 0x00000002, 0x00000001, 0x0000001b, 0x00000003, 0x0000001b',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'lookupswitch'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x0000001e'),
            (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00000002'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00000001'),
            (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x0000001b'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00000003'),
            (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x0000001b'),
            (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['lookupswitch 30 2 1 27 3 27'],
        'java_repl_imm': ['lookupswitch {jmpdst} {immval} {immval} {immval} {immval} {immval}'],
    },

    {
        'input': 'tableswitch 0x00000071, 0x00000000, 0x00000009, 0x00000038, 0x0000003e, 0x00000044, 0x0000004a, 0x00000050, 0x00000056, 0x0000005c, 0x00000062, 0x00000068, 0x0000006e',
        'java_base_tokenizer': [
            (Tokens.OPCODE, 'tableswitch'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x00000071'),
            (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00000000'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00000009'),
            (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00000038'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x0000003e'),
            (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00000044'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x0000004a'),
            (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00000050'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00000056'),
            (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x0000005c'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00000062'),
            (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00000068'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x0000006e'),
            (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['tableswitch 113 0 9 56 62 68 74 80 86 92 98 104 110'],
        'java_repl_imm': ['tableswitch {jmpdst} {immval} {immval} {immval} {immval} {immval} {immval} {immval} {immval} {immval} {immval} {immval} {immval}'],
    },

    {
        'input': 'wide   iinc 0x0009, 0xfc18<-1000>',
        'java_base_tokenizer': [
            (Tokens.INSTRUCTION_PREFIX, 'wide'), (Tokens.SPACING, '   '), (Tokens.OPCODE, 'iinc'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x0009'), 
            (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0xfc18'), (Tokens.DISASSEMBLER_INFO, '<-1000>'), (Tokens.NEWLINE, '\n'),
        ],
        'java_base': ['wide iinc 9 -1000'],
        'java_repl_imm': ['wide iinc {immval} -{immval}'],
    },

    {  # Token mismatch (unknown character)
        'input': 'add ###',
        'java_base_tokenizer': TokenMismatchError,
        'java_base': TokenMismatchError,
        'java_repl_imm': TokenMismatchError,
    },

]