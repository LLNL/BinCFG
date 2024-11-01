"""Makes an example CFG file for testing"""
import os
from collections import Counter
from bincfg import CFG, MemCFG, CFGBasicBlock, CFGFunction, CFGEdge
from cfg.manual_cfgs.fake_classes import *

# The CFG itself. You have to write to code to load it in here
from smda.Disassembler import Disassembler
cfg = CFG(Disassembler().disassembleFile('./cfg/manual_cfgs/x86/gol.compiled'))
arch = 'x86'
output_path = './cfg/manual_cfgs/x86/cfg_gol_smda.py'
file_comment = """Conway's GOL binary analyzed with smda

Code:

```
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

#define aliveChar 'X'
#define deadChar ' '
#define boardWidth 30
#define boardHeight 15
#define initAliveChance 0.3f
#define waitTimeMillis 500

#define SEED 12345 // (unsigned int)time(NULL)


/* msleep(): Sleep for the requested number of milliseconds. */
int msleep(long msec)
{
    struct timespec ts;
    int res;

    if (msec < 0)
    {
        errno = EINVAL;
        return -1;
    }

    ts.tv_sec = msec / 1000;
    ts.tv_nsec = (msec % 1000) * 1000000;

    do {
        res = nanosleep(&ts, &ts);
    } while (res && errno == EINTR);

    return res;
}

void print_board(char board[boardHeight][boardWidth]) {
    putchar('\n');

    // print horizontal line
    for (int i = 0; i < boardWidth + 2; i++) {
        putchar('-');
    }
    putchar('\n');

    // print board
    for (int r = 0; r < boardHeight; r++) {
        putchar('|');
        for (int c = 0; c < boardWidth; c++) {
            putchar(board[r][c] ? aliveChar : deadChar);
        }
        putchar('|');
        putchar('\n');
    }

    // print horizontal line
    for (int i = 0; i < boardWidth + 2; i++) {
        putchar('-');
    }
    putchar('\n');
}

int getNeighbors(char board[boardHeight][boardWidth], int r, int c) {
    int count = 0;

    if (r > 0 && c > 0 && board[r-1][c-1]) count ++;  // UL
    if (r > 0 && board[r-1][c]) count ++;  // U
    if (r > 0 && c < boardWidth - 1 && board[r-1][c+1]) count ++;  // UR
    if (c > 0 && board[r][c-1]) count ++;  // L
    if (c < boardWidth - 1 && board[r][c+1]) count ++;  // R
    if (r < boardHeight - 1 && c > 0 && board[r+1][c-1]) count ++;  // BL
    if (r < boardHeight - 1 && board[r+1][c]) count ++;  // B
    if (r < boardHeight - 1 && c < boardWidth - 1 && board[r+1][c+1])count ++;  // BR

    return count;
}


int main() {
    srand(SEED);

    // Initialize board to random values. 
    char board[boardHeight][boardWidth];
    for (int r = 0; r < boardHeight; r++) {
        for (int c = 0; c < boardWidth; c++) {
            board[r][c] = (char)( ((float)rand()/(float)(RAND_MAX)) < initAliveChance);
        }
    }

    // Continuously update and print the board
    while (1) {
        print_board(board);
        
        // Update our board
        char newBoard[boardHeight][boardWidth];
        for (int r = 0; r < boardHeight; r++) {
            for (int c = 0; c < boardWidth; c++) {
                int neighbors = getNeighbors(board, r, c);

                if (board[r][c] && (neighbors < 2 || neighbors > 3)) newBoard[r][c] = 0;
                else if (!board[r][c] && (neighbors == 3)) newBoard[r][c] = 1;
                else newBoard[r][c] = board[r][c];
            }
        }

        for (int r = 0; r < boardHeight; r++) {
            for (int c = 0; c < boardWidth; c++) {
                board[r][c] = newBoard[r][c];
            }
        }

        msleep(waitTimeMillis);
    }

    return 0;
}
```
"""


def make_file():
    tab_space = '    '
    func_innerds = '\n'.join([tab_space + l.replace("CFGFunction", "func_type") for l in cfg.get_cfg_build_code(
        insert_autogen_str=False, addfuncs_str=_ADD_FUNCS_STR, cfg_str=_CFG_STR,
    ).split('\n')])

    func_code = _BUILD_CFG_FUNC_STR % func_innerds
    exec(func_code)
    expected_dict_str = build_expected_dict(locals()['get_manual_cfg'](build_level='cfg'), arch=arch, tab_space=tab_space)

    func_code = func_code.replace(_REPL_EXPECTED, expected_dict_str)

    file_string = _FILE_STR % (file_comment, func_code)

    with open(output_path, 'w') as f:
        f.write(file_string)


_REPL_EXPECTED = "\"_REPL_EXPECTED\""
_BUILD_CFG_FUNC_STR = """def get_manual_cfg(build_level):
    \"\"\"Returns a manually built control flow graph
    
    Args:
        build_level (str): the level at which to build the cfg. Can be:

            - 'cfg': will build the full CFG() object
            - 'function': will build full CFGFunction's, but use a FakeCFG() object for the CFG
            - 'block': will build only CFGBasicBlock's. A FakeCFGFunction() object will be built as the functions that is
              simply an empty data structure to hold all the construction attributes being used, but with no processing

    Returns:
        dict[str, Any]: dictionary with the following keys/values:

            - 'blocks' (dict[int, CFGBasicBlock]): dictionary mapping basic block addresses to basic blocks
            - 'functions' (Union[dict[int, CFGFunction], dict[int, FakeCFGFunction]]): dictionary mapping function addresses
              to CFGFunctions (or, FakeCFGFunctions if we are not building functions)
            - 'cfg' (Union[CFG, object]): the CFG() object (if make_cfg=True) else a new object()
            - 'inputs' (list[CFGInputDataType]): input values that, when passed into CFG() constructor, should produce
              the exact same CFG() as that in 'cfg' (when build_level='cfg')
            - 'expected' (dict[str, Any]): dictionary of expected values. The following values are present:

                * 'sorted_block_order' (list[int]): list of basic block addresses in sorted order
                * 'sorted_func_order' (list[int]): list of function addresses in sorted order
                * 'num_blocks' (dict[int, int]): number of blocks per function, keys are function addresses
                * 'num_asm_lines_per_block' (dict[int, int]): number of asm lines per block, keys are block addresses
                * 'num_asm_lines_per_function' (dict[int, int]): number of asm lines per function, keys are function addresses
                * 'num_functions' (int): the number of functions
                * 'is_root_function' (dict[int, bool]): True if the function is a root function, keys are function addresses
                * 'is_extern_function' (dict[int, bool]): True if the function is an external function, keys are function addresses
                * 'is_intern_function' (dict[int, bool]): True if the function is an internal function, keys are function addresses
                * 'function_entry_block' (dict[int, int]): the address of the function entry block for each function, keys are function addresses
                * 'called_by' (dict[int, set[int]]): set of addresses of basic blocks that call each function, keys are function addresses
                * 'asm_counts_per_block' (dict[int, dict[str, int]]): dictionary of assembly line counts for each block, keys are block addresses
                * 'asm_counts_per_function' (dict[int, dict[str, int]]): dictionary of assembly line counts for each function, keys are function addresses
                * 'asm_counts' (dict[str, int]): dictinary of assembly line counts for entire CFG
    \"\"\"

    if build_level not in ['cfg', 'function', 'block']:
        raise ValueError("Bad build_level: %%s" %% repr(build_level))
    
    func_type = CFGFunction if build_level in ['function', 'cfg'] else FakeCFGFunction

    metadata = {'some': 'cfg-level', 134: ('metadata', True, None), (1, 2, 3): 'apples'}
    %s
"""

_ADD_FUNCS_STR = """if build_level in ['cfg']:
    __auto_cfg.add_function(*__auto_functions.values())
else:
    __auto_cfg.functions = list(__auto_functions.values())
    __auto_cfg.blocks = list(__auto_blocks.values())

expected = %s

return {
    'blocks': __auto_blocks,
    'file': os.path.basename(__file__),
    'inputs': [],
    'cfg': __auto_cfg,
    'functions': __auto_functions,
    'expected': expected,
}
""" % _REPL_EXPECTED

_CFG_STR = """__auto_cfg = CFG(metadata=metadata) if build_level in ['cfg'] else FakeCFG(metadata=metadata)"""


from bincfg import get_normalizer, Architectures, CFG, X86BaseNormalizer, X86InnerEyeNormalizer, X86SafeNormalizer, \
    X86DeepBinDiffNormalizer, X86DeepSemanticNormalizer, X86CompressedStatsNormalizer, X86HPCDataNormalizer,\
    JavaBaseNormalizer, JavaReplaceImmediateNormalizer


# All of the normalizers to test per architecture
ARCH_NORMS = {
    Architectures.X86: {
        'base_norm': (X86BaseNormalizer, {}),
        'innereye': (X86InnerEyeNormalizer, {}),
        'safe': (X86SafeNormalizer, {}),
        'deepbindiff': (X86DeepBinDiffNormalizer, {}),
        'deepsemantic': (X86DeepSemanticNormalizer, {}),
        'compressed_stats': (X86CompressedStatsNormalizer, {}),
        'hpcdata': (X86HPCDataNormalizer, {}),
    },
    Architectures.JAVA: {
        'java_base': (JavaBaseNormalizer, {}),
        'java_repl_imm': (JavaReplaceImmediateNormalizer, {}),
    },
}


def build_expected_dict(cfg_res, arch, tab_space='    '):
    """Makes the 'expected' dictionary values. Returns the string to copy/paste into the test file
    
    Args:
        cfg_res (dict): the result of a call to a manual_cfg function with build_level='cfg'
        arch (str): the expected architecture of these blocks/functions
    """

    expected = {
        'sorted_func_order': [cfg_res['functions'][a].address for a in sorted([f.address for f in cfg_res['functions'].values()])],
        'sorted_block_order': [cfg_res['blocks'][a].address for a in sorted([b.address for b in cfg_res['blocks'].values()])],
        'architecture': arch,
        'num_blocks': {
            k: len(f.blocks) for k, f in cfg_res['functions'].items()
        },
        'num_asm_lines_per_block': {
            k: len(b.asm_lines) for k, b in cfg_res['blocks'].items()
        },
        'num_asm_lines_per_function': {
            k: sum(len(b.asm_lines) for b in f.blocks) for k, f in cfg_res['functions'].items()
        },
        'num_functions': len(cfg_res['functions']),
        'is_root_function': {
            k: all(e.edge_type not in FUNC_EDGE_TYPES or e.to_block.address not in set(b1.address for b1 in f.blocks) for b in cfg_res['blocks'].values() for e in b.edges_out) for k, f in cfg_res['functions'].items()
        },
        'is_recursive': {
            k: any(e.to_block.address in set(b.address for b in f.blocks) and e.edge_type in FUNC_EDGE_TYPES for b in f.blocks for e in b.edges_out) for k, f in cfg_res['functions'].items()
        },
        'is_extern_function': {
            k: f.is_extern_function for k, f in cfg_res['functions'].items()
        },
        'is_intern_function': {
            k: not f.is_extern_function for k, f in cfg_res['functions'].items()
        },
        'function_entry_block': {
            k: [b.address for b in f.blocks if b.address == f.address][0] for k, f in cfg_res['functions'].items()
        },
        'called_by': {
            k: set(b.address for b in cfg_res['blocks'].values() if any((e.to_block.address in set(b2.address for b2 in f.blocks) and e.edge_type in FUNC_EDGE_TYPES) for e in b.edges_out)) for k, f in cfg_res['functions'].items()
        },
        'function_hashes': {f.address: hash(f) for f in cfg_res['functions'].values()},
        'block_hashes': {b.address: hash(b) for b in cfg_res['blocks'].values()},
        'cfg_hash': hash(cfg_res['cfg']),
        'memcfg_hashes': {k: hash(MemCFG(cfg_res['cfg'], normalizer=n, keep_memory_addresses=True)) for k, n in [(k+'-'+tl, nc(tokenization_level=tl, **nk)) for k, (nc, nk) in ARCH_NORMS[cfg_res['cfg'].architecture].items() for tl in ['op', 'inst']]},
        'metadata': cfg_res['cfg'].metadata,
        'block_metadatas': {b.address: b.metadata for b in cfg_res['blocks'].values()},
        'function_metadatas': {f.address: f.metadata for f in cfg_res['functions'].values()},
        'asm_counts_per_block': {
            k: dict(Counter(b.asm_lines)) for k, b in cfg_res['blocks'].items()
        },
        'asm_counts_per_function': {
            k: dict(Counter(l for b in f.blocks for l in b.asm_lines)) for k, f in cfg_res['functions'].items()
        },
        'asm_counts': dict(Counter(l for f in cfg_res['functions'].values() for b in f.blocks for l in b.asm_lines)),
    }

    def v_str(k, v):
        if k in ['asm_counts_per_function']:
            return '{\n        %s\n    }' % '\n        '.join(['%s: %s,' % (repr(k), '{\n            %s\n        }' % '\n            '.join(['%s: %s,' % (repr(k), repr(v2)) for k, v2 in v1.items()])) for k, v1 in v.items()])
        elif k in ['asm_counts_per_block', 'asm_counts']:
            return '{\n        %s\n    }' % '\n        '.join(['%s: %s,' % (repr(k), repr(v)) for k, v in v.items()])
        return repr(v)
    print_str = '{\n    %s\n}' % '\n    '.join(['%s: %s,' % (repr(k), v_str(k, v)) for k, v in expected.items()])

    return print_str.replace('\n', '\n' + tab_space)


_FILE_STR = """\"\"\"%s\"\"\"
import os
from bincfg import CFGFunction, CFGBasicBlock, CFG, CFGEdge, EdgeType, get_module
from ..fake_classes import FakeCFG, FakeCFGFunction


%s"""


if __name__ == '__main__':
    make_file()
