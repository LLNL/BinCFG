"""Conway's GOL binary analyzed with smda

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
    putchar('
');

    // print horizontal line
    for (int i = 0; i < boardWidth + 2; i++) {
        putchar('-');
    }
    putchar('
');

    // print board
    for (int r = 0; r < boardHeight; r++) {
        putchar('|');
        for (int c = 0; c < boardWidth; c++) {
            putchar(board[r][c] ? aliveChar : deadChar);
        }
        putchar('|');
        putchar('
');
    }

    // print horizontal line
    for (int i = 0; i < boardWidth + 2; i++) {
        putchar('-');
    }
    putchar('
');
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
import os
from bincfg import CFGFunction, CFGBasicBlock, CFG, CFGEdge, EdgeType, get_module
from ..fake_classes import FakeCFG, FakeCFGFunction


def get_manual_cfg(build_level):
    """Returns a manually built control flow graph
    
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
    """

    if build_level not in ['cfg', 'function', 'block']:
        raise ValueError("Bad build_level: %s" % repr(build_level))
    
    func_type = CFGFunction if build_level in ['function', 'cfg'] else FakeCFGFunction

    metadata = {'some': 'cfg-level', 134: ('metadata', True, None), (1, 2, 3): 'apples'}
        # Create the cfg object. This cfg has 20 functions, 108 basic blocks, 133 edges, and 555 lines of assembly.
    __auto_cfg = CFG(metadata=metadata) if build_level in ['cfg'] else FakeCFG(metadata=metadata)
    
    # Building all functions. Dictionary maps integer address to func_type() object
    __auto_functions = {
        4195664: func_type(parent_cfg=__auto_cfg, address=4195664, name='__UNNAMED_FUNC_4195664', is_extern_function=False, metadata={}),
        4195696: func_type(parent_cfg=__auto_cfg, address=4195696, name='__UNNAMED_FUNC_4195696', is_extern_function=False, metadata={}),
        4195712: func_type(parent_cfg=__auto_cfg, address=4195712, name='__UNNAMED_FUNC_4195712', is_extern_function=False, metadata={}),
        4195728: func_type(parent_cfg=__auto_cfg, address=4195728, name='__UNNAMED_FUNC_4195728', is_extern_function=False, metadata={}),
        4195744: func_type(parent_cfg=__auto_cfg, address=4195744, name='__UNNAMED_FUNC_4195744', is_extern_function=False, metadata={}),
        4195760: func_type(parent_cfg=__auto_cfg, address=4195760, name='__UNNAMED_FUNC_4195760', is_extern_function=False, metadata={}),
        4195776: func_type(parent_cfg=__auto_cfg, address=4195776, name='__UNNAMED_FUNC_4195776', is_extern_function=False, metadata={}),
        4195792: func_type(parent_cfg=__auto_cfg, address=4195792, name='__UNNAMED_FUNC_4195792', is_extern_function=False, metadata={}),
        4195840: func_type(parent_cfg=__auto_cfg, address=4195840, name='__UNNAMED_FUNC_4195840', is_extern_function=False, metadata={}),
        4195856: func_type(parent_cfg=__auto_cfg, address=4195856, name='__UNNAMED_FUNC_4195856', is_extern_function=False, metadata={}),
        4195904: func_type(parent_cfg=__auto_cfg, address=4195904, name='__UNNAMED_FUNC_4195904', is_extern_function=False, metadata={}),
        4195968: func_type(parent_cfg=__auto_cfg, address=4195968, name='__UNNAMED_FUNC_4195968', is_extern_function=False, metadata={}),
        4196016: func_type(parent_cfg=__auto_cfg, address=4196016, name='__UNNAMED_FUNC_4196016', is_extern_function=False, metadata={}),
        4196022: func_type(parent_cfg=__auto_cfg, address=4196022, name='__UNNAMED_FUNC_4196022', is_extern_function=False, metadata={}),
        4196209: func_type(parent_cfg=__auto_cfg, address=4196209, name='__UNNAMED_FUNC_4196209', is_extern_function=False, metadata={}),
        4196441: func_type(parent_cfg=__auto_cfg, address=4196441, name='__UNNAMED_FUNC_4196441', is_extern_function=False, metadata={}),
        4196937: func_type(parent_cfg=__auto_cfg, address=4196937, name='__UNNAMED_FUNC_4196937', is_extern_function=False, metadata={}),
        4197584: func_type(parent_cfg=__auto_cfg, address=4197584, name='__UNNAMED_FUNC_4197584', is_extern_function=False, metadata={}),
        4197696: func_type(parent_cfg=__auto_cfg, address=4197696, name='__UNNAMED_FUNC_4197696', is_extern_function=False, metadata={}),
        4197704: func_type(parent_cfg=__auto_cfg, address=4197704, name='__UNNAMED_FUNC_4197704', is_extern_function=False, metadata={}),
    }
    
    # Building basic blocks. Dictionary maps integer address to CFGBasicBlock() object
    __auto_blocks = {
        4195664: CFGBasicBlock(parent_function=__auto_functions[4195664], address=4195664, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400550: endbr64 ',
            '0x00400554: sub rsp, 8',
            '0x00400558: mov rax, qword ptr [rip + 0x201a91]',
            '0x0040055f: test rax, rax',
            '0x00400562: je 0x400566',
        ]),
        4195684: CFGBasicBlock(parent_function=__auto_functions[4195664], address=4195684, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400564: call rax',
        ]),
        4195686: CFGBasicBlock(parent_function=__auto_functions[4195664], address=4195686, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400566: add rsp, 8',
            '0x0040056a: ret ',
        ]),
        4195696: CFGBasicBlock(parent_function=__auto_functions[4195696], address=4195696, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400570: push qword ptr [rip + 0x201a92]',
            '0x00400576: jmp qword ptr [rip + 0x201a94]',
        ]),
        4195712: CFGBasicBlock(parent_function=__auto_functions[4195712], address=4195712, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400580: jmp qword ptr [rip + 0x201a92]',
        ]),
        4195728: CFGBasicBlock(parent_function=__auto_functions[4195728], address=4195728, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400590: jmp qword ptr [rip + 0x201a8a]',
        ]),
        4195744: CFGBasicBlock(parent_function=__auto_functions[4195744], address=4195744, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004005a0: jmp qword ptr [rip + 0x201a82]',
        ]),
        4195760: CFGBasicBlock(parent_function=__auto_functions[4195760], address=4195760, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004005b0: jmp qword ptr [rip + 0x201a7a]',
        ]),
        4195776: CFGBasicBlock(parent_function=__auto_functions[4195776], address=4195776, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004005c0: jmp qword ptr [rip + 0x201a72]',
        ]),
        4195792: CFGBasicBlock(parent_function=__auto_functions[4195792], address=4195792, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004005d0: endbr64 ',
            '0x004005d4: xor ebp, ebp',
            '0x004005d6: mov r9, rdx',
            '0x004005d9: pop rsi',
            '0x004005da: mov rdx, rsp',
            '0x004005dd: and rsp, 0xfffffffffffffff0',
            '0x004005e1: push rax',
            '0x004005e2: push rsp',
            '0x004005e3: mov r8, 0x400d40',
            '0x004005ea: mov rcx, 0x400cd0',
            '0x004005f1: mov rdi, 0x400a49',
            '0x004005f8: call qword ptr [rip + 0x2019ea]',
            '0x004005fe: hlt ',
        ]),
        4195840: CFGBasicBlock(parent_function=__auto_functions[4195840], address=4195840, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400600: endbr64 ',
            '0x00400604: ret ',
        ]),
        4195856: CFGBasicBlock(parent_function=__auto_functions[4195856], address=4195856, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400610: lea rdi, [rip + 0x201a31]',
            '0x00400617: lea rax, [rip + 0x201a2a]',
            '0x0040061e: cmp rax, rdi',
            '0x00400621: je 0x400638',
        ]),
        4195875: CFGBasicBlock(parent_function=__auto_functions[4195856], address=4195875, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400623: mov rax, qword ptr [rip + 0x2019b6]',
            '0x0040062a: test rax, rax',
            '0x0040062d: je 0x400638',
        ]),
        4195887: CFGBasicBlock(parent_function=__auto_functions[4195856], address=4195887, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040062f: jmp rax',
        ]),
        4195896: CFGBasicBlock(parent_function=__auto_functions[4195856], address=4195896, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400638: ret ',
        ]),
        4195904: CFGBasicBlock(parent_function=__auto_functions[4195904], address=4195904, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400640: lea rdi, [rip + 0x201a01]',
            '0x00400647: lea rsi, [rip + 0x2019fa]',
            '0x0040064e: sub rsi, rdi',
            '0x00400651: sar rsi, 3',
            '0x00400655: mov rax, rsi',
            '0x00400658: shr rax, 0x3f',
            '0x0040065c: add rsi, rax',
            '0x0040065f: sar rsi, 1',
            '0x00400662: je 0x400678',
        ]),
        4195940: CFGBasicBlock(parent_function=__auto_functions[4195904], address=4195940, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400664: mov rax, qword ptr [rip + 0x20198d]',
            '0x0040066b: test rax, rax',
            '0x0040066e: je 0x400678',
        ]),
        4195952: CFGBasicBlock(parent_function=__auto_functions[4195904], address=4195952, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400670: jmp rax',
        ]),
        4195960: CFGBasicBlock(parent_function=__auto_functions[4195904], address=4195960, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400678: ret ',
        ]),
        4195968: CFGBasicBlock(parent_function=__auto_functions[4195968], address=4195968, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400680: endbr64 ',
            '0x00400684: cmp byte ptr [rip + 0x2019b9], 0',
            '0x0040068b: jne 0x4006a0',
        ]),
        4195981: CFGBasicBlock(parent_function=__auto_functions[4195968], address=4195981, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040068d: push rbp',
            '0x0040068e: mov rbp, rsp',
            '0x00400691: call 0x400610',
            '0x00400696: mov byte ptr [rip + 0x2019a7], 1',
            '0x0040069d: pop rbp',
            '0x0040069e: ret ',
        ]),
        4196000: CFGBasicBlock(parent_function=__auto_functions[4195968], address=4196000, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004006a0: ret ',
        ]),
        4196016: CFGBasicBlock(parent_function=__auto_functions[4196016], address=4196016, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004006b0: endbr64 ',
            '0x004006b4: jmp 0x400640',
        ]),
        4196022: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196022, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004006b6: push rbp',
            '0x004006b7: mov rbp, rsp',
            '0x004006ba: sub rsp, 0x30',
            '0x004006be: mov qword ptr [rbp - 0x28], rdi',
            '0x004006c2: cmp qword ptr [rbp - 0x28], 0',
            '0x004006c7: jns 0x4006de',
        ]),
        4196041: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196041, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004006c9: call 0x400590',
            '0x004006ce: mov dword ptr [rax], 0x16',
            '0x004006d4: mov eax, 0xffffffff',
            '0x004006d9: jmp 0x40076f',
        ]),
        4196062: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196062, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004006de: mov rcx, qword ptr [rbp - 0x28]',
            '0x004006e2: movabs rdx, 0x20c49ba5e353f7cf',
            '0x004006ec: mov rax, rcx',
            '0x004006ef: imul rdx',
            '0x004006f2: sar rdx, 7',
            '0x004006f6: mov rax, rcx',
            '0x004006f9: sar rax, 0x3f',
            '0x004006fd: sub rdx, rax',
            '0x00400700: mov rax, rdx',
            '0x00400703: mov qword ptr [rbp - 0x20], rax',
            '0x00400707: mov rcx, qword ptr [rbp - 0x28]',
            '0x0040070b: movabs rdx, 0x20c49ba5e353f7cf',
            '0x00400715: mov rax, rcx',
            '0x00400718: imul rdx',
            '0x0040071b: sar rdx, 7',
            '0x0040071f: mov rax, rcx',
            '0x00400722: sar rax, 0x3f',
            '0x00400726: sub rdx, rax',
            '0x00400729: mov rax, rdx',
            '0x0040072c: imul rax, rax, 0x3e8',
            '0x00400733: sub rcx, rax',
            '0x00400736: mov rax, rcx',
            '0x00400739: imul rax, rax, 0xf4240',
            '0x00400740: mov qword ptr [rbp - 0x18], rax',
        ]),
        4196164: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196164, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400744: lea rdx, [rbp - 0x20]',
            '0x00400748: lea rax, [rbp - 0x20]',
            '0x0040074c: mov rsi, rdx',
            '0x0040074f: mov rdi, rax',
            '0x00400752: call 0x4005a0',
            '0x00400757: mov dword ptr [rbp - 4], eax',
            '0x0040075a: cmp dword ptr [rbp - 4], 0',
            '0x0040075e: je 0x40076c',
        ]),
        4196192: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196192, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400760: call 0x400590',
            '0x00400765: mov eax, dword ptr [rax]',
            '0x00400767: cmp eax, 4',
            '0x0040076a: je 0x400744',
        ]),
        4196204: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196204, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040076c: mov eax, dword ptr [rbp - 4]',
        ]),
        4196207: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196207, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040076f: leave ',
            '0x00400770: ret ',
        ]),
        4196209: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196209, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400771: push rbp',
            '0x00400772: mov rbp, rsp',
            '0x00400775: sub rsp, 0x20',
            '0x00400779: mov qword ptr [rbp - 0x18], rdi',
            '0x0040077d: mov edi, 0xa',
            '0x00400782: call 0x400580',
            '0x00400787: mov dword ptr [rbp - 4], 0',
            '0x0040078e: jmp 0x40079e',
        ]),
        4196240: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196240, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400790: mov edi, 0x2d',
            '0x00400795: call 0x400580',
            '0x0040079a: add dword ptr [rbp - 4], 1',
        ]),
        4196254: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196254, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040079e: cmp dword ptr [rbp - 4], 0x1f',
            '0x004007a2: jle 0x400790',
        ]),
        4196260: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196260, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004007a4: mov edi, 0xa',
            '0x004007a9: call 0x400580',
            '0x004007ae: mov dword ptr [rbp - 8], 0',
            '0x004007b5: jmp 0x400829',
        ]),
        4196279: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196279, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004007b7: mov edi, 0x7c',
            '0x004007bc: call 0x400580',
            '0x004007c1: mov dword ptr [rbp - 0xc], 0',
            '0x004007c8: jmp 0x40080b',
        ]),
        4196298: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196298, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004007ca: mov eax, dword ptr [rbp - 8]',
            '0x004007cd: movsxd rdx, eax',
            '0x004007d0: mov rax, rdx',
            '0x004007d3: shl rax, 4',
            '0x004007d7: sub rax, rdx',
            '0x004007da: add rax, rax',
            '0x004007dd: mov rdx, rax',
            '0x004007e0: mov rax, qword ptr [rbp - 0x18]',
            '0x004007e4: add rdx, rax',
            '0x004007e7: mov eax, dword ptr [rbp - 0xc]',
            '0x004007ea: cdqe ',
            '0x004007ec: movzx eax, byte ptr [rdx + rax]',
            '0x004007f0: test al, al',
            '0x004007f2: je 0x4007fb',
        ]),
        4196340: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196340, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004007f4: mov eax, 0x58',
            '0x004007f9: jmp 0x400800',
        ]),
        4196347: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196347, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004007fb: mov eax, 0x20',
        ]),
        4196352: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196352, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400800: mov edi, eax',
            '0x00400802: call 0x400580',
            '0x00400807: add dword ptr [rbp - 0xc], 1',
        ]),
        4196363: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196363, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040080b: cmp dword ptr [rbp - 0xc], 0x1d',
            '0x0040080f: jle 0x4007ca',
        ]),
        4196369: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196369, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400811: mov edi, 0x7c',
            '0x00400816: call 0x400580',
            '0x0040081b: mov edi, 0xa',
            '0x00400820: call 0x400580',
            '0x00400825: add dword ptr [rbp - 8], 1',
        ]),
        4196393: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196393, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400829: cmp dword ptr [rbp - 8], 0xe',
            '0x0040082d: jle 0x4007b7',
        ]),
        4196399: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196399, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040082f: mov dword ptr [rbp - 0x10], 0',
            '0x00400836: jmp 0x400846',
        ]),
        4196408: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196408, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400838: mov edi, 0x2d',
            '0x0040083d: call 0x400580',
            '0x00400842: add dword ptr [rbp - 0x10], 1',
        ]),
        4196422: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196422, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400846: cmp dword ptr [rbp - 0x10], 0x1f',
            '0x0040084a: jle 0x400838',
        ]),
        4196428: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196428, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040084c: mov edi, 0xa',
            '0x00400851: call 0x400580',
            '0x00400856: nop ',
            '0x00400857: leave ',
            '0x00400858: ret ',
        ]),
        4196441: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196441, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400859: push rbp',
            '0x0040085a: mov rbp, rsp',
            '0x0040085d: mov qword ptr [rbp - 0x18], rdi',
            '0x00400861: mov dword ptr [rbp - 0x1c], esi',
            '0x00400864: mov dword ptr [rbp - 0x20], edx',
            '0x00400867: mov dword ptr [rbp - 4], 0',
            '0x0040086e: cmp dword ptr [rbp - 0x1c], 0',
            '0x00400872: jle 0x4008ac',
        ]),
        4196468: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196468, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400874: cmp dword ptr [rbp - 0x20], 0',
            '0x00400878: jle 0x4008ac',
        ]),
        4196474: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196474, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040087a: mov eax, dword ptr [rbp - 0x1c]',
            '0x0040087d: movsxd rdx, eax',
            '0x00400880: mov rax, rdx',
            '0x00400883: shl rax, 4',
            '0x00400887: sub rax, rdx',
            '0x0040088a: add rax, rax',
            '0x0040088d: lea rdx, [rax - 0x1e]',
            '0x00400891: mov rax, qword ptr [rbp - 0x18]',
            '0x00400895: add rdx, rax',
            '0x00400898: mov eax, dword ptr [rbp - 0x20]',
            '0x0040089b: sub eax, 1',
            '0x0040089e: cdqe ',
            '0x004008a0: movzx eax, byte ptr [rdx + rax]',
            '0x004008a4: test al, al',
            '0x004008a6: je 0x4008ac',
        ]),
        4196520: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196520, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004008a8: add dword ptr [rbp - 4], 1',
        ]),
        4196524: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196524, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004008ac: cmp dword ptr [rbp - 0x1c], 0',
            '0x004008b0: jle 0x4008e1',
        ]),
        4196530: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196530, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004008b2: mov eax, dword ptr [rbp - 0x1c]',
            '0x004008b5: movsxd rdx, eax',
            '0x004008b8: mov rax, rdx',
            '0x004008bb: shl rax, 4',
            '0x004008bf: sub rax, rdx',
            '0x004008c2: add rax, rax',
            '0x004008c5: lea rdx, [rax - 0x1e]',
            '0x004008c9: mov rax, qword ptr [rbp - 0x18]',
            '0x004008cd: add rdx, rax',
            '0x004008d0: mov eax, dword ptr [rbp - 0x20]',
            '0x004008d3: cdqe ',
            '0x004008d5: movzx eax, byte ptr [rdx + rax]',
            '0x004008d9: test al, al',
            '0x004008db: je 0x4008e1',
        ]),
        4196573: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196573, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004008dd: add dword ptr [rbp - 4], 1',
        ]),
        4196577: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196577, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004008e1: cmp dword ptr [rbp - 0x1c], 0',
            '0x004008e5: jle 0x40091f',
        ]),
        4196583: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196583, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004008e7: cmp dword ptr [rbp - 0x20], 0x1c',
            '0x004008eb: jg 0x40091f',
        ]),
        4196589: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196589, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004008ed: mov eax, dword ptr [rbp - 0x1c]',
            '0x004008f0: movsxd rdx, eax',
            '0x004008f3: mov rax, rdx',
            '0x004008f6: shl rax, 4',
            '0x004008fa: sub rax, rdx',
            '0x004008fd: add rax, rax',
            '0x00400900: lea rdx, [rax - 0x1e]',
            '0x00400904: mov rax, qword ptr [rbp - 0x18]',
            '0x00400908: add rdx, rax',
            '0x0040090b: mov eax, dword ptr [rbp - 0x20]',
            '0x0040090e: add eax, 1',
            '0x00400911: cdqe ',
            '0x00400913: movzx eax, byte ptr [rdx + rax]',
            '0x00400917: test al, al',
            '0x00400919: je 0x40091f',
        ]),
        4196635: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196635, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040091b: add dword ptr [rbp - 4], 1',
        ]),
        4196639: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196639, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040091f: cmp dword ptr [rbp - 0x20], 0',
            '0x00400923: jle 0x400956',
        ]),
        4196645: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196645, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400925: mov eax, dword ptr [rbp - 0x1c]',
            '0x00400928: movsxd rdx, eax',
            '0x0040092b: mov rax, rdx',
            '0x0040092e: shl rax, 4',
            '0x00400932: sub rax, rdx',
            '0x00400935: add rax, rax',
            '0x00400938: mov rdx, rax',
            '0x0040093b: mov rax, qword ptr [rbp - 0x18]',
            '0x0040093f: add rdx, rax',
            '0x00400942: mov eax, dword ptr [rbp - 0x20]',
            '0x00400945: sub eax, 1',
            '0x00400948: cdqe ',
            '0x0040094a: movzx eax, byte ptr [rdx + rax]',
            '0x0040094e: test al, al',
            '0x00400950: je 0x400956',
        ]),
        4196690: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196690, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400952: add dword ptr [rbp - 4], 1',
        ]),
        4196694: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196694, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400956: cmp dword ptr [rbp - 0x20], 0x1c',
            '0x0040095a: jg 0x40098d',
        ]),
        4196700: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196700, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040095c: mov eax, dword ptr [rbp - 0x1c]',
            '0x0040095f: movsxd rdx, eax',
            '0x00400962: mov rax, rdx',
            '0x00400965: shl rax, 4',
            '0x00400969: sub rax, rdx',
            '0x0040096c: add rax, rax',
            '0x0040096f: mov rdx, rax',
            '0x00400972: mov rax, qword ptr [rbp - 0x18]',
            '0x00400976: add rdx, rax',
            '0x00400979: mov eax, dword ptr [rbp - 0x20]',
            '0x0040097c: add eax, 1',
            '0x0040097f: cdqe ',
            '0x00400981: movzx eax, byte ptr [rdx + rax]',
            '0x00400985: test al, al',
            '0x00400987: je 0x40098d',
        ]),
        4196745: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196745, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400989: add dword ptr [rbp - 4], 1',
        ]),
        4196749: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196749, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x0040098d: cmp dword ptr [rbp - 0x1c], 0xd',
            '0x00400991: jg 0x4009cd',
        ]),
        4196755: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196755, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400993: cmp dword ptr [rbp - 0x20], 0',
            '0x00400997: jle 0x4009cd',
        ]),
        4196761: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196761, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400999: mov eax, dword ptr [rbp - 0x1c]',
            '0x0040099c: cdqe ',
            '0x0040099e: lea rdx, [rax + 1]',
            '0x004009a2: mov rax, rdx',
            '0x004009a5: shl rax, 4',
            '0x004009a9: sub rax, rdx',
            '0x004009ac: add rax, rax',
            '0x004009af: mov rdx, rax',
            '0x004009b2: mov rax, qword ptr [rbp - 0x18]',
            '0x004009b6: add rdx, rax',
            '0x004009b9: mov eax, dword ptr [rbp - 0x20]',
            '0x004009bc: sub eax, 1',
            '0x004009bf: cdqe ',
            '0x004009c1: movzx eax, byte ptr [rdx + rax]',
            '0x004009c5: test al, al',
            '0x004009c7: je 0x4009cd',
        ]),
        4196809: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196809, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004009c9: add dword ptr [rbp - 4], 1',
        ]),
        4196813: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196813, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004009cd: cmp dword ptr [rbp - 0x1c], 0xd',
            '0x004009d1: jg 0x400a04',
        ]),
        4196819: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196819, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x004009d3: mov eax, dword ptr [rbp - 0x1c]',
            '0x004009d6: cdqe ',
            '0x004009d8: lea rdx, [rax + 1]',
            '0x004009dc: mov rax, rdx',
            '0x004009df: shl rax, 4',
            '0x004009e3: sub rax, rdx',
            '0x004009e6: add rax, rax',
            '0x004009e9: mov rdx, rax',
            '0x004009ec: mov rax, qword ptr [rbp - 0x18]',
            '0x004009f0: add rdx, rax',
            '0x004009f3: mov eax, dword ptr [rbp - 0x20]',
            '0x004009f6: cdqe ',
            '0x004009f8: movzx eax, byte ptr [rdx + rax]',
            '0x004009fc: test al, al',
            '0x004009fe: je 0x400a04',
        ]),
        4196864: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196864, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400a00: add dword ptr [rbp - 4], 1',
        ]),
        4196868: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196868, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400a04: cmp dword ptr [rbp - 0x1c], 0xd',
            '0x00400a08: jg 0x400a44',
        ]),
        4196874: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196874, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400a0a: cmp dword ptr [rbp - 0x20], 0x1c',
            '0x00400a0e: jg 0x400a44',
        ]),
        4196880: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196880, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400a10: mov eax, dword ptr [rbp - 0x1c]',
            '0x00400a13: cdqe ',
            '0x00400a15: lea rdx, [rax + 1]',
            '0x00400a19: mov rax, rdx',
            '0x00400a1c: shl rax, 4',
            '0x00400a20: sub rax, rdx',
            '0x00400a23: add rax, rax',
            '0x00400a26: mov rdx, rax',
            '0x00400a29: mov rax, qword ptr [rbp - 0x18]',
            '0x00400a2d: add rdx, rax',
            '0x00400a30: mov eax, dword ptr [rbp - 0x20]',
            '0x00400a33: add eax, 1',
            '0x00400a36: cdqe ',
            '0x00400a38: movzx eax, byte ptr [rdx + rax]',
            '0x00400a3c: test al, al',
            '0x00400a3e: je 0x400a44',
        ]),
        4196928: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196928, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400a40: add dword ptr [rbp - 4], 1',
        ]),
        4196932: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196932, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400a44: mov eax, dword ptr [rbp - 4]',
            '0x00400a47: pop rbp',
            '0x00400a48: ret ',
        ]),
        4196937: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4196937, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400a49: push rbp',
            '0x00400a4a: mov rbp, rsp',
            '0x00400a4d: sub rsp, 0x3b0',
            '0x00400a54: mov edi, 0x3039',
            '0x00400a59: call 0x4005b0',
            '0x00400a5e: mov dword ptr [rbp - 4], 0',
            '0x00400a65: jmp 0x400ace',
        ]),
        4196967: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4196967, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400a67: mov dword ptr [rbp - 8], 0',
            '0x00400a6e: jmp 0x400ac4',
        ]),
        4196976: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4196976, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400a70: call 0x4005c0',
            '0x00400a75: cvtsi2ss xmm0, eax',
            '0x00400a79: movss xmm1, dword ptr [rip + 0x2e7]',
            '0x00400a81: divss xmm0, xmm1',
            '0x00400a85: movaps xmm1, xmm0',
            '0x00400a88: movss xmm0, dword ptr [rip + 0x2dc]',
            '0x00400a90: comiss xmm0, xmm1',
            '0x00400a93: seta al',
            '0x00400a96: mov esi, eax',
            '0x00400a98: mov eax, dword ptr [rbp - 8]',
            '0x00400a9b: movsxd rcx, eax',
            '0x00400a9e: mov eax, dword ptr [rbp - 4]',
            '0x00400aa1: movsxd rdx, eax',
            '0x00400aa4: mov rax, rdx',
            '0x00400aa7: shl rax, 4',
            '0x00400aab: sub rax, rdx',
            '0x00400aae: add rax, rax',
            '0x00400ab1: add rax, rbp',
            '0x00400ab4: add rax, rcx',
            '0x00400ab7: sub rax, 0x1e0',
            '0x00400abd: mov byte ptr [rax], sil',
            '0x00400ac0: add dword ptr [rbp - 8], 1',
        ]),
        4197060: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197060, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400ac4: cmp dword ptr [rbp - 8], 0x1d',
            '0x00400ac8: jle 0x400a70',
        ]),
        4197066: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197066, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400aca: add dword ptr [rbp - 4], 1',
        ]),
        4197070: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197070, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400ace: cmp dword ptr [rbp - 4], 0xe',
            '0x00400ad2: jle 0x400a67',
        ]),
        4197076: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197076, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400ad4: lea rax, [rbp - 0x1e0]',
            '0x00400adb: mov rdi, rax',
            '0x00400ade: call 0x400771',
            '0x00400ae3: mov dword ptr [rbp - 0xc], 0',
            '0x00400aea: jmp 0x400c37',
        ]),
        4197103: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197103, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400aef: mov dword ptr [rbp - 0x10], 0',
            '0x00400af6: jmp 0x400c29',
        ]),
        4197115: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197115, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400afb: mov edx, dword ptr [rbp - 0x10]',
            '0x00400afe: mov ecx, dword ptr [rbp - 0xc]',
            '0x00400b01: lea rax, [rbp - 0x1e0]',
            '0x00400b08: mov esi, ecx',
            '0x00400b0a: mov rdi, rax',
            '0x00400b0d: call 0x400859',
            '0x00400b12: mov dword ptr [rbp - 0x1c], eax',
            '0x00400b15: mov eax, dword ptr [rbp - 0x10]',
            '0x00400b18: movsxd rcx, eax',
            '0x00400b1b: mov eax, dword ptr [rbp - 0xc]',
            '0x00400b1e: movsxd rdx, eax',
            '0x00400b21: mov rax, rdx',
            '0x00400b24: shl rax, 4',
            '0x00400b28: sub rax, rdx',
            '0x00400b2b: add rax, rax',
            '0x00400b2e: add rax, rbp',
            '0x00400b31: add rax, rcx',
            '0x00400b34: sub rax, 0x1e0',
            '0x00400b3a: movzx eax, byte ptr [rax]',
            '0x00400b3d: test al, al',
            '0x00400b3f: je 0x400b7a',
        ]),
        4197185: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197185, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400b41: cmp dword ptr [rbp - 0x1c], 1',
            '0x00400b45: jle 0x400b4d',
        ]),
        4197191: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197191, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400b47: cmp dword ptr [rbp - 0x1c], 3',
            '0x00400b4b: jle 0x400b7a',
        ]),
        4197197: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197197, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400b4d: mov eax, dword ptr [rbp - 0x10]',
            '0x00400b50: movsxd rcx, eax',
            '0x00400b53: mov eax, dword ptr [rbp - 0xc]',
            '0x00400b56: movsxd rdx, eax',
            '0x00400b59: mov rax, rdx',
            '0x00400b5c: shl rax, 4',
            '0x00400b60: sub rax, rdx',
            '0x00400b63: add rax, rax',
            '0x00400b66: add rax, rbp',
            '0x00400b69: add rax, rcx',
            '0x00400b6c: sub rax, 0x3b0',
            '0x00400b72: mov byte ptr [rax], 0',
            '0x00400b75: jmp 0x400c25',
        ]),
        4197242: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197242, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400b7a: mov eax, dword ptr [rbp - 0x10]',
            '0x00400b7d: movsxd rcx, eax',
            '0x00400b80: mov eax, dword ptr [rbp - 0xc]',
            '0x00400b83: movsxd rdx, eax',
            '0x00400b86: mov rax, rdx',
            '0x00400b89: shl rax, 4',
            '0x00400b8d: sub rax, rdx',
            '0x00400b90: add rax, rax',
            '0x00400b93: add rax, rbp',
            '0x00400b96: add rax, rcx',
            '0x00400b99: sub rax, 0x1e0',
            '0x00400b9f: movzx eax, byte ptr [rax]',
            '0x00400ba2: test al, al',
            '0x00400ba4: jne 0x400bd6',
        ]),
        4197286: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197286, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400ba6: cmp dword ptr [rbp - 0x1c], 3',
            '0x00400baa: jne 0x400bd6',
        ]),
        4197292: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197292, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400bac: mov eax, dword ptr [rbp - 0x10]',
            '0x00400baf: movsxd rcx, eax',
            '0x00400bb2: mov eax, dword ptr [rbp - 0xc]',
            '0x00400bb5: movsxd rdx, eax',
            '0x00400bb8: mov rax, rdx',
            '0x00400bbb: shl rax, 4',
            '0x00400bbf: sub rax, rdx',
            '0x00400bc2: add rax, rax',
            '0x00400bc5: add rax, rbp',
            '0x00400bc8: add rax, rcx',
            '0x00400bcb: sub rax, 0x3b0',
            '0x00400bd1: mov byte ptr [rax], 1',
            '0x00400bd4: jmp 0x400c25',
        ]),
        4197334: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197334, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400bd6: mov eax, dword ptr [rbp - 0x10]',
            '0x00400bd9: movsxd rcx, eax',
            '0x00400bdc: mov eax, dword ptr [rbp - 0xc]',
            '0x00400bdf: movsxd rdx, eax',
            '0x00400be2: mov rax, rdx',
            '0x00400be5: shl rax, 4',
            '0x00400be9: sub rax, rdx',
            '0x00400bec: add rax, rax',
            '0x00400bef: add rax, rbp',
            '0x00400bf2: add rax, rcx',
            '0x00400bf5: sub rax, 0x1e0',
            '0x00400bfb: movzx ecx, byte ptr [rax]',
            '0x00400bfe: mov eax, dword ptr [rbp - 0x10]',
            '0x00400c01: movsxd rsi, eax',
            '0x00400c04: mov eax, dword ptr [rbp - 0xc]',
            '0x00400c07: movsxd rdx, eax',
            '0x00400c0a: mov rax, rdx',
            '0x00400c0d: shl rax, 4',
            '0x00400c11: sub rax, rdx',
            '0x00400c14: add rax, rax',
            '0x00400c17: add rax, rbp',
            '0x00400c1a: add rax, rsi',
            '0x00400c1d: sub rax, 0x3b0',
            '0x00400c23: mov byte ptr [rax], cl',
        ]),
        4197413: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197413, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400c25: add dword ptr [rbp - 0x10], 1',
        ]),
        4197417: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197417, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400c29: cmp dword ptr [rbp - 0x10], 0x1d',
            '0x00400c2d: jle 0x400afb',
        ]),
        4197427: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197427, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400c33: add dword ptr [rbp - 0xc], 1',
        ]),
        4197431: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197431, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400c37: cmp dword ptr [rbp - 0xc], 0xe',
            '0x00400c3b: jle 0x400aef',
        ]),
        4197441: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197441, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400c41: mov dword ptr [rbp - 0x14], 0',
            '0x00400c48: jmp 0x400cb0',
        ]),
        4197450: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197450, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400c4a: mov dword ptr [rbp - 0x18], 0',
            '0x00400c51: jmp 0x400ca6',
        ]),
        4197459: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197459, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400c53: mov eax, dword ptr [rbp - 0x18]',
            '0x00400c56: movsxd rcx, eax',
            '0x00400c59: mov eax, dword ptr [rbp - 0x14]',
            '0x00400c5c: movsxd rdx, eax',
            '0x00400c5f: mov rax, rdx',
            '0x00400c62: shl rax, 4',
            '0x00400c66: sub rax, rdx',
            '0x00400c69: add rax, rax',
            '0x00400c6c: add rax, rbp',
            '0x00400c6f: add rax, rcx',
            '0x00400c72: sub rax, 0x3b0',
            '0x00400c78: movzx ecx, byte ptr [rax]',
            '0x00400c7b: mov eax, dword ptr [rbp - 0x18]',
            '0x00400c7e: movsxd rsi, eax',
            '0x00400c81: mov eax, dword ptr [rbp - 0x14]',
            '0x00400c84: movsxd rdx, eax',
            '0x00400c87: mov rax, rdx',
            '0x00400c8a: shl rax, 4',
            '0x00400c8e: sub rax, rdx',
            '0x00400c91: add rax, rax',
            '0x00400c94: add rax, rbp',
            '0x00400c97: add rax, rsi',
            '0x00400c9a: sub rax, 0x1e0',
            '0x00400ca0: mov byte ptr [rax], cl',
            '0x00400ca2: add dword ptr [rbp - 0x18], 1',
        ]),
        4197542: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197542, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400ca6: cmp dword ptr [rbp - 0x18], 0x1d',
            '0x00400caa: jle 0x400c53',
        ]),
        4197548: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197548, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400cac: add dword ptr [rbp - 0x14], 1',
        ]),
        4197552: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197552, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400cb0: cmp dword ptr [rbp - 0x14], 0xe',
            '0x00400cb4: jle 0x400c4a',
        ]),
        4197558: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197558, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400cb6: mov edi, 0x1f4',
            '0x00400cbb: call 0x4006b6',
            '0x00400cc0: jmp 0x400ad4',
        ]),
        4197584: CFGBasicBlock(parent_function=__auto_functions[4197584], address=4197584, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400cd0: endbr64 ',
            '0x00400cd4: push r15',
            '0x00400cd6: mov r15, rdx',
            '0x00400cd9: push r14',
            '0x00400cdb: mov r14, rsi',
            '0x00400cde: push r13',
            '0x00400ce0: mov r13d, edi',
            '0x00400ce3: push r12',
            '0x00400ce5: lea r12, [rip + 0x201114]',
            '0x00400cec: push rbp',
            '0x00400ced: lea rbp, [rip + 0x201114]',
            '0x00400cf4: push rbx',
            '0x00400cf5: sub rbp, r12',
            '0x00400cf8: sub rsp, 8',
            '0x00400cfc: call 0x400550',
            '0x00400d01: sar rbp, 3',
            '0x00400d05: je 0x400d26',
        ]),
        4197639: CFGBasicBlock(parent_function=__auto_functions[4197584], address=4197639, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400d07: xor ebx, ebx',
            '0x00400d09: nop dword ptr [rax]',
        ]),
        4197648: CFGBasicBlock(parent_function=__auto_functions[4197584], address=4197648, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400d10: mov rdx, r15',
            '0x00400d13: mov rsi, r14',
            '0x00400d16: mov edi, r13d',
            '0x00400d19: call qword ptr [r12 + rbx*8]',
            '0x00400d1d: add rbx, 1',
            '0x00400d21: cmp rbp, rbx',
            '0x00400d24: jne 0x400d10',
        ]),
        4197670: CFGBasicBlock(parent_function=__auto_functions[4197584], address=4197670, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400d26: add rsp, 8',
            '0x00400d2a: pop rbx',
            '0x00400d2b: pop rbp',
            '0x00400d2c: pop r12',
            '0x00400d2e: pop r13',
            '0x00400d30: pop r14',
            '0x00400d32: pop r15',
            '0x00400d34: ret ',
        ]),
        4197696: CFGBasicBlock(parent_function=__auto_functions[4197696], address=4197696, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400d40: endbr64 ',
            '0x00400d44: ret ',
        ]),
        4197704: CFGBasicBlock(parent_function=__auto_functions[4197704], address=4197704, asm_memory_addresses=[], metadata={}, asm_lines=[
            '0x00400d48: endbr64 ',
            '0x00400d4c: sub rsp, 8',
            '0x00400d50: add rsp, 8',
            '0x00400d54: ret ',
        ]),
    }
    
    # Building all edges
    __auto_blocks[4195664].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195664], to_block=__auto_blocks[4195686], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4195664], to_block=__auto_blocks[4195684], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4195684].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195684], to_block=__auto_blocks[4195686], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4195686].edges_out = set([
        
    ])
    
    __auto_blocks[4195696].edges_out = set([
        
    ])
    
    __auto_blocks[4195712].edges_out = set([
        
    ])
    
    __auto_blocks[4195728].edges_out = set([
        
    ])
    
    __auto_blocks[4195744].edges_out = set([
        
    ])
    
    __auto_blocks[4195760].edges_out = set([
        
    ])
    
    __auto_blocks[4195776].edges_out = set([
        
    ])
    
    __auto_blocks[4195792].edges_out = set([
        
    ])
    
    __auto_blocks[4195840].edges_out = set([
        
    ])
    
    __auto_blocks[4195856].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195856], to_block=__auto_blocks[4195875], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4195856], to_block=__auto_blocks[4195896], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4195875].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195875], to_block=__auto_blocks[4195887], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4195875], to_block=__auto_blocks[4195896], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4195887].edges_out = set([
        
    ])
    
    __auto_blocks[4195896].edges_out = set([
        
    ])
    
    __auto_blocks[4195904].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195904], to_block=__auto_blocks[4195960], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4195904], to_block=__auto_blocks[4195940], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4195940].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195940], to_block=__auto_blocks[4195960], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4195940], to_block=__auto_blocks[4195952], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4195952].edges_out = set([
        
    ])
    
    __auto_blocks[4195960].edges_out = set([
        
    ])
    
    __auto_blocks[4195968].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195968], to_block=__auto_blocks[4195981], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4195968], to_block=__auto_blocks[4196000], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4195981].edges_out = set([
        
    ])
    
    __auto_blocks[4196000].edges_out = set([
        
    ])
    
    __auto_blocks[4196016].edges_out = set([
        
    ])
    
    __auto_blocks[4196022].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196022], to_block=__auto_blocks[4196041], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196022], to_block=__auto_blocks[4196062], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196041].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196041], to_block=__auto_blocks[4196207], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196062].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196062], to_block=__auto_blocks[4196164], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196164].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196164], to_block=__auto_blocks[4196204], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196164], to_block=__auto_blocks[4196192], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196192].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196192], to_block=__auto_blocks[4196204], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196192], to_block=__auto_blocks[4196164], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196204].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196204], to_block=__auto_blocks[4196207], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196207].edges_out = set([
        
    ])
    
    __auto_blocks[4196209].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196209], to_block=__auto_blocks[4196254], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196240].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196240], to_block=__auto_blocks[4196254], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196254].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196254], to_block=__auto_blocks[4196260], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196254], to_block=__auto_blocks[4196240], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196260].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196260], to_block=__auto_blocks[4196393], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196279].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196279], to_block=__auto_blocks[4196363], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196298].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196298], to_block=__auto_blocks[4196347], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196298], to_block=__auto_blocks[4196340], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196340].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196340], to_block=__auto_blocks[4196352], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196347].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196347], to_block=__auto_blocks[4196352], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196352].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196352], to_block=__auto_blocks[4196363], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196363].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196363], to_block=__auto_blocks[4196369], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196363], to_block=__auto_blocks[4196298], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196369].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196369], to_block=__auto_blocks[4196393], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196393].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196393], to_block=__auto_blocks[4196279], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196393], to_block=__auto_blocks[4196399], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196399].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196399], to_block=__auto_blocks[4196422], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196408].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196408], to_block=__auto_blocks[4196422], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196422].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196422], to_block=__auto_blocks[4196428], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196422], to_block=__auto_blocks[4196408], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196428].edges_out = set([
        
    ])
    
    __auto_blocks[4196441].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196441], to_block=__auto_blocks[4196524], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196441], to_block=__auto_blocks[4196468], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196468].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196468], to_block=__auto_blocks[4196474], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196468], to_block=__auto_blocks[4196524], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196474].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196474], to_block=__auto_blocks[4196524], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196474], to_block=__auto_blocks[4196520], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196520].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196520], to_block=__auto_blocks[4196524], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196524].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196524], to_block=__auto_blocks[4196577], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196524], to_block=__auto_blocks[4196530], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196530].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196530], to_block=__auto_blocks[4196577], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196530], to_block=__auto_blocks[4196573], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196573].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196573], to_block=__auto_blocks[4196577], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196577].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196577], to_block=__auto_blocks[4196639], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196577], to_block=__auto_blocks[4196583], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196583].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196583], to_block=__auto_blocks[4196589], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196583], to_block=__auto_blocks[4196639], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196589].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196589], to_block=__auto_blocks[4196635], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196589], to_block=__auto_blocks[4196639], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196635].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196635], to_block=__auto_blocks[4196639], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196639].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196639], to_block=__auto_blocks[4196694], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196639], to_block=__auto_blocks[4196645], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196645].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196645], to_block=__auto_blocks[4196694], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196645], to_block=__auto_blocks[4196690], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196690].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196690], to_block=__auto_blocks[4196694], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196694].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196694], to_block=__auto_blocks[4196700], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196694], to_block=__auto_blocks[4196749], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196700].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196700], to_block=__auto_blocks[4196745], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196700], to_block=__auto_blocks[4196749], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196745].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196745], to_block=__auto_blocks[4196749], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196749].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196749], to_block=__auto_blocks[4196813], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196749], to_block=__auto_blocks[4196755], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196755].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196755], to_block=__auto_blocks[4196813], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196755], to_block=__auto_blocks[4196761], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196761].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196761], to_block=__auto_blocks[4196813], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196761], to_block=__auto_blocks[4196809], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196809].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196809], to_block=__auto_blocks[4196813], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196813].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196813], to_block=__auto_blocks[4196868], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196813], to_block=__auto_blocks[4196819], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196819].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196819], to_block=__auto_blocks[4196864], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196819], to_block=__auto_blocks[4196868], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196864].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196864], to_block=__auto_blocks[4196868], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196868].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196868], to_block=__auto_blocks[4196874], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196868], to_block=__auto_blocks[4196932], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196874].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196874], to_block=__auto_blocks[4196932], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196874], to_block=__auto_blocks[4196880], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196880].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196880], to_block=__auto_blocks[4196932], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196880], to_block=__auto_blocks[4196928], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196928].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196928], to_block=__auto_blocks[4196932], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196932].edges_out = set([
        
    ])
    
    __auto_blocks[4196937].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196937], to_block=__auto_blocks[4197070], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196967].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196967], to_block=__auto_blocks[4197060], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4196976].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196976], to_block=__auto_blocks[4197060], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197060].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197060], to_block=__auto_blocks[4196976], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197060], to_block=__auto_blocks[4197066], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197066].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197066], to_block=__auto_blocks[4197070], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197070].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197070], to_block=__auto_blocks[4197076], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197070], to_block=__auto_blocks[4196967], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197076].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197076], to_block=__auto_blocks[4197431], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197103].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197103], to_block=__auto_blocks[4197417], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197115].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197115], to_block=__auto_blocks[4197242], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197115], to_block=__auto_blocks[4197185], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197185].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197185], to_block=__auto_blocks[4197197], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197185], to_block=__auto_blocks[4197191], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197191].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197191], to_block=__auto_blocks[4197197], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197191], to_block=__auto_blocks[4197242], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197197].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197197], to_block=__auto_blocks[4197413], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197242].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197242], to_block=__auto_blocks[4197286], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197242], to_block=__auto_blocks[4197334], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197286].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197286], to_block=__auto_blocks[4197334], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197286], to_block=__auto_blocks[4197292], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197292].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197292], to_block=__auto_blocks[4197413], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197334].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197334], to_block=__auto_blocks[4197413], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197413].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197413], to_block=__auto_blocks[4197417], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197417].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197417], to_block=__auto_blocks[4197115], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197417], to_block=__auto_blocks[4197427], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197427].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197427], to_block=__auto_blocks[4197431], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197431].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197431], to_block=__auto_blocks[4197103], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197431], to_block=__auto_blocks[4197441], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197441].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197441], to_block=__auto_blocks[4197552], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197450].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197450], to_block=__auto_blocks[4197542], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197459].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197459], to_block=__auto_blocks[4197542], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197542].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197542], to_block=__auto_blocks[4197548], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197542], to_block=__auto_blocks[4197459], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197548].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197548], to_block=__auto_blocks[4197552], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197552].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197552], to_block=__auto_blocks[4197558], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197552], to_block=__auto_blocks[4197450], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197558].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197558], to_block=__auto_blocks[4197076], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197584].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197584], to_block=__auto_blocks[4197639], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197584], to_block=__auto_blocks[4197670], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197639].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197639], to_block=__auto_blocks[4197648], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197648].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197648], to_block=__auto_blocks[4197648], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197648], to_block=__auto_blocks[4197670], edge_type=EdgeType.NORMAL),
    ])
    
    __auto_blocks[4197670].edges_out = set([
        
    ])
    
    __auto_blocks[4197696].edges_out = set([
        
    ])
    
    __auto_blocks[4197704].edges_out = set([
        
    ])
    
    
    # Set the edges_in on the blocks
    for b in __auto_blocks.values():
        for e in b.edges_out:
            e.to_block.edges_in.add(CFGEdge(b, e.to_block, e.edge_type))
    
    # Adding basic blocks to their associated functions
    __auto_functions[4195664].blocks = [
        __auto_blocks[4195664],
        __auto_blocks[4195684],
        __auto_blocks[4195686],
    ]
    
    __auto_functions[4195696].blocks = [
        __auto_blocks[4195696],
    ]
    
    __auto_functions[4195712].blocks = [
        __auto_blocks[4195712],
    ]
    
    __auto_functions[4195728].blocks = [
        __auto_blocks[4195728],
    ]
    
    __auto_functions[4195744].blocks = [
        __auto_blocks[4195744],
    ]
    
    __auto_functions[4195760].blocks = [
        __auto_blocks[4195760],
    ]
    
    __auto_functions[4195776].blocks = [
        __auto_blocks[4195776],
    ]
    
    __auto_functions[4195792].blocks = [
        __auto_blocks[4195792],
    ]
    
    __auto_functions[4195840].blocks = [
        __auto_blocks[4195840],
    ]
    
    __auto_functions[4195856].blocks = [
        __auto_blocks[4195856],
        __auto_blocks[4195875],
        __auto_blocks[4195887],
        __auto_blocks[4195896],
    ]
    
    __auto_functions[4195904].blocks = [
        __auto_blocks[4195904],
        __auto_blocks[4195940],
        __auto_blocks[4195952],
        __auto_blocks[4195960],
    ]
    
    __auto_functions[4195968].blocks = [
        __auto_blocks[4195968],
        __auto_blocks[4195981],
        __auto_blocks[4196000],
    ]
    
    __auto_functions[4196016].blocks = [
        __auto_blocks[4196016],
    ]
    
    __auto_functions[4196022].blocks = [
        __auto_blocks[4196022],
        __auto_blocks[4196041],
        __auto_blocks[4196062],
        __auto_blocks[4196164],
        __auto_blocks[4196192],
        __auto_blocks[4196204],
        __auto_blocks[4196207],
    ]
    
    __auto_functions[4196209].blocks = [
        __auto_blocks[4196209],
        __auto_blocks[4196240],
        __auto_blocks[4196254],
        __auto_blocks[4196260],
        __auto_blocks[4196279],
        __auto_blocks[4196298],
        __auto_blocks[4196340],
        __auto_blocks[4196347],
        __auto_blocks[4196352],
        __auto_blocks[4196363],
        __auto_blocks[4196369],
        __auto_blocks[4196393],
        __auto_blocks[4196399],
        __auto_blocks[4196408],
        __auto_blocks[4196422],
        __auto_blocks[4196428],
    ]
    
    __auto_functions[4196441].blocks = [
        __auto_blocks[4196441],
        __auto_blocks[4196468],
        __auto_blocks[4196474],
        __auto_blocks[4196520],
        __auto_blocks[4196524],
        __auto_blocks[4196530],
        __auto_blocks[4196573],
        __auto_blocks[4196577],
        __auto_blocks[4196583],
        __auto_blocks[4196589],
        __auto_blocks[4196635],
        __auto_blocks[4196639],
        __auto_blocks[4196645],
        __auto_blocks[4196690],
        __auto_blocks[4196694],
        __auto_blocks[4196700],
        __auto_blocks[4196745],
        __auto_blocks[4196749],
        __auto_blocks[4196755],
        __auto_blocks[4196761],
        __auto_blocks[4196809],
        __auto_blocks[4196813],
        __auto_blocks[4196819],
        __auto_blocks[4196864],
        __auto_blocks[4196868],
        __auto_blocks[4196874],
        __auto_blocks[4196880],
        __auto_blocks[4196928],
        __auto_blocks[4196932],
    ]
    
    __auto_functions[4196937].blocks = [
        __auto_blocks[4196937],
        __auto_blocks[4196967],
        __auto_blocks[4196976],
        __auto_blocks[4197060],
        __auto_blocks[4197066],
        __auto_blocks[4197070],
        __auto_blocks[4197076],
        __auto_blocks[4197103],
        __auto_blocks[4197115],
        __auto_blocks[4197185],
        __auto_blocks[4197191],
        __auto_blocks[4197197],
        __auto_blocks[4197242],
        __auto_blocks[4197286],
        __auto_blocks[4197292],
        __auto_blocks[4197334],
        __auto_blocks[4197413],
        __auto_blocks[4197417],
        __auto_blocks[4197427],
        __auto_blocks[4197431],
        __auto_blocks[4197441],
        __auto_blocks[4197450],
        __auto_blocks[4197459],
        __auto_blocks[4197542],
        __auto_blocks[4197548],
        __auto_blocks[4197552],
        __auto_blocks[4197558],
    ]
    
    __auto_functions[4197584].blocks = [
        __auto_blocks[4197584],
        __auto_blocks[4197639],
        __auto_blocks[4197648],
        __auto_blocks[4197670],
    ]
    
    __auto_functions[4197696].blocks = [
        __auto_blocks[4197696],
    ]
    
    __auto_functions[4197704].blocks = [
        __auto_blocks[4197704],
    ]
    
    # Adding functions to the cfg
    if build_level in ['cfg']:
        __auto_cfg.add_function(*__auto_functions.values())
    else:
        __auto_cfg.functions = list(__auto_functions.values())
        __auto_cfg.blocks = list(__auto_blocks.values())
    
    expected = {
        'sorted_func_order': [4195664, 4195696, 4195712, 4195728, 4195744, 4195760, 4195776, 4195792, 4195840, 4195856, 4195904, 4195968, 4196016, 4196022, 4196209, 4196441, 4196937, 4197584, 4197696, 4197704],
        'sorted_block_order': [4195664, 4195684, 4195686, 4195696, 4195712, 4195728, 4195744, 4195760, 4195776, 4195792, 4195840, 4195856, 4195875, 4195887, 4195896, 4195904, 4195940, 4195952, 4195960, 4195968, 4195981, 4196000, 4196016, 4196022, 4196041, 4196062, 4196164, 4196192, 4196204, 4196207, 4196209, 4196240, 4196254, 4196260, 4196279, 4196298, 4196340, 4196347, 4196352, 4196363, 4196369, 4196393, 4196399, 4196408, 4196422, 4196428, 4196441, 4196468, 4196474, 4196520, 4196524, 4196530, 4196573, 4196577, 4196583, 4196589, 4196635, 4196639, 4196645, 4196690, 4196694, 4196700, 4196745, 4196749, 4196755, 4196761, 4196809, 4196813, 4196819, 4196864, 4196868, 4196874, 4196880, 4196928, 4196932, 4196937, 4196967, 4196976, 4197060, 4197066, 4197070, 4197076, 4197103, 4197115, 4197185, 4197191, 4197197, 4197242, 4197286, 4197292, 4197334, 4197413, 4197417, 4197427, 4197431, 4197441, 4197450, 4197459, 4197542, 4197548, 4197552, 4197558, 4197584, 4197639, 4197648, 4197670, 4197696, 4197704],
        'architecture': 'x86',
        'num_blocks': {4195664: 3, 4195696: 1, 4195712: 1, 4195728: 1, 4195744: 1, 4195760: 1, 4195776: 1, 4195792: 1, 4195840: 1, 4195856: 4, 4195904: 4, 4195968: 3, 4196016: 1, 4196022: 7, 4196209: 16, 4196441: 29, 4196937: 27, 4197584: 4, 4197696: 1, 4197704: 1},
        'num_asm_lines_per_block': {4195664: 5, 4195684: 1, 4195686: 2, 4195696: 2, 4195712: 1, 4195728: 1, 4195744: 1, 4195760: 1, 4195776: 1, 4195792: 13, 4195840: 2, 4195856: 4, 4195875: 3, 4195887: 1, 4195896: 1, 4195904: 9, 4195940: 3, 4195952: 1, 4195960: 1, 4195968: 3, 4195981: 6, 4196000: 1, 4196016: 2, 4196022: 6, 4196041: 4, 4196062: 24, 4196164: 8, 4196192: 4, 4196204: 1, 4196207: 2, 4196209: 8, 4196240: 3, 4196254: 2, 4196260: 4, 4196279: 4, 4196298: 14, 4196340: 2, 4196347: 1, 4196352: 3, 4196363: 2, 4196369: 5, 4196393: 2, 4196399: 2, 4196408: 3, 4196422: 2, 4196428: 5, 4196441: 8, 4196468: 2, 4196474: 15, 4196520: 1, 4196524: 2, 4196530: 14, 4196573: 1, 4196577: 2, 4196583: 2, 4196589: 15, 4196635: 1, 4196639: 2, 4196645: 15, 4196690: 1, 4196694: 2, 4196700: 15, 4196745: 1, 4196749: 2, 4196755: 2, 4196761: 16, 4196809: 1, 4196813: 2, 4196819: 15, 4196864: 1, 4196868: 2, 4196874: 2, 4196880: 16, 4196928: 1, 4196932: 3, 4196937: 7, 4196967: 2, 4196976: 22, 4197060: 2, 4197066: 1, 4197070: 2, 4197076: 5, 4197103: 2, 4197115: 21, 4197185: 2, 4197191: 2, 4197197: 13, 4197242: 14, 4197286: 2, 4197292: 13, 4197334: 24, 4197413: 1, 4197417: 2, 4197427: 1, 4197431: 2, 4197441: 2, 4197450: 2, 4197459: 25, 4197542: 2, 4197548: 1, 4197552: 2, 4197558: 3, 4197584: 17, 4197639: 2, 4197648: 7, 4197670: 8, 4197696: 2, 4197704: 4},
        'num_asm_lines_per_function': {4195664: 8, 4195696: 2, 4195712: 1, 4195728: 1, 4195744: 1, 4195760: 1, 4195776: 1, 4195792: 13, 4195840: 2, 4195856: 9, 4195904: 14, 4195968: 10, 4196016: 2, 4196022: 49, 4196209: 62, 4196441: 162, 4196937: 177, 4197584: 34, 4197696: 2, 4197704: 4},
        'num_functions': 20,
        'is_root_function': {4195664: True, 4195696: True, 4195712: True, 4195728: True, 4195744: True, 4195760: True, 4195776: True, 4195792: True, 4195840: True, 4195856: True, 4195904: True, 4195968: True, 4196016: True, 4196022: True, 4196209: True, 4196441: True, 4196937: True, 4197584: True, 4197696: True, 4197704: True},
        'is_recursive': {4195664: False, 4195696: False, 4195712: False, 4195728: False, 4195744: False, 4195760: False, 4195776: False, 4195792: False, 4195840: False, 4195856: False, 4195904: False, 4195968: False, 4196016: False, 4196022: False, 4196209: False, 4196441: False, 4196937: False, 4197584: False, 4197696: False, 4197704: False},
        'is_extern_function': {4195664: False, 4195696: False, 4195712: False, 4195728: False, 4195744: False, 4195760: False, 4195776: False, 4195792: False, 4195840: False, 4195856: False, 4195904: False, 4195968: False, 4196016: False, 4196022: False, 4196209: False, 4196441: False, 4196937: False, 4197584: False, 4197696: False, 4197704: False},
        'is_intern_function': {4195664: True, 4195696: True, 4195712: True, 4195728: True, 4195744: True, 4195760: True, 4195776: True, 4195792: True, 4195840: True, 4195856: True, 4195904: True, 4195968: True, 4196016: True, 4196022: True, 4196209: True, 4196441: True, 4196937: True, 4197584: True, 4197696: True, 4197704: True},
        'function_entry_block': {4195664: 4195664, 4195696: 4195696, 4195712: 4195712, 4195728: 4195728, 4195744: 4195744, 4195760: 4195760, 4195776: 4195776, 4195792: 4195792, 4195840: 4195840, 4195856: 4195856, 4195904: 4195904, 4195968: 4195968, 4196016: 4196016, 4196022: 4196022, 4196209: 4196209, 4196441: 4196441, 4196937: 4196937, 4197584: 4197584, 4197696: 4197696, 4197704: 4197704},
        'called_by': {4195664: set(), 4195696: set(), 4195712: set(), 4195728: set(), 4195744: set(), 4195760: set(), 4195776: set(), 4195792: set(), 4195840: set(), 4195856: set(), 4195904: set(), 4195968: set(), 4196016: set(), 4196022: set(), 4196209: set(), 4196441: set(), 4196937: set(), 4197584: set(), 4197696: set(), 4197704: set()},
        'function_hashes': {4195664: 1042174234363728777, 4195696: 164478624686695691, 4195712: 1128999434478625009, 4195728: 2012132754407556443, 4195744: 1597495986429418947, 4195760: 121100365967644996, 4195776: 1783888654792860432, 4195792: 1327568358977775617, 4195840: 819370305390894718, 4195856: 277327555555946912, 4195904: 1195211864552873608, 4195968: 1769120476215767484, 4196016: 1318055053955072212, 4196022: 1727944861203752558, 4196209: 331757516128338215, 4196441: 767702055209789831, 4196937: 2046611245941107777, 4197584: 340075353222524661, 4197696: 266207953397025245, 4197704: 1084049178784031213},
        'block_hashes': {4195664: 1924420227858712302, 4195684: 1237066638332059396, 4195686: 160195148734369595, 4195696: 565439919582290472, 4195712: 1943219943200330366, 4195728: 1422249250415624440, 4195744: 1537972968592897431, 4195760: 39498139336593624, 4195776: 1395061596168447686, 4195792: 276885985570071959, 4195840: 409940995407848839, 4195856: 2108760766958634997, 4195875: 608216890049409995, 4195887: 43141568798015081, 4195896: 1872380550362185252, 4195904: 820823464596078619, 4195940: 1897817249343724836, 4195952: 286354695594695082, 4195960: 1750560136947423104, 4195968: 1216385003418652664, 4195981: 1871915261346200786, 4196000: 829457715620558220, 4196016: 2174968294314579861, 4196022: 322893019453508730, 4196041: 1127267691494286318, 4196062: 884298994013234557, 4196164: 1818273378133144180, 4196192: 1211700141401620541, 4196204: 1834860347468652234, 4196207: 409940430485091570, 4196209: 609972823044183828, 4196240: 1404649228839206834, 4196254: 81473451168457298, 4196260: 1299735275748787444, 4196279: 449403208242610507, 4196298: 629153233815140558, 4196340: 1068677329776642559, 4196347: 94149261936851070, 4196352: 1121590275754211289, 4196363: 2009788244253497800, 4196369: 1954230051462374225, 4196393: 1706670939270264045, 4196399: 257062469445252021, 4196408: 1742041435542251804, 4196422: 1720235821907715750, 4196428: 2125801775545796387, 4196441: 896630650702078555, 4196468: 851875204091540647, 4196474: 2015878782515537938, 4196520: 1734058668384655200, 4196524: 74732909728353758, 4196530: 1267623160603494837, 4196573: 1436549471914733391, 4196577: 152497902413382461, 4196583: 1135939162370939166, 4196589: 806842756041430428, 4196635: 559745681835529082, 4196639: 1837368126203135256, 4196645: 1741541634995441620, 4196690: 622734335054644989, 4196694: 264761284272044808, 4196700: 2030642801175726199, 4196745: 190798773200738263, 4196749: 2279505873830061371, 4196755: 2184944687696277140, 4196761: 2138604644024865765, 4196809: 1201571357477720534, 4196813: 612918595191200891, 4196819: 302402643748559687, 4196864: 720423450180916003, 4196868: 481552622233026365, 4196874: 1041528880562933421, 4196880: 337242346089402821, 4196928: 1759703099208657115, 4196932: 1823853979540809392, 4196937: 669679296055201330, 4196967: 24379491431881033, 4196976: 1605739233354032035, 4197060: 172212367786921842, 4197066: 1357578662721014679, 4197070: 68169130078024731, 4197076: 1386379938303632319, 4197103: 1930088482666803040, 4197115: 1906966785959999855, 4197185: 912363649819651554, 4197191: 2197859999750038563, 4197197: 668093708934076050, 4197242: 832848744066282074, 4197286: 1954741455494912271, 4197292: 513607993871486014, 4197334: 1559444315154414403, 4197413: 2177334210962566368, 4197417: 586240241251122192, 4197427: 1897195794031661351, 4197431: 390686840740515446, 4197441: 1603907141677079847, 4197450: 1237789831780335935, 4197459: 1406834377408808898, 4197542: 1222137413915642431, 4197548: 754143738620535042, 4197552: 1848249430694727956, 4197558: 1910981503235745073, 4197584: 493993764414440216, 4197639: 1245923071456126430, 4197648: 2089222933176759261, 4197670: 1805306155861815671, 4197696: 2137028785917672139, 4197704: 305346685633802255},
        'cfg_hash': 296910516629970935,
        'memcfg_hashes': {'base_norm-op': 2208224361652930758, 'base_norm-inst': 117629441062223330, 'innereye-op': 696851857215543510, 'innereye-inst': 1399126300108886316, 'safe-op': 1619113130869700180, 'safe-inst': 2157565415467959497, 'deepbindiff-op': 364299475857425474, 'deepbindiff-inst': 1979883084805468841, 'deepsemantic-op': 896446949481489775, 'deepsemantic-inst': 946547602774416579, 'compressed_stats-op': 2229989420185948847, 'compressed_stats-inst': 2184636455452077509, 'hpcdata-op': 1128497087045666901, 'hpcdata-inst': 2238987329308968046},
        'metadata': {'some': 'cfg-level', 134: ('metadata', True, None), (1, 2, 3): 'apples'},
        'block_metadatas': {4195664: {}, 4195684: {}, 4195686: {}, 4195696: {}, 4195712: {}, 4195728: {}, 4195744: {}, 4195760: {}, 4195776: {}, 4195792: {}, 4195840: {}, 4195856: {}, 4195875: {}, 4195887: {}, 4195896: {}, 4195904: {}, 4195940: {}, 4195952: {}, 4195960: {}, 4195968: {}, 4195981: {}, 4196000: {}, 4196016: {}, 4196022: {}, 4196041: {}, 4196062: {}, 4196164: {}, 4196192: {}, 4196204: {}, 4196207: {}, 4196209: {}, 4196240: {}, 4196254: {}, 4196260: {}, 4196279: {}, 4196298: {}, 4196340: {}, 4196347: {}, 4196352: {}, 4196363: {}, 4196369: {}, 4196393: {}, 4196399: {}, 4196408: {}, 4196422: {}, 4196428: {}, 4196441: {}, 4196468: {}, 4196474: {}, 4196520: {}, 4196524: {}, 4196530: {}, 4196573: {}, 4196577: {}, 4196583: {}, 4196589: {}, 4196635: {}, 4196639: {}, 4196645: {}, 4196690: {}, 4196694: {}, 4196700: {}, 4196745: {}, 4196749: {}, 4196755: {}, 4196761: {}, 4196809: {}, 4196813: {}, 4196819: {}, 4196864: {}, 4196868: {}, 4196874: {}, 4196880: {}, 4196928: {}, 4196932: {}, 4196937: {}, 4196967: {}, 4196976: {}, 4197060: {}, 4197066: {}, 4197070: {}, 4197076: {}, 4197103: {}, 4197115: {}, 4197185: {}, 4197191: {}, 4197197: {}, 4197242: {}, 4197286: {}, 4197292: {}, 4197334: {}, 4197413: {}, 4197417: {}, 4197427: {}, 4197431: {}, 4197441: {}, 4197450: {}, 4197459: {}, 4197542: {}, 4197548: {}, 4197552: {}, 4197558: {}, 4197584: {}, 4197639: {}, 4197648: {}, 4197670: {}, 4197696: {}, 4197704: {}},
        'function_metadatas': {4195664: {}, 4195696: {}, 4195712: {}, 4195728: {}, 4195744: {}, 4195760: {}, 4195776: {}, 4195792: {}, 4195840: {}, 4195856: {}, 4195904: {}, 4195968: {}, 4196016: {}, 4196022: {}, 4196209: {}, 4196441: {}, 4196937: {}, 4197584: {}, 4197696: {}, 4197704: {}},
        'asm_counts_per_block': {
            4195664: {'0x00400550: endbr64 ': 1, '0x00400554: sub rsp, 8': 1, '0x00400558: mov rax, qword ptr [rip + 0x201a91]': 1, '0x0040055f: test rax, rax': 1, '0x00400562: je 0x400566': 1},
            4195684: {'0x00400564: call rax': 1},
            4195686: {'0x00400566: add rsp, 8': 1, '0x0040056a: ret ': 1},
            4195696: {'0x00400570: push qword ptr [rip + 0x201a92]': 1, '0x00400576: jmp qword ptr [rip + 0x201a94]': 1},
            4195712: {'0x00400580: jmp qword ptr [rip + 0x201a92]': 1},
            4195728: {'0x00400590: jmp qword ptr [rip + 0x201a8a]': 1},
            4195744: {'0x004005a0: jmp qword ptr [rip + 0x201a82]': 1},
            4195760: {'0x004005b0: jmp qword ptr [rip + 0x201a7a]': 1},
            4195776: {'0x004005c0: jmp qword ptr [rip + 0x201a72]': 1},
            4195792: {'0x004005d0: endbr64 ': 1, '0x004005d4: xor ebp, ebp': 1, '0x004005d6: mov r9, rdx': 1, '0x004005d9: pop rsi': 1, '0x004005da: mov rdx, rsp': 1, '0x004005dd: and rsp, 0xfffffffffffffff0': 1, '0x004005e1: push rax': 1, '0x004005e2: push rsp': 1, '0x004005e3: mov r8, 0x400d40': 1, '0x004005ea: mov rcx, 0x400cd0': 1, '0x004005f1: mov rdi, 0x400a49': 1, '0x004005f8: call qword ptr [rip + 0x2019ea]': 1, '0x004005fe: hlt ': 1},
            4195840: {'0x00400600: endbr64 ': 1, '0x00400604: ret ': 1},
            4195856: {'0x00400610: lea rdi, [rip + 0x201a31]': 1, '0x00400617: lea rax, [rip + 0x201a2a]': 1, '0x0040061e: cmp rax, rdi': 1, '0x00400621: je 0x400638': 1},
            4195875: {'0x00400623: mov rax, qword ptr [rip + 0x2019b6]': 1, '0x0040062a: test rax, rax': 1, '0x0040062d: je 0x400638': 1},
            4195887: {'0x0040062f: jmp rax': 1},
            4195896: {'0x00400638: ret ': 1},
            4195904: {'0x00400640: lea rdi, [rip + 0x201a01]': 1, '0x00400647: lea rsi, [rip + 0x2019fa]': 1, '0x0040064e: sub rsi, rdi': 1, '0x00400651: sar rsi, 3': 1, '0x00400655: mov rax, rsi': 1, '0x00400658: shr rax, 0x3f': 1, '0x0040065c: add rsi, rax': 1, '0x0040065f: sar rsi, 1': 1, '0x00400662: je 0x400678': 1},
            4195940: {'0x00400664: mov rax, qword ptr [rip + 0x20198d]': 1, '0x0040066b: test rax, rax': 1, '0x0040066e: je 0x400678': 1},
            4195952: {'0x00400670: jmp rax': 1},
            4195960: {'0x00400678: ret ': 1},
            4195968: {'0x00400680: endbr64 ': 1, '0x00400684: cmp byte ptr [rip + 0x2019b9], 0': 1, '0x0040068b: jne 0x4006a0': 1},
            4195981: {'0x0040068d: push rbp': 1, '0x0040068e: mov rbp, rsp': 1, '0x00400691: call 0x400610': 1, '0x00400696: mov byte ptr [rip + 0x2019a7], 1': 1, '0x0040069d: pop rbp': 1, '0x0040069e: ret ': 1},
            4196000: {'0x004006a0: ret ': 1},
            4196016: {'0x004006b0: endbr64 ': 1, '0x004006b4: jmp 0x400640': 1},
            4196022: {'0x004006b6: push rbp': 1, '0x004006b7: mov rbp, rsp': 1, '0x004006ba: sub rsp, 0x30': 1, '0x004006be: mov qword ptr [rbp - 0x28], rdi': 1, '0x004006c2: cmp qword ptr [rbp - 0x28], 0': 1, '0x004006c7: jns 0x4006de': 1},
            4196041: {'0x004006c9: call 0x400590': 1, '0x004006ce: mov dword ptr [rax], 0x16': 1, '0x004006d4: mov eax, 0xffffffff': 1, '0x004006d9: jmp 0x40076f': 1},
            4196062: {'0x004006de: mov rcx, qword ptr [rbp - 0x28]': 1, '0x004006e2: movabs rdx, 0x20c49ba5e353f7cf': 1, '0x004006ec: mov rax, rcx': 1, '0x004006ef: imul rdx': 1, '0x004006f2: sar rdx, 7': 1, '0x004006f6: mov rax, rcx': 1, '0x004006f9: sar rax, 0x3f': 1, '0x004006fd: sub rdx, rax': 1, '0x00400700: mov rax, rdx': 1, '0x00400703: mov qword ptr [rbp - 0x20], rax': 1, '0x00400707: mov rcx, qword ptr [rbp - 0x28]': 1, '0x0040070b: movabs rdx, 0x20c49ba5e353f7cf': 1, '0x00400715: mov rax, rcx': 1, '0x00400718: imul rdx': 1, '0x0040071b: sar rdx, 7': 1, '0x0040071f: mov rax, rcx': 1, '0x00400722: sar rax, 0x3f': 1, '0x00400726: sub rdx, rax': 1, '0x00400729: mov rax, rdx': 1, '0x0040072c: imul rax, rax, 0x3e8': 1, '0x00400733: sub rcx, rax': 1, '0x00400736: mov rax, rcx': 1, '0x00400739: imul rax, rax, 0xf4240': 1, '0x00400740: mov qword ptr [rbp - 0x18], rax': 1},
            4196164: {'0x00400744: lea rdx, [rbp - 0x20]': 1, '0x00400748: lea rax, [rbp - 0x20]': 1, '0x0040074c: mov rsi, rdx': 1, '0x0040074f: mov rdi, rax': 1, '0x00400752: call 0x4005a0': 1, '0x00400757: mov dword ptr [rbp - 4], eax': 1, '0x0040075a: cmp dword ptr [rbp - 4], 0': 1, '0x0040075e: je 0x40076c': 1},
            4196192: {'0x00400760: call 0x400590': 1, '0x00400765: mov eax, dword ptr [rax]': 1, '0x00400767: cmp eax, 4': 1, '0x0040076a: je 0x400744': 1},
            4196204: {'0x0040076c: mov eax, dword ptr [rbp - 4]': 1},
            4196207: {'0x0040076f: leave ': 1, '0x00400770: ret ': 1},
            4196209: {'0x00400771: push rbp': 1, '0x00400772: mov rbp, rsp': 1, '0x00400775: sub rsp, 0x20': 1, '0x00400779: mov qword ptr [rbp - 0x18], rdi': 1, '0x0040077d: mov edi, 0xa': 1, '0x00400782: call 0x400580': 1, '0x00400787: mov dword ptr [rbp - 4], 0': 1, '0x0040078e: jmp 0x40079e': 1},
            4196240: {'0x00400790: mov edi, 0x2d': 1, '0x00400795: call 0x400580': 1, '0x0040079a: add dword ptr [rbp - 4], 1': 1},
            4196254: {'0x0040079e: cmp dword ptr [rbp - 4], 0x1f': 1, '0x004007a2: jle 0x400790': 1},
            4196260: {'0x004007a4: mov edi, 0xa': 1, '0x004007a9: call 0x400580': 1, '0x004007ae: mov dword ptr [rbp - 8], 0': 1, '0x004007b5: jmp 0x400829': 1},
            4196279: {'0x004007b7: mov edi, 0x7c': 1, '0x004007bc: call 0x400580': 1, '0x004007c1: mov dword ptr [rbp - 0xc], 0': 1, '0x004007c8: jmp 0x40080b': 1},
            4196298: {'0x004007ca: mov eax, dword ptr [rbp - 8]': 1, '0x004007cd: movsxd rdx, eax': 1, '0x004007d0: mov rax, rdx': 1, '0x004007d3: shl rax, 4': 1, '0x004007d7: sub rax, rdx': 1, '0x004007da: add rax, rax': 1, '0x004007dd: mov rdx, rax': 1, '0x004007e0: mov rax, qword ptr [rbp - 0x18]': 1, '0x004007e4: add rdx, rax': 1, '0x004007e7: mov eax, dword ptr [rbp - 0xc]': 1, '0x004007ea: cdqe ': 1, '0x004007ec: movzx eax, byte ptr [rdx + rax]': 1, '0x004007f0: test al, al': 1, '0x004007f2: je 0x4007fb': 1},
            4196340: {'0x004007f4: mov eax, 0x58': 1, '0x004007f9: jmp 0x400800': 1},
            4196347: {'0x004007fb: mov eax, 0x20': 1},
            4196352: {'0x00400800: mov edi, eax': 1, '0x00400802: call 0x400580': 1, '0x00400807: add dword ptr [rbp - 0xc], 1': 1},
            4196363: {'0x0040080b: cmp dword ptr [rbp - 0xc], 0x1d': 1, '0x0040080f: jle 0x4007ca': 1},
            4196369: {'0x00400811: mov edi, 0x7c': 1, '0x00400816: call 0x400580': 1, '0x0040081b: mov edi, 0xa': 1, '0x00400820: call 0x400580': 1, '0x00400825: add dword ptr [rbp - 8], 1': 1},
            4196393: {'0x00400829: cmp dword ptr [rbp - 8], 0xe': 1, '0x0040082d: jle 0x4007b7': 1},
            4196399: {'0x0040082f: mov dword ptr [rbp - 0x10], 0': 1, '0x00400836: jmp 0x400846': 1},
            4196408: {'0x00400838: mov edi, 0x2d': 1, '0x0040083d: call 0x400580': 1, '0x00400842: add dword ptr [rbp - 0x10], 1': 1},
            4196422: {'0x00400846: cmp dword ptr [rbp - 0x10], 0x1f': 1, '0x0040084a: jle 0x400838': 1},
            4196428: {'0x0040084c: mov edi, 0xa': 1, '0x00400851: call 0x400580': 1, '0x00400856: nop ': 1, '0x00400857: leave ': 1, '0x00400858: ret ': 1},
            4196441: {'0x00400859: push rbp': 1, '0x0040085a: mov rbp, rsp': 1, '0x0040085d: mov qword ptr [rbp - 0x18], rdi': 1, '0x00400861: mov dword ptr [rbp - 0x1c], esi': 1, '0x00400864: mov dword ptr [rbp - 0x20], edx': 1, '0x00400867: mov dword ptr [rbp - 4], 0': 1, '0x0040086e: cmp dword ptr [rbp - 0x1c], 0': 1, '0x00400872: jle 0x4008ac': 1},
            4196468: {'0x00400874: cmp dword ptr [rbp - 0x20], 0': 1, '0x00400878: jle 0x4008ac': 1},
            4196474: {'0x0040087a: mov eax, dword ptr [rbp - 0x1c]': 1, '0x0040087d: movsxd rdx, eax': 1, '0x00400880: mov rax, rdx': 1, '0x00400883: shl rax, 4': 1, '0x00400887: sub rax, rdx': 1, '0x0040088a: add rax, rax': 1, '0x0040088d: lea rdx, [rax - 0x1e]': 1, '0x00400891: mov rax, qword ptr [rbp - 0x18]': 1, '0x00400895: add rdx, rax': 1, '0x00400898: mov eax, dword ptr [rbp - 0x20]': 1, '0x0040089b: sub eax, 1': 1, '0x0040089e: cdqe ': 1, '0x004008a0: movzx eax, byte ptr [rdx + rax]': 1, '0x004008a4: test al, al': 1, '0x004008a6: je 0x4008ac': 1},
            4196520: {'0x004008a8: add dword ptr [rbp - 4], 1': 1},
            4196524: {'0x004008ac: cmp dword ptr [rbp - 0x1c], 0': 1, '0x004008b0: jle 0x4008e1': 1},
            4196530: {'0x004008b2: mov eax, dword ptr [rbp - 0x1c]': 1, '0x004008b5: movsxd rdx, eax': 1, '0x004008b8: mov rax, rdx': 1, '0x004008bb: shl rax, 4': 1, '0x004008bf: sub rax, rdx': 1, '0x004008c2: add rax, rax': 1, '0x004008c5: lea rdx, [rax - 0x1e]': 1, '0x004008c9: mov rax, qword ptr [rbp - 0x18]': 1, '0x004008cd: add rdx, rax': 1, '0x004008d0: mov eax, dword ptr [rbp - 0x20]': 1, '0x004008d3: cdqe ': 1, '0x004008d5: movzx eax, byte ptr [rdx + rax]': 1, '0x004008d9: test al, al': 1, '0x004008db: je 0x4008e1': 1},
            4196573: {'0x004008dd: add dword ptr [rbp - 4], 1': 1},
            4196577: {'0x004008e1: cmp dword ptr [rbp - 0x1c], 0': 1, '0x004008e5: jle 0x40091f': 1},
            4196583: {'0x004008e7: cmp dword ptr [rbp - 0x20], 0x1c': 1, '0x004008eb: jg 0x40091f': 1},
            4196589: {'0x004008ed: mov eax, dword ptr [rbp - 0x1c]': 1, '0x004008f0: movsxd rdx, eax': 1, '0x004008f3: mov rax, rdx': 1, '0x004008f6: shl rax, 4': 1, '0x004008fa: sub rax, rdx': 1, '0x004008fd: add rax, rax': 1, '0x00400900: lea rdx, [rax - 0x1e]': 1, '0x00400904: mov rax, qword ptr [rbp - 0x18]': 1, '0x00400908: add rdx, rax': 1, '0x0040090b: mov eax, dword ptr [rbp - 0x20]': 1, '0x0040090e: add eax, 1': 1, '0x00400911: cdqe ': 1, '0x00400913: movzx eax, byte ptr [rdx + rax]': 1, '0x00400917: test al, al': 1, '0x00400919: je 0x40091f': 1},
            4196635: {'0x0040091b: add dword ptr [rbp - 4], 1': 1},
            4196639: {'0x0040091f: cmp dword ptr [rbp - 0x20], 0': 1, '0x00400923: jle 0x400956': 1},
            4196645: {'0x00400925: mov eax, dword ptr [rbp - 0x1c]': 1, '0x00400928: movsxd rdx, eax': 1, '0x0040092b: mov rax, rdx': 1, '0x0040092e: shl rax, 4': 1, '0x00400932: sub rax, rdx': 1, '0x00400935: add rax, rax': 1, '0x00400938: mov rdx, rax': 1, '0x0040093b: mov rax, qword ptr [rbp - 0x18]': 1, '0x0040093f: add rdx, rax': 1, '0x00400942: mov eax, dword ptr [rbp - 0x20]': 1, '0x00400945: sub eax, 1': 1, '0x00400948: cdqe ': 1, '0x0040094a: movzx eax, byte ptr [rdx + rax]': 1, '0x0040094e: test al, al': 1, '0x00400950: je 0x400956': 1},
            4196690: {'0x00400952: add dword ptr [rbp - 4], 1': 1},
            4196694: {'0x00400956: cmp dword ptr [rbp - 0x20], 0x1c': 1, '0x0040095a: jg 0x40098d': 1},
            4196700: {'0x0040095c: mov eax, dword ptr [rbp - 0x1c]': 1, '0x0040095f: movsxd rdx, eax': 1, '0x00400962: mov rax, rdx': 1, '0x00400965: shl rax, 4': 1, '0x00400969: sub rax, rdx': 1, '0x0040096c: add rax, rax': 1, '0x0040096f: mov rdx, rax': 1, '0x00400972: mov rax, qword ptr [rbp - 0x18]': 1, '0x00400976: add rdx, rax': 1, '0x00400979: mov eax, dword ptr [rbp - 0x20]': 1, '0x0040097c: add eax, 1': 1, '0x0040097f: cdqe ': 1, '0x00400981: movzx eax, byte ptr [rdx + rax]': 1, '0x00400985: test al, al': 1, '0x00400987: je 0x40098d': 1},
            4196745: {'0x00400989: add dword ptr [rbp - 4], 1': 1},
            4196749: {'0x0040098d: cmp dword ptr [rbp - 0x1c], 0xd': 1, '0x00400991: jg 0x4009cd': 1},
            4196755: {'0x00400993: cmp dword ptr [rbp - 0x20], 0': 1, '0x00400997: jle 0x4009cd': 1},
            4196761: {'0x00400999: mov eax, dword ptr [rbp - 0x1c]': 1, '0x0040099c: cdqe ': 1, '0x0040099e: lea rdx, [rax + 1]': 1, '0x004009a2: mov rax, rdx': 1, '0x004009a5: shl rax, 4': 1, '0x004009a9: sub rax, rdx': 1, '0x004009ac: add rax, rax': 1, '0x004009af: mov rdx, rax': 1, '0x004009b2: mov rax, qword ptr [rbp - 0x18]': 1, '0x004009b6: add rdx, rax': 1, '0x004009b9: mov eax, dword ptr [rbp - 0x20]': 1, '0x004009bc: sub eax, 1': 1, '0x004009bf: cdqe ': 1, '0x004009c1: movzx eax, byte ptr [rdx + rax]': 1, '0x004009c5: test al, al': 1, '0x004009c7: je 0x4009cd': 1},
            4196809: {'0x004009c9: add dword ptr [rbp - 4], 1': 1},
            4196813: {'0x004009cd: cmp dword ptr [rbp - 0x1c], 0xd': 1, '0x004009d1: jg 0x400a04': 1},
            4196819: {'0x004009d3: mov eax, dword ptr [rbp - 0x1c]': 1, '0x004009d6: cdqe ': 1, '0x004009d8: lea rdx, [rax + 1]': 1, '0x004009dc: mov rax, rdx': 1, '0x004009df: shl rax, 4': 1, '0x004009e3: sub rax, rdx': 1, '0x004009e6: add rax, rax': 1, '0x004009e9: mov rdx, rax': 1, '0x004009ec: mov rax, qword ptr [rbp - 0x18]': 1, '0x004009f0: add rdx, rax': 1, '0x004009f3: mov eax, dword ptr [rbp - 0x20]': 1, '0x004009f6: cdqe ': 1, '0x004009f8: movzx eax, byte ptr [rdx + rax]': 1, '0x004009fc: test al, al': 1, '0x004009fe: je 0x400a04': 1},
            4196864: {'0x00400a00: add dword ptr [rbp - 4], 1': 1},
            4196868: {'0x00400a04: cmp dword ptr [rbp - 0x1c], 0xd': 1, '0x00400a08: jg 0x400a44': 1},
            4196874: {'0x00400a0a: cmp dword ptr [rbp - 0x20], 0x1c': 1, '0x00400a0e: jg 0x400a44': 1},
            4196880: {'0x00400a10: mov eax, dword ptr [rbp - 0x1c]': 1, '0x00400a13: cdqe ': 1, '0x00400a15: lea rdx, [rax + 1]': 1, '0x00400a19: mov rax, rdx': 1, '0x00400a1c: shl rax, 4': 1, '0x00400a20: sub rax, rdx': 1, '0x00400a23: add rax, rax': 1, '0x00400a26: mov rdx, rax': 1, '0x00400a29: mov rax, qword ptr [rbp - 0x18]': 1, '0x00400a2d: add rdx, rax': 1, '0x00400a30: mov eax, dword ptr [rbp - 0x20]': 1, '0x00400a33: add eax, 1': 1, '0x00400a36: cdqe ': 1, '0x00400a38: movzx eax, byte ptr [rdx + rax]': 1, '0x00400a3c: test al, al': 1, '0x00400a3e: je 0x400a44': 1},
            4196928: {'0x00400a40: add dword ptr [rbp - 4], 1': 1},
            4196932: {'0x00400a44: mov eax, dword ptr [rbp - 4]': 1, '0x00400a47: pop rbp': 1, '0x00400a48: ret ': 1},
            4196937: {'0x00400a49: push rbp': 1, '0x00400a4a: mov rbp, rsp': 1, '0x00400a4d: sub rsp, 0x3b0': 1, '0x00400a54: mov edi, 0x3039': 1, '0x00400a59: call 0x4005b0': 1, '0x00400a5e: mov dword ptr [rbp - 4], 0': 1, '0x00400a65: jmp 0x400ace': 1},
            4196967: {'0x00400a67: mov dword ptr [rbp - 8], 0': 1, '0x00400a6e: jmp 0x400ac4': 1},
            4196976: {'0x00400a70: call 0x4005c0': 1, '0x00400a75: cvtsi2ss xmm0, eax': 1, '0x00400a79: movss xmm1, dword ptr [rip + 0x2e7]': 1, '0x00400a81: divss xmm0, xmm1': 1, '0x00400a85: movaps xmm1, xmm0': 1, '0x00400a88: movss xmm0, dword ptr [rip + 0x2dc]': 1, '0x00400a90: comiss xmm0, xmm1': 1, '0x00400a93: seta al': 1, '0x00400a96: mov esi, eax': 1, '0x00400a98: mov eax, dword ptr [rbp - 8]': 1, '0x00400a9b: movsxd rcx, eax': 1, '0x00400a9e: mov eax, dword ptr [rbp - 4]': 1, '0x00400aa1: movsxd rdx, eax': 1, '0x00400aa4: mov rax, rdx': 1, '0x00400aa7: shl rax, 4': 1, '0x00400aab: sub rax, rdx': 1, '0x00400aae: add rax, rax': 1, '0x00400ab1: add rax, rbp': 1, '0x00400ab4: add rax, rcx': 1, '0x00400ab7: sub rax, 0x1e0': 1, '0x00400abd: mov byte ptr [rax], sil': 1, '0x00400ac0: add dword ptr [rbp - 8], 1': 1},
            4197060: {'0x00400ac4: cmp dword ptr [rbp - 8], 0x1d': 1, '0x00400ac8: jle 0x400a70': 1},
            4197066: {'0x00400aca: add dword ptr [rbp - 4], 1': 1},
            4197070: {'0x00400ace: cmp dword ptr [rbp - 4], 0xe': 1, '0x00400ad2: jle 0x400a67': 1},
            4197076: {'0x00400ad4: lea rax, [rbp - 0x1e0]': 1, '0x00400adb: mov rdi, rax': 1, '0x00400ade: call 0x400771': 1, '0x00400ae3: mov dword ptr [rbp - 0xc], 0': 1, '0x00400aea: jmp 0x400c37': 1},
            4197103: {'0x00400aef: mov dword ptr [rbp - 0x10], 0': 1, '0x00400af6: jmp 0x400c29': 1},
            4197115: {'0x00400afb: mov edx, dword ptr [rbp - 0x10]': 1, '0x00400afe: mov ecx, dword ptr [rbp - 0xc]': 1, '0x00400b01: lea rax, [rbp - 0x1e0]': 1, '0x00400b08: mov esi, ecx': 1, '0x00400b0a: mov rdi, rax': 1, '0x00400b0d: call 0x400859': 1, '0x00400b12: mov dword ptr [rbp - 0x1c], eax': 1, '0x00400b15: mov eax, dword ptr [rbp - 0x10]': 1, '0x00400b18: movsxd rcx, eax': 1, '0x00400b1b: mov eax, dword ptr [rbp - 0xc]': 1, '0x00400b1e: movsxd rdx, eax': 1, '0x00400b21: mov rax, rdx': 1, '0x00400b24: shl rax, 4': 1, '0x00400b28: sub rax, rdx': 1, '0x00400b2b: add rax, rax': 1, '0x00400b2e: add rax, rbp': 1, '0x00400b31: add rax, rcx': 1, '0x00400b34: sub rax, 0x1e0': 1, '0x00400b3a: movzx eax, byte ptr [rax]': 1, '0x00400b3d: test al, al': 1, '0x00400b3f: je 0x400b7a': 1},
            4197185: {'0x00400b41: cmp dword ptr [rbp - 0x1c], 1': 1, '0x00400b45: jle 0x400b4d': 1},
            4197191: {'0x00400b47: cmp dword ptr [rbp - 0x1c], 3': 1, '0x00400b4b: jle 0x400b7a': 1},
            4197197: {'0x00400b4d: mov eax, dword ptr [rbp - 0x10]': 1, '0x00400b50: movsxd rcx, eax': 1, '0x00400b53: mov eax, dword ptr [rbp - 0xc]': 1, '0x00400b56: movsxd rdx, eax': 1, '0x00400b59: mov rax, rdx': 1, '0x00400b5c: shl rax, 4': 1, '0x00400b60: sub rax, rdx': 1, '0x00400b63: add rax, rax': 1, '0x00400b66: add rax, rbp': 1, '0x00400b69: add rax, rcx': 1, '0x00400b6c: sub rax, 0x3b0': 1, '0x00400b72: mov byte ptr [rax], 0': 1, '0x00400b75: jmp 0x400c25': 1},
            4197242: {'0x00400b7a: mov eax, dword ptr [rbp - 0x10]': 1, '0x00400b7d: movsxd rcx, eax': 1, '0x00400b80: mov eax, dword ptr [rbp - 0xc]': 1, '0x00400b83: movsxd rdx, eax': 1, '0x00400b86: mov rax, rdx': 1, '0x00400b89: shl rax, 4': 1, '0x00400b8d: sub rax, rdx': 1, '0x00400b90: add rax, rax': 1, '0x00400b93: add rax, rbp': 1, '0x00400b96: add rax, rcx': 1, '0x00400b99: sub rax, 0x1e0': 1, '0x00400b9f: movzx eax, byte ptr [rax]': 1, '0x00400ba2: test al, al': 1, '0x00400ba4: jne 0x400bd6': 1},
            4197286: {'0x00400ba6: cmp dword ptr [rbp - 0x1c], 3': 1, '0x00400baa: jne 0x400bd6': 1},
            4197292: {'0x00400bac: mov eax, dword ptr [rbp - 0x10]': 1, '0x00400baf: movsxd rcx, eax': 1, '0x00400bb2: mov eax, dword ptr [rbp - 0xc]': 1, '0x00400bb5: movsxd rdx, eax': 1, '0x00400bb8: mov rax, rdx': 1, '0x00400bbb: shl rax, 4': 1, '0x00400bbf: sub rax, rdx': 1, '0x00400bc2: add rax, rax': 1, '0x00400bc5: add rax, rbp': 1, '0x00400bc8: add rax, rcx': 1, '0x00400bcb: sub rax, 0x3b0': 1, '0x00400bd1: mov byte ptr [rax], 1': 1, '0x00400bd4: jmp 0x400c25': 1},
            4197334: {'0x00400bd6: mov eax, dword ptr [rbp - 0x10]': 1, '0x00400bd9: movsxd rcx, eax': 1, '0x00400bdc: mov eax, dword ptr [rbp - 0xc]': 1, '0x00400bdf: movsxd rdx, eax': 1, '0x00400be2: mov rax, rdx': 1, '0x00400be5: shl rax, 4': 1, '0x00400be9: sub rax, rdx': 1, '0x00400bec: add rax, rax': 1, '0x00400bef: add rax, rbp': 1, '0x00400bf2: add rax, rcx': 1, '0x00400bf5: sub rax, 0x1e0': 1, '0x00400bfb: movzx ecx, byte ptr [rax]': 1, '0x00400bfe: mov eax, dword ptr [rbp - 0x10]': 1, '0x00400c01: movsxd rsi, eax': 1, '0x00400c04: mov eax, dword ptr [rbp - 0xc]': 1, '0x00400c07: movsxd rdx, eax': 1, '0x00400c0a: mov rax, rdx': 1, '0x00400c0d: shl rax, 4': 1, '0x00400c11: sub rax, rdx': 1, '0x00400c14: add rax, rax': 1, '0x00400c17: add rax, rbp': 1, '0x00400c1a: add rax, rsi': 1, '0x00400c1d: sub rax, 0x3b0': 1, '0x00400c23: mov byte ptr [rax], cl': 1},
            4197413: {'0x00400c25: add dword ptr [rbp - 0x10], 1': 1},
            4197417: {'0x00400c29: cmp dword ptr [rbp - 0x10], 0x1d': 1, '0x00400c2d: jle 0x400afb': 1},
            4197427: {'0x00400c33: add dword ptr [rbp - 0xc], 1': 1},
            4197431: {'0x00400c37: cmp dword ptr [rbp - 0xc], 0xe': 1, '0x00400c3b: jle 0x400aef': 1},
            4197441: {'0x00400c41: mov dword ptr [rbp - 0x14], 0': 1, '0x00400c48: jmp 0x400cb0': 1},
            4197450: {'0x00400c4a: mov dword ptr [rbp - 0x18], 0': 1, '0x00400c51: jmp 0x400ca6': 1},
            4197459: {'0x00400c53: mov eax, dword ptr [rbp - 0x18]': 1, '0x00400c56: movsxd rcx, eax': 1, '0x00400c59: mov eax, dword ptr [rbp - 0x14]': 1, '0x00400c5c: movsxd rdx, eax': 1, '0x00400c5f: mov rax, rdx': 1, '0x00400c62: shl rax, 4': 1, '0x00400c66: sub rax, rdx': 1, '0x00400c69: add rax, rax': 1, '0x00400c6c: add rax, rbp': 1, '0x00400c6f: add rax, rcx': 1, '0x00400c72: sub rax, 0x3b0': 1, '0x00400c78: movzx ecx, byte ptr [rax]': 1, '0x00400c7b: mov eax, dword ptr [rbp - 0x18]': 1, '0x00400c7e: movsxd rsi, eax': 1, '0x00400c81: mov eax, dword ptr [rbp - 0x14]': 1, '0x00400c84: movsxd rdx, eax': 1, '0x00400c87: mov rax, rdx': 1, '0x00400c8a: shl rax, 4': 1, '0x00400c8e: sub rax, rdx': 1, '0x00400c91: add rax, rax': 1, '0x00400c94: add rax, rbp': 1, '0x00400c97: add rax, rsi': 1, '0x00400c9a: sub rax, 0x1e0': 1, '0x00400ca0: mov byte ptr [rax], cl': 1, '0x00400ca2: add dword ptr [rbp - 0x18], 1': 1},
            4197542: {'0x00400ca6: cmp dword ptr [rbp - 0x18], 0x1d': 1, '0x00400caa: jle 0x400c53': 1},
            4197548: {'0x00400cac: add dword ptr [rbp - 0x14], 1': 1},
            4197552: {'0x00400cb0: cmp dword ptr [rbp - 0x14], 0xe': 1, '0x00400cb4: jle 0x400c4a': 1},
            4197558: {'0x00400cb6: mov edi, 0x1f4': 1, '0x00400cbb: call 0x4006b6': 1, '0x00400cc0: jmp 0x400ad4': 1},
            4197584: {'0x00400cd0: endbr64 ': 1, '0x00400cd4: push r15': 1, '0x00400cd6: mov r15, rdx': 1, '0x00400cd9: push r14': 1, '0x00400cdb: mov r14, rsi': 1, '0x00400cde: push r13': 1, '0x00400ce0: mov r13d, edi': 1, '0x00400ce3: push r12': 1, '0x00400ce5: lea r12, [rip + 0x201114]': 1, '0x00400cec: push rbp': 1, '0x00400ced: lea rbp, [rip + 0x201114]': 1, '0x00400cf4: push rbx': 1, '0x00400cf5: sub rbp, r12': 1, '0x00400cf8: sub rsp, 8': 1, '0x00400cfc: call 0x400550': 1, '0x00400d01: sar rbp, 3': 1, '0x00400d05: je 0x400d26': 1},
            4197639: {'0x00400d07: xor ebx, ebx': 1, '0x00400d09: nop dword ptr [rax]': 1},
            4197648: {'0x00400d10: mov rdx, r15': 1, '0x00400d13: mov rsi, r14': 1, '0x00400d16: mov edi, r13d': 1, '0x00400d19: call qword ptr [r12 + rbx*8]': 1, '0x00400d1d: add rbx, 1': 1, '0x00400d21: cmp rbp, rbx': 1, '0x00400d24: jne 0x400d10': 1},
            4197670: {'0x00400d26: add rsp, 8': 1, '0x00400d2a: pop rbx': 1, '0x00400d2b: pop rbp': 1, '0x00400d2c: pop r12': 1, '0x00400d2e: pop r13': 1, '0x00400d30: pop r14': 1, '0x00400d32: pop r15': 1, '0x00400d34: ret ': 1},
            4197696: {'0x00400d40: endbr64 ': 1, '0x00400d44: ret ': 1},
            4197704: {'0x00400d48: endbr64 ': 1, '0x00400d4c: sub rsp, 8': 1, '0x00400d50: add rsp, 8': 1, '0x00400d54: ret ': 1},
        },
        'asm_counts_per_function': {
            4195664: {
                '0x00400550: endbr64 ': 1,
                '0x00400554: sub rsp, 8': 1,
                '0x00400558: mov rax, qword ptr [rip + 0x201a91]': 1,
                '0x0040055f: test rax, rax': 1,
                '0x00400562: je 0x400566': 1,
                '0x00400564: call rax': 1,
                '0x00400566: add rsp, 8': 1,
                '0x0040056a: ret ': 1,
            },
            4195696: {
                '0x00400570: push qword ptr [rip + 0x201a92]': 1,
                '0x00400576: jmp qword ptr [rip + 0x201a94]': 1,
            },
            4195712: {
                '0x00400580: jmp qword ptr [rip + 0x201a92]': 1,
            },
            4195728: {
                '0x00400590: jmp qword ptr [rip + 0x201a8a]': 1,
            },
            4195744: {
                '0x004005a0: jmp qword ptr [rip + 0x201a82]': 1,
            },
            4195760: {
                '0x004005b0: jmp qword ptr [rip + 0x201a7a]': 1,
            },
            4195776: {
                '0x004005c0: jmp qword ptr [rip + 0x201a72]': 1,
            },
            4195792: {
                '0x004005d0: endbr64 ': 1,
                '0x004005d4: xor ebp, ebp': 1,
                '0x004005d6: mov r9, rdx': 1,
                '0x004005d9: pop rsi': 1,
                '0x004005da: mov rdx, rsp': 1,
                '0x004005dd: and rsp, 0xfffffffffffffff0': 1,
                '0x004005e1: push rax': 1,
                '0x004005e2: push rsp': 1,
                '0x004005e3: mov r8, 0x400d40': 1,
                '0x004005ea: mov rcx, 0x400cd0': 1,
                '0x004005f1: mov rdi, 0x400a49': 1,
                '0x004005f8: call qword ptr [rip + 0x2019ea]': 1,
                '0x004005fe: hlt ': 1,
            },
            4195840: {
                '0x00400600: endbr64 ': 1,
                '0x00400604: ret ': 1,
            },
            4195856: {
                '0x00400610: lea rdi, [rip + 0x201a31]': 1,
                '0x00400617: lea rax, [rip + 0x201a2a]': 1,
                '0x0040061e: cmp rax, rdi': 1,
                '0x00400621: je 0x400638': 1,
                '0x00400623: mov rax, qword ptr [rip + 0x2019b6]': 1,
                '0x0040062a: test rax, rax': 1,
                '0x0040062d: je 0x400638': 1,
                '0x0040062f: jmp rax': 1,
                '0x00400638: ret ': 1,
            },
            4195904: {
                '0x00400640: lea rdi, [rip + 0x201a01]': 1,
                '0x00400647: lea rsi, [rip + 0x2019fa]': 1,
                '0x0040064e: sub rsi, rdi': 1,
                '0x00400651: sar rsi, 3': 1,
                '0x00400655: mov rax, rsi': 1,
                '0x00400658: shr rax, 0x3f': 1,
                '0x0040065c: add rsi, rax': 1,
                '0x0040065f: sar rsi, 1': 1,
                '0x00400662: je 0x400678': 1,
                '0x00400664: mov rax, qword ptr [rip + 0x20198d]': 1,
                '0x0040066b: test rax, rax': 1,
                '0x0040066e: je 0x400678': 1,
                '0x00400670: jmp rax': 1,
                '0x00400678: ret ': 1,
            },
            4195968: {
                '0x00400680: endbr64 ': 1,
                '0x00400684: cmp byte ptr [rip + 0x2019b9], 0': 1,
                '0x0040068b: jne 0x4006a0': 1,
                '0x0040068d: push rbp': 1,
                '0x0040068e: mov rbp, rsp': 1,
                '0x00400691: call 0x400610': 1,
                '0x00400696: mov byte ptr [rip + 0x2019a7], 1': 1,
                '0x0040069d: pop rbp': 1,
                '0x0040069e: ret ': 1,
                '0x004006a0: ret ': 1,
            },
            4196016: {
                '0x004006b0: endbr64 ': 1,
                '0x004006b4: jmp 0x400640': 1,
            },
            4196022: {
                '0x004006b6: push rbp': 1,
                '0x004006b7: mov rbp, rsp': 1,
                '0x004006ba: sub rsp, 0x30': 1,
                '0x004006be: mov qword ptr [rbp - 0x28], rdi': 1,
                '0x004006c2: cmp qword ptr [rbp - 0x28], 0': 1,
                '0x004006c7: jns 0x4006de': 1,
                '0x004006c9: call 0x400590': 1,
                '0x004006ce: mov dword ptr [rax], 0x16': 1,
                '0x004006d4: mov eax, 0xffffffff': 1,
                '0x004006d9: jmp 0x40076f': 1,
                '0x004006de: mov rcx, qword ptr [rbp - 0x28]': 1,
                '0x004006e2: movabs rdx, 0x20c49ba5e353f7cf': 1,
                '0x004006ec: mov rax, rcx': 1,
                '0x004006ef: imul rdx': 1,
                '0x004006f2: sar rdx, 7': 1,
                '0x004006f6: mov rax, rcx': 1,
                '0x004006f9: sar rax, 0x3f': 1,
                '0x004006fd: sub rdx, rax': 1,
                '0x00400700: mov rax, rdx': 1,
                '0x00400703: mov qword ptr [rbp - 0x20], rax': 1,
                '0x00400707: mov rcx, qword ptr [rbp - 0x28]': 1,
                '0x0040070b: movabs rdx, 0x20c49ba5e353f7cf': 1,
                '0x00400715: mov rax, rcx': 1,
                '0x00400718: imul rdx': 1,
                '0x0040071b: sar rdx, 7': 1,
                '0x0040071f: mov rax, rcx': 1,
                '0x00400722: sar rax, 0x3f': 1,
                '0x00400726: sub rdx, rax': 1,
                '0x00400729: mov rax, rdx': 1,
                '0x0040072c: imul rax, rax, 0x3e8': 1,
                '0x00400733: sub rcx, rax': 1,
                '0x00400736: mov rax, rcx': 1,
                '0x00400739: imul rax, rax, 0xf4240': 1,
                '0x00400740: mov qword ptr [rbp - 0x18], rax': 1,
                '0x00400744: lea rdx, [rbp - 0x20]': 1,
                '0x00400748: lea rax, [rbp - 0x20]': 1,
                '0x0040074c: mov rsi, rdx': 1,
                '0x0040074f: mov rdi, rax': 1,
                '0x00400752: call 0x4005a0': 1,
                '0x00400757: mov dword ptr [rbp - 4], eax': 1,
                '0x0040075a: cmp dword ptr [rbp - 4], 0': 1,
                '0x0040075e: je 0x40076c': 1,
                '0x00400760: call 0x400590': 1,
                '0x00400765: mov eax, dword ptr [rax]': 1,
                '0x00400767: cmp eax, 4': 1,
                '0x0040076a: je 0x400744': 1,
                '0x0040076c: mov eax, dword ptr [rbp - 4]': 1,
                '0x0040076f: leave ': 1,
                '0x00400770: ret ': 1,
            },
            4196209: {
                '0x00400771: push rbp': 1,
                '0x00400772: mov rbp, rsp': 1,
                '0x00400775: sub rsp, 0x20': 1,
                '0x00400779: mov qword ptr [rbp - 0x18], rdi': 1,
                '0x0040077d: mov edi, 0xa': 1,
                '0x00400782: call 0x400580': 1,
                '0x00400787: mov dword ptr [rbp - 4], 0': 1,
                '0x0040078e: jmp 0x40079e': 1,
                '0x00400790: mov edi, 0x2d': 1,
                '0x00400795: call 0x400580': 1,
                '0x0040079a: add dword ptr [rbp - 4], 1': 1,
                '0x0040079e: cmp dword ptr [rbp - 4], 0x1f': 1,
                '0x004007a2: jle 0x400790': 1,
                '0x004007a4: mov edi, 0xa': 1,
                '0x004007a9: call 0x400580': 1,
                '0x004007ae: mov dword ptr [rbp - 8], 0': 1,
                '0x004007b5: jmp 0x400829': 1,
                '0x004007b7: mov edi, 0x7c': 1,
                '0x004007bc: call 0x400580': 1,
                '0x004007c1: mov dword ptr [rbp - 0xc], 0': 1,
                '0x004007c8: jmp 0x40080b': 1,
                '0x004007ca: mov eax, dword ptr [rbp - 8]': 1,
                '0x004007cd: movsxd rdx, eax': 1,
                '0x004007d0: mov rax, rdx': 1,
                '0x004007d3: shl rax, 4': 1,
                '0x004007d7: sub rax, rdx': 1,
                '0x004007da: add rax, rax': 1,
                '0x004007dd: mov rdx, rax': 1,
                '0x004007e0: mov rax, qword ptr [rbp - 0x18]': 1,
                '0x004007e4: add rdx, rax': 1,
                '0x004007e7: mov eax, dword ptr [rbp - 0xc]': 1,
                '0x004007ea: cdqe ': 1,
                '0x004007ec: movzx eax, byte ptr [rdx + rax]': 1,
                '0x004007f0: test al, al': 1,
                '0x004007f2: je 0x4007fb': 1,
                '0x004007f4: mov eax, 0x58': 1,
                '0x004007f9: jmp 0x400800': 1,
                '0x004007fb: mov eax, 0x20': 1,
                '0x00400800: mov edi, eax': 1,
                '0x00400802: call 0x400580': 1,
                '0x00400807: add dword ptr [rbp - 0xc], 1': 1,
                '0x0040080b: cmp dword ptr [rbp - 0xc], 0x1d': 1,
                '0x0040080f: jle 0x4007ca': 1,
                '0x00400811: mov edi, 0x7c': 1,
                '0x00400816: call 0x400580': 1,
                '0x0040081b: mov edi, 0xa': 1,
                '0x00400820: call 0x400580': 1,
                '0x00400825: add dword ptr [rbp - 8], 1': 1,
                '0x00400829: cmp dword ptr [rbp - 8], 0xe': 1,
                '0x0040082d: jle 0x4007b7': 1,
                '0x0040082f: mov dword ptr [rbp - 0x10], 0': 1,
                '0x00400836: jmp 0x400846': 1,
                '0x00400838: mov edi, 0x2d': 1,
                '0x0040083d: call 0x400580': 1,
                '0x00400842: add dword ptr [rbp - 0x10], 1': 1,
                '0x00400846: cmp dword ptr [rbp - 0x10], 0x1f': 1,
                '0x0040084a: jle 0x400838': 1,
                '0x0040084c: mov edi, 0xa': 1,
                '0x00400851: call 0x400580': 1,
                '0x00400856: nop ': 1,
                '0x00400857: leave ': 1,
                '0x00400858: ret ': 1,
            },
            4196441: {
                '0x00400859: push rbp': 1,
                '0x0040085a: mov rbp, rsp': 1,
                '0x0040085d: mov qword ptr [rbp - 0x18], rdi': 1,
                '0x00400861: mov dword ptr [rbp - 0x1c], esi': 1,
                '0x00400864: mov dword ptr [rbp - 0x20], edx': 1,
                '0x00400867: mov dword ptr [rbp - 4], 0': 1,
                '0x0040086e: cmp dword ptr [rbp - 0x1c], 0': 1,
                '0x00400872: jle 0x4008ac': 1,
                '0x00400874: cmp dword ptr [rbp - 0x20], 0': 1,
                '0x00400878: jle 0x4008ac': 1,
                '0x0040087a: mov eax, dword ptr [rbp - 0x1c]': 1,
                '0x0040087d: movsxd rdx, eax': 1,
                '0x00400880: mov rax, rdx': 1,
                '0x00400883: shl rax, 4': 1,
                '0x00400887: sub rax, rdx': 1,
                '0x0040088a: add rax, rax': 1,
                '0x0040088d: lea rdx, [rax - 0x1e]': 1,
                '0x00400891: mov rax, qword ptr [rbp - 0x18]': 1,
                '0x00400895: add rdx, rax': 1,
                '0x00400898: mov eax, dword ptr [rbp - 0x20]': 1,
                '0x0040089b: sub eax, 1': 1,
                '0x0040089e: cdqe ': 1,
                '0x004008a0: movzx eax, byte ptr [rdx + rax]': 1,
                '0x004008a4: test al, al': 1,
                '0x004008a6: je 0x4008ac': 1,
                '0x004008a8: add dword ptr [rbp - 4], 1': 1,
                '0x004008ac: cmp dword ptr [rbp - 0x1c], 0': 1,
                '0x004008b0: jle 0x4008e1': 1,
                '0x004008b2: mov eax, dword ptr [rbp - 0x1c]': 1,
                '0x004008b5: movsxd rdx, eax': 1,
                '0x004008b8: mov rax, rdx': 1,
                '0x004008bb: shl rax, 4': 1,
                '0x004008bf: sub rax, rdx': 1,
                '0x004008c2: add rax, rax': 1,
                '0x004008c5: lea rdx, [rax - 0x1e]': 1,
                '0x004008c9: mov rax, qword ptr [rbp - 0x18]': 1,
                '0x004008cd: add rdx, rax': 1,
                '0x004008d0: mov eax, dword ptr [rbp - 0x20]': 1,
                '0x004008d3: cdqe ': 1,
                '0x004008d5: movzx eax, byte ptr [rdx + rax]': 1,
                '0x004008d9: test al, al': 1,
                '0x004008db: je 0x4008e1': 1,
                '0x004008dd: add dword ptr [rbp - 4], 1': 1,
                '0x004008e1: cmp dword ptr [rbp - 0x1c], 0': 1,
                '0x004008e5: jle 0x40091f': 1,
                '0x004008e7: cmp dword ptr [rbp - 0x20], 0x1c': 1,
                '0x004008eb: jg 0x40091f': 1,
                '0x004008ed: mov eax, dword ptr [rbp - 0x1c]': 1,
                '0x004008f0: movsxd rdx, eax': 1,
                '0x004008f3: mov rax, rdx': 1,
                '0x004008f6: shl rax, 4': 1,
                '0x004008fa: sub rax, rdx': 1,
                '0x004008fd: add rax, rax': 1,
                '0x00400900: lea rdx, [rax - 0x1e]': 1,
                '0x00400904: mov rax, qword ptr [rbp - 0x18]': 1,
                '0x00400908: add rdx, rax': 1,
                '0x0040090b: mov eax, dword ptr [rbp - 0x20]': 1,
                '0x0040090e: add eax, 1': 1,
                '0x00400911: cdqe ': 1,
                '0x00400913: movzx eax, byte ptr [rdx + rax]': 1,
                '0x00400917: test al, al': 1,
                '0x00400919: je 0x40091f': 1,
                '0x0040091b: add dword ptr [rbp - 4], 1': 1,
                '0x0040091f: cmp dword ptr [rbp - 0x20], 0': 1,
                '0x00400923: jle 0x400956': 1,
                '0x00400925: mov eax, dword ptr [rbp - 0x1c]': 1,
                '0x00400928: movsxd rdx, eax': 1,
                '0x0040092b: mov rax, rdx': 1,
                '0x0040092e: shl rax, 4': 1,
                '0x00400932: sub rax, rdx': 1,
                '0x00400935: add rax, rax': 1,
                '0x00400938: mov rdx, rax': 1,
                '0x0040093b: mov rax, qword ptr [rbp - 0x18]': 1,
                '0x0040093f: add rdx, rax': 1,
                '0x00400942: mov eax, dword ptr [rbp - 0x20]': 1,
                '0x00400945: sub eax, 1': 1,
                '0x00400948: cdqe ': 1,
                '0x0040094a: movzx eax, byte ptr [rdx + rax]': 1,
                '0x0040094e: test al, al': 1,
                '0x00400950: je 0x400956': 1,
                '0x00400952: add dword ptr [rbp - 4], 1': 1,
                '0x00400956: cmp dword ptr [rbp - 0x20], 0x1c': 1,
                '0x0040095a: jg 0x40098d': 1,
                '0x0040095c: mov eax, dword ptr [rbp - 0x1c]': 1,
                '0x0040095f: movsxd rdx, eax': 1,
                '0x00400962: mov rax, rdx': 1,
                '0x00400965: shl rax, 4': 1,
                '0x00400969: sub rax, rdx': 1,
                '0x0040096c: add rax, rax': 1,
                '0x0040096f: mov rdx, rax': 1,
                '0x00400972: mov rax, qword ptr [rbp - 0x18]': 1,
                '0x00400976: add rdx, rax': 1,
                '0x00400979: mov eax, dword ptr [rbp - 0x20]': 1,
                '0x0040097c: add eax, 1': 1,
                '0x0040097f: cdqe ': 1,
                '0x00400981: movzx eax, byte ptr [rdx + rax]': 1,
                '0x00400985: test al, al': 1,
                '0x00400987: je 0x40098d': 1,
                '0x00400989: add dword ptr [rbp - 4], 1': 1,
                '0x0040098d: cmp dword ptr [rbp - 0x1c], 0xd': 1,
                '0x00400991: jg 0x4009cd': 1,
                '0x00400993: cmp dword ptr [rbp - 0x20], 0': 1,
                '0x00400997: jle 0x4009cd': 1,
                '0x00400999: mov eax, dword ptr [rbp - 0x1c]': 1,
                '0x0040099c: cdqe ': 1,
                '0x0040099e: lea rdx, [rax + 1]': 1,
                '0x004009a2: mov rax, rdx': 1,
                '0x004009a5: shl rax, 4': 1,
                '0x004009a9: sub rax, rdx': 1,
                '0x004009ac: add rax, rax': 1,
                '0x004009af: mov rdx, rax': 1,
                '0x004009b2: mov rax, qword ptr [rbp - 0x18]': 1,
                '0x004009b6: add rdx, rax': 1,
                '0x004009b9: mov eax, dword ptr [rbp - 0x20]': 1,
                '0x004009bc: sub eax, 1': 1,
                '0x004009bf: cdqe ': 1,
                '0x004009c1: movzx eax, byte ptr [rdx + rax]': 1,
                '0x004009c5: test al, al': 1,
                '0x004009c7: je 0x4009cd': 1,
                '0x004009c9: add dword ptr [rbp - 4], 1': 1,
                '0x004009cd: cmp dword ptr [rbp - 0x1c], 0xd': 1,
                '0x004009d1: jg 0x400a04': 1,
                '0x004009d3: mov eax, dword ptr [rbp - 0x1c]': 1,
                '0x004009d6: cdqe ': 1,
                '0x004009d8: lea rdx, [rax + 1]': 1,
                '0x004009dc: mov rax, rdx': 1,
                '0x004009df: shl rax, 4': 1,
                '0x004009e3: sub rax, rdx': 1,
                '0x004009e6: add rax, rax': 1,
                '0x004009e9: mov rdx, rax': 1,
                '0x004009ec: mov rax, qword ptr [rbp - 0x18]': 1,
                '0x004009f0: add rdx, rax': 1,
                '0x004009f3: mov eax, dword ptr [rbp - 0x20]': 1,
                '0x004009f6: cdqe ': 1,
                '0x004009f8: movzx eax, byte ptr [rdx + rax]': 1,
                '0x004009fc: test al, al': 1,
                '0x004009fe: je 0x400a04': 1,
                '0x00400a00: add dword ptr [rbp - 4], 1': 1,
                '0x00400a04: cmp dword ptr [rbp - 0x1c], 0xd': 1,
                '0x00400a08: jg 0x400a44': 1,
                '0x00400a0a: cmp dword ptr [rbp - 0x20], 0x1c': 1,
                '0x00400a0e: jg 0x400a44': 1,
                '0x00400a10: mov eax, dword ptr [rbp - 0x1c]': 1,
                '0x00400a13: cdqe ': 1,
                '0x00400a15: lea rdx, [rax + 1]': 1,
                '0x00400a19: mov rax, rdx': 1,
                '0x00400a1c: shl rax, 4': 1,
                '0x00400a20: sub rax, rdx': 1,
                '0x00400a23: add rax, rax': 1,
                '0x00400a26: mov rdx, rax': 1,
                '0x00400a29: mov rax, qword ptr [rbp - 0x18]': 1,
                '0x00400a2d: add rdx, rax': 1,
                '0x00400a30: mov eax, dword ptr [rbp - 0x20]': 1,
                '0x00400a33: add eax, 1': 1,
                '0x00400a36: cdqe ': 1,
                '0x00400a38: movzx eax, byte ptr [rdx + rax]': 1,
                '0x00400a3c: test al, al': 1,
                '0x00400a3e: je 0x400a44': 1,
                '0x00400a40: add dword ptr [rbp - 4], 1': 1,
                '0x00400a44: mov eax, dword ptr [rbp - 4]': 1,
                '0x00400a47: pop rbp': 1,
                '0x00400a48: ret ': 1,
            },
            4196937: {
                '0x00400a49: push rbp': 1,
                '0x00400a4a: mov rbp, rsp': 1,
                '0x00400a4d: sub rsp, 0x3b0': 1,
                '0x00400a54: mov edi, 0x3039': 1,
                '0x00400a59: call 0x4005b0': 1,
                '0x00400a5e: mov dword ptr [rbp - 4], 0': 1,
                '0x00400a65: jmp 0x400ace': 1,
                '0x00400a67: mov dword ptr [rbp - 8], 0': 1,
                '0x00400a6e: jmp 0x400ac4': 1,
                '0x00400a70: call 0x4005c0': 1,
                '0x00400a75: cvtsi2ss xmm0, eax': 1,
                '0x00400a79: movss xmm1, dword ptr [rip + 0x2e7]': 1,
                '0x00400a81: divss xmm0, xmm1': 1,
                '0x00400a85: movaps xmm1, xmm0': 1,
                '0x00400a88: movss xmm0, dword ptr [rip + 0x2dc]': 1,
                '0x00400a90: comiss xmm0, xmm1': 1,
                '0x00400a93: seta al': 1,
                '0x00400a96: mov esi, eax': 1,
                '0x00400a98: mov eax, dword ptr [rbp - 8]': 1,
                '0x00400a9b: movsxd rcx, eax': 1,
                '0x00400a9e: mov eax, dword ptr [rbp - 4]': 1,
                '0x00400aa1: movsxd rdx, eax': 1,
                '0x00400aa4: mov rax, rdx': 1,
                '0x00400aa7: shl rax, 4': 1,
                '0x00400aab: sub rax, rdx': 1,
                '0x00400aae: add rax, rax': 1,
                '0x00400ab1: add rax, rbp': 1,
                '0x00400ab4: add rax, rcx': 1,
                '0x00400ab7: sub rax, 0x1e0': 1,
                '0x00400abd: mov byte ptr [rax], sil': 1,
                '0x00400ac0: add dword ptr [rbp - 8], 1': 1,
                '0x00400ac4: cmp dword ptr [rbp - 8], 0x1d': 1,
                '0x00400ac8: jle 0x400a70': 1,
                '0x00400aca: add dword ptr [rbp - 4], 1': 1,
                '0x00400ace: cmp dword ptr [rbp - 4], 0xe': 1,
                '0x00400ad2: jle 0x400a67': 1,
                '0x00400ad4: lea rax, [rbp - 0x1e0]': 1,
                '0x00400adb: mov rdi, rax': 1,
                '0x00400ade: call 0x400771': 1,
                '0x00400ae3: mov dword ptr [rbp - 0xc], 0': 1,
                '0x00400aea: jmp 0x400c37': 1,
                '0x00400aef: mov dword ptr [rbp - 0x10], 0': 1,
                '0x00400af6: jmp 0x400c29': 1,
                '0x00400afb: mov edx, dword ptr [rbp - 0x10]': 1,
                '0x00400afe: mov ecx, dword ptr [rbp - 0xc]': 1,
                '0x00400b01: lea rax, [rbp - 0x1e0]': 1,
                '0x00400b08: mov esi, ecx': 1,
                '0x00400b0a: mov rdi, rax': 1,
                '0x00400b0d: call 0x400859': 1,
                '0x00400b12: mov dword ptr [rbp - 0x1c], eax': 1,
                '0x00400b15: mov eax, dword ptr [rbp - 0x10]': 1,
                '0x00400b18: movsxd rcx, eax': 1,
                '0x00400b1b: mov eax, dword ptr [rbp - 0xc]': 1,
                '0x00400b1e: movsxd rdx, eax': 1,
                '0x00400b21: mov rax, rdx': 1,
                '0x00400b24: shl rax, 4': 1,
                '0x00400b28: sub rax, rdx': 1,
                '0x00400b2b: add rax, rax': 1,
                '0x00400b2e: add rax, rbp': 1,
                '0x00400b31: add rax, rcx': 1,
                '0x00400b34: sub rax, 0x1e0': 1,
                '0x00400b3a: movzx eax, byte ptr [rax]': 1,
                '0x00400b3d: test al, al': 1,
                '0x00400b3f: je 0x400b7a': 1,
                '0x00400b41: cmp dword ptr [rbp - 0x1c], 1': 1,
                '0x00400b45: jle 0x400b4d': 1,
                '0x00400b47: cmp dword ptr [rbp - 0x1c], 3': 1,
                '0x00400b4b: jle 0x400b7a': 1,
                '0x00400b4d: mov eax, dword ptr [rbp - 0x10]': 1,
                '0x00400b50: movsxd rcx, eax': 1,
                '0x00400b53: mov eax, dword ptr [rbp - 0xc]': 1,
                '0x00400b56: movsxd rdx, eax': 1,
                '0x00400b59: mov rax, rdx': 1,
                '0x00400b5c: shl rax, 4': 1,
                '0x00400b60: sub rax, rdx': 1,
                '0x00400b63: add rax, rax': 1,
                '0x00400b66: add rax, rbp': 1,
                '0x00400b69: add rax, rcx': 1,
                '0x00400b6c: sub rax, 0x3b0': 1,
                '0x00400b72: mov byte ptr [rax], 0': 1,
                '0x00400b75: jmp 0x400c25': 1,
                '0x00400b7a: mov eax, dword ptr [rbp - 0x10]': 1,
                '0x00400b7d: movsxd rcx, eax': 1,
                '0x00400b80: mov eax, dword ptr [rbp - 0xc]': 1,
                '0x00400b83: movsxd rdx, eax': 1,
                '0x00400b86: mov rax, rdx': 1,
                '0x00400b89: shl rax, 4': 1,
                '0x00400b8d: sub rax, rdx': 1,
                '0x00400b90: add rax, rax': 1,
                '0x00400b93: add rax, rbp': 1,
                '0x00400b96: add rax, rcx': 1,
                '0x00400b99: sub rax, 0x1e0': 1,
                '0x00400b9f: movzx eax, byte ptr [rax]': 1,
                '0x00400ba2: test al, al': 1,
                '0x00400ba4: jne 0x400bd6': 1,
                '0x00400ba6: cmp dword ptr [rbp - 0x1c], 3': 1,
                '0x00400baa: jne 0x400bd6': 1,
                '0x00400bac: mov eax, dword ptr [rbp - 0x10]': 1,
                '0x00400baf: movsxd rcx, eax': 1,
                '0x00400bb2: mov eax, dword ptr [rbp - 0xc]': 1,
                '0x00400bb5: movsxd rdx, eax': 1,
                '0x00400bb8: mov rax, rdx': 1,
                '0x00400bbb: shl rax, 4': 1,
                '0x00400bbf: sub rax, rdx': 1,
                '0x00400bc2: add rax, rax': 1,
                '0x00400bc5: add rax, rbp': 1,
                '0x00400bc8: add rax, rcx': 1,
                '0x00400bcb: sub rax, 0x3b0': 1,
                '0x00400bd1: mov byte ptr [rax], 1': 1,
                '0x00400bd4: jmp 0x400c25': 1,
                '0x00400bd6: mov eax, dword ptr [rbp - 0x10]': 1,
                '0x00400bd9: movsxd rcx, eax': 1,
                '0x00400bdc: mov eax, dword ptr [rbp - 0xc]': 1,
                '0x00400bdf: movsxd rdx, eax': 1,
                '0x00400be2: mov rax, rdx': 1,
                '0x00400be5: shl rax, 4': 1,
                '0x00400be9: sub rax, rdx': 1,
                '0x00400bec: add rax, rax': 1,
                '0x00400bef: add rax, rbp': 1,
                '0x00400bf2: add rax, rcx': 1,
                '0x00400bf5: sub rax, 0x1e0': 1,
                '0x00400bfb: movzx ecx, byte ptr [rax]': 1,
                '0x00400bfe: mov eax, dword ptr [rbp - 0x10]': 1,
                '0x00400c01: movsxd rsi, eax': 1,
                '0x00400c04: mov eax, dword ptr [rbp - 0xc]': 1,
                '0x00400c07: movsxd rdx, eax': 1,
                '0x00400c0a: mov rax, rdx': 1,
                '0x00400c0d: shl rax, 4': 1,
                '0x00400c11: sub rax, rdx': 1,
                '0x00400c14: add rax, rax': 1,
                '0x00400c17: add rax, rbp': 1,
                '0x00400c1a: add rax, rsi': 1,
                '0x00400c1d: sub rax, 0x3b0': 1,
                '0x00400c23: mov byte ptr [rax], cl': 1,
                '0x00400c25: add dword ptr [rbp - 0x10], 1': 1,
                '0x00400c29: cmp dword ptr [rbp - 0x10], 0x1d': 1,
                '0x00400c2d: jle 0x400afb': 1,
                '0x00400c33: add dword ptr [rbp - 0xc], 1': 1,
                '0x00400c37: cmp dword ptr [rbp - 0xc], 0xe': 1,
                '0x00400c3b: jle 0x400aef': 1,
                '0x00400c41: mov dword ptr [rbp - 0x14], 0': 1,
                '0x00400c48: jmp 0x400cb0': 1,
                '0x00400c4a: mov dword ptr [rbp - 0x18], 0': 1,
                '0x00400c51: jmp 0x400ca6': 1,
                '0x00400c53: mov eax, dword ptr [rbp - 0x18]': 1,
                '0x00400c56: movsxd rcx, eax': 1,
                '0x00400c59: mov eax, dword ptr [rbp - 0x14]': 1,
                '0x00400c5c: movsxd rdx, eax': 1,
                '0x00400c5f: mov rax, rdx': 1,
                '0x00400c62: shl rax, 4': 1,
                '0x00400c66: sub rax, rdx': 1,
                '0x00400c69: add rax, rax': 1,
                '0x00400c6c: add rax, rbp': 1,
                '0x00400c6f: add rax, rcx': 1,
                '0x00400c72: sub rax, 0x3b0': 1,
                '0x00400c78: movzx ecx, byte ptr [rax]': 1,
                '0x00400c7b: mov eax, dword ptr [rbp - 0x18]': 1,
                '0x00400c7e: movsxd rsi, eax': 1,
                '0x00400c81: mov eax, dword ptr [rbp - 0x14]': 1,
                '0x00400c84: movsxd rdx, eax': 1,
                '0x00400c87: mov rax, rdx': 1,
                '0x00400c8a: shl rax, 4': 1,
                '0x00400c8e: sub rax, rdx': 1,
                '0x00400c91: add rax, rax': 1,
                '0x00400c94: add rax, rbp': 1,
                '0x00400c97: add rax, rsi': 1,
                '0x00400c9a: sub rax, 0x1e0': 1,
                '0x00400ca0: mov byte ptr [rax], cl': 1,
                '0x00400ca2: add dword ptr [rbp - 0x18], 1': 1,
                '0x00400ca6: cmp dword ptr [rbp - 0x18], 0x1d': 1,
                '0x00400caa: jle 0x400c53': 1,
                '0x00400cac: add dword ptr [rbp - 0x14], 1': 1,
                '0x00400cb0: cmp dword ptr [rbp - 0x14], 0xe': 1,
                '0x00400cb4: jle 0x400c4a': 1,
                '0x00400cb6: mov edi, 0x1f4': 1,
                '0x00400cbb: call 0x4006b6': 1,
                '0x00400cc0: jmp 0x400ad4': 1,
            },
            4197584: {
                '0x00400cd0: endbr64 ': 1,
                '0x00400cd4: push r15': 1,
                '0x00400cd6: mov r15, rdx': 1,
                '0x00400cd9: push r14': 1,
                '0x00400cdb: mov r14, rsi': 1,
                '0x00400cde: push r13': 1,
                '0x00400ce0: mov r13d, edi': 1,
                '0x00400ce3: push r12': 1,
                '0x00400ce5: lea r12, [rip + 0x201114]': 1,
                '0x00400cec: push rbp': 1,
                '0x00400ced: lea rbp, [rip + 0x201114]': 1,
                '0x00400cf4: push rbx': 1,
                '0x00400cf5: sub rbp, r12': 1,
                '0x00400cf8: sub rsp, 8': 1,
                '0x00400cfc: call 0x400550': 1,
                '0x00400d01: sar rbp, 3': 1,
                '0x00400d05: je 0x400d26': 1,
                '0x00400d07: xor ebx, ebx': 1,
                '0x00400d09: nop dword ptr [rax]': 1,
                '0x00400d10: mov rdx, r15': 1,
                '0x00400d13: mov rsi, r14': 1,
                '0x00400d16: mov edi, r13d': 1,
                '0x00400d19: call qword ptr [r12 + rbx*8]': 1,
                '0x00400d1d: add rbx, 1': 1,
                '0x00400d21: cmp rbp, rbx': 1,
                '0x00400d24: jne 0x400d10': 1,
                '0x00400d26: add rsp, 8': 1,
                '0x00400d2a: pop rbx': 1,
                '0x00400d2b: pop rbp': 1,
                '0x00400d2c: pop r12': 1,
                '0x00400d2e: pop r13': 1,
                '0x00400d30: pop r14': 1,
                '0x00400d32: pop r15': 1,
                '0x00400d34: ret ': 1,
            },
            4197696: {
                '0x00400d40: endbr64 ': 1,
                '0x00400d44: ret ': 1,
            },
            4197704: {
                '0x00400d48: endbr64 ': 1,
                '0x00400d4c: sub rsp, 8': 1,
                '0x00400d50: add rsp, 8': 1,
                '0x00400d54: ret ': 1,
            },
        },
        'asm_counts': {
            '0x00400550: endbr64 ': 1,
            '0x00400554: sub rsp, 8': 1,
            '0x00400558: mov rax, qword ptr [rip + 0x201a91]': 1,
            '0x0040055f: test rax, rax': 1,
            '0x00400562: je 0x400566': 1,
            '0x00400564: call rax': 1,
            '0x00400566: add rsp, 8': 1,
            '0x0040056a: ret ': 1,
            '0x00400570: push qword ptr [rip + 0x201a92]': 1,
            '0x00400576: jmp qword ptr [rip + 0x201a94]': 1,
            '0x00400580: jmp qword ptr [rip + 0x201a92]': 1,
            '0x00400590: jmp qword ptr [rip + 0x201a8a]': 1,
            '0x004005a0: jmp qword ptr [rip + 0x201a82]': 1,
            '0x004005b0: jmp qword ptr [rip + 0x201a7a]': 1,
            '0x004005c0: jmp qword ptr [rip + 0x201a72]': 1,
            '0x004005d0: endbr64 ': 1,
            '0x004005d4: xor ebp, ebp': 1,
            '0x004005d6: mov r9, rdx': 1,
            '0x004005d9: pop rsi': 1,
            '0x004005da: mov rdx, rsp': 1,
            '0x004005dd: and rsp, 0xfffffffffffffff0': 1,
            '0x004005e1: push rax': 1,
            '0x004005e2: push rsp': 1,
            '0x004005e3: mov r8, 0x400d40': 1,
            '0x004005ea: mov rcx, 0x400cd0': 1,
            '0x004005f1: mov rdi, 0x400a49': 1,
            '0x004005f8: call qword ptr [rip + 0x2019ea]': 1,
            '0x004005fe: hlt ': 1,
            '0x00400600: endbr64 ': 1,
            '0x00400604: ret ': 1,
            '0x00400610: lea rdi, [rip + 0x201a31]': 1,
            '0x00400617: lea rax, [rip + 0x201a2a]': 1,
            '0x0040061e: cmp rax, rdi': 1,
            '0x00400621: je 0x400638': 1,
            '0x00400623: mov rax, qword ptr [rip + 0x2019b6]': 1,
            '0x0040062a: test rax, rax': 1,
            '0x0040062d: je 0x400638': 1,
            '0x0040062f: jmp rax': 1,
            '0x00400638: ret ': 1,
            '0x00400640: lea rdi, [rip + 0x201a01]': 1,
            '0x00400647: lea rsi, [rip + 0x2019fa]': 1,
            '0x0040064e: sub rsi, rdi': 1,
            '0x00400651: sar rsi, 3': 1,
            '0x00400655: mov rax, rsi': 1,
            '0x00400658: shr rax, 0x3f': 1,
            '0x0040065c: add rsi, rax': 1,
            '0x0040065f: sar rsi, 1': 1,
            '0x00400662: je 0x400678': 1,
            '0x00400664: mov rax, qword ptr [rip + 0x20198d]': 1,
            '0x0040066b: test rax, rax': 1,
            '0x0040066e: je 0x400678': 1,
            '0x00400670: jmp rax': 1,
            '0x00400678: ret ': 1,
            '0x00400680: endbr64 ': 1,
            '0x00400684: cmp byte ptr [rip + 0x2019b9], 0': 1,
            '0x0040068b: jne 0x4006a0': 1,
            '0x0040068d: push rbp': 1,
            '0x0040068e: mov rbp, rsp': 1,
            '0x00400691: call 0x400610': 1,
            '0x00400696: mov byte ptr [rip + 0x2019a7], 1': 1,
            '0x0040069d: pop rbp': 1,
            '0x0040069e: ret ': 1,
            '0x004006a0: ret ': 1,
            '0x004006b0: endbr64 ': 1,
            '0x004006b4: jmp 0x400640': 1,
            '0x004006b6: push rbp': 1,
            '0x004006b7: mov rbp, rsp': 1,
            '0x004006ba: sub rsp, 0x30': 1,
            '0x004006be: mov qword ptr [rbp - 0x28], rdi': 1,
            '0x004006c2: cmp qword ptr [rbp - 0x28], 0': 1,
            '0x004006c7: jns 0x4006de': 1,
            '0x004006c9: call 0x400590': 1,
            '0x004006ce: mov dword ptr [rax], 0x16': 1,
            '0x004006d4: mov eax, 0xffffffff': 1,
            '0x004006d9: jmp 0x40076f': 1,
            '0x004006de: mov rcx, qword ptr [rbp - 0x28]': 1,
            '0x004006e2: movabs rdx, 0x20c49ba5e353f7cf': 1,
            '0x004006ec: mov rax, rcx': 1,
            '0x004006ef: imul rdx': 1,
            '0x004006f2: sar rdx, 7': 1,
            '0x004006f6: mov rax, rcx': 1,
            '0x004006f9: sar rax, 0x3f': 1,
            '0x004006fd: sub rdx, rax': 1,
            '0x00400700: mov rax, rdx': 1,
            '0x00400703: mov qword ptr [rbp - 0x20], rax': 1,
            '0x00400707: mov rcx, qword ptr [rbp - 0x28]': 1,
            '0x0040070b: movabs rdx, 0x20c49ba5e353f7cf': 1,
            '0x00400715: mov rax, rcx': 1,
            '0x00400718: imul rdx': 1,
            '0x0040071b: sar rdx, 7': 1,
            '0x0040071f: mov rax, rcx': 1,
            '0x00400722: sar rax, 0x3f': 1,
            '0x00400726: sub rdx, rax': 1,
            '0x00400729: mov rax, rdx': 1,
            '0x0040072c: imul rax, rax, 0x3e8': 1,
            '0x00400733: sub rcx, rax': 1,
            '0x00400736: mov rax, rcx': 1,
            '0x00400739: imul rax, rax, 0xf4240': 1,
            '0x00400740: mov qword ptr [rbp - 0x18], rax': 1,
            '0x00400744: lea rdx, [rbp - 0x20]': 1,
            '0x00400748: lea rax, [rbp - 0x20]': 1,
            '0x0040074c: mov rsi, rdx': 1,
            '0x0040074f: mov rdi, rax': 1,
            '0x00400752: call 0x4005a0': 1,
            '0x00400757: mov dword ptr [rbp - 4], eax': 1,
            '0x0040075a: cmp dword ptr [rbp - 4], 0': 1,
            '0x0040075e: je 0x40076c': 1,
            '0x00400760: call 0x400590': 1,
            '0x00400765: mov eax, dword ptr [rax]': 1,
            '0x00400767: cmp eax, 4': 1,
            '0x0040076a: je 0x400744': 1,
            '0x0040076c: mov eax, dword ptr [rbp - 4]': 1,
            '0x0040076f: leave ': 1,
            '0x00400770: ret ': 1,
            '0x00400771: push rbp': 1,
            '0x00400772: mov rbp, rsp': 1,
            '0x00400775: sub rsp, 0x20': 1,
            '0x00400779: mov qword ptr [rbp - 0x18], rdi': 1,
            '0x0040077d: mov edi, 0xa': 1,
            '0x00400782: call 0x400580': 1,
            '0x00400787: mov dword ptr [rbp - 4], 0': 1,
            '0x0040078e: jmp 0x40079e': 1,
            '0x00400790: mov edi, 0x2d': 1,
            '0x00400795: call 0x400580': 1,
            '0x0040079a: add dword ptr [rbp - 4], 1': 1,
            '0x0040079e: cmp dword ptr [rbp - 4], 0x1f': 1,
            '0x004007a2: jle 0x400790': 1,
            '0x004007a4: mov edi, 0xa': 1,
            '0x004007a9: call 0x400580': 1,
            '0x004007ae: mov dword ptr [rbp - 8], 0': 1,
            '0x004007b5: jmp 0x400829': 1,
            '0x004007b7: mov edi, 0x7c': 1,
            '0x004007bc: call 0x400580': 1,
            '0x004007c1: mov dword ptr [rbp - 0xc], 0': 1,
            '0x004007c8: jmp 0x40080b': 1,
            '0x004007ca: mov eax, dword ptr [rbp - 8]': 1,
            '0x004007cd: movsxd rdx, eax': 1,
            '0x004007d0: mov rax, rdx': 1,
            '0x004007d3: shl rax, 4': 1,
            '0x004007d7: sub rax, rdx': 1,
            '0x004007da: add rax, rax': 1,
            '0x004007dd: mov rdx, rax': 1,
            '0x004007e0: mov rax, qword ptr [rbp - 0x18]': 1,
            '0x004007e4: add rdx, rax': 1,
            '0x004007e7: mov eax, dword ptr [rbp - 0xc]': 1,
            '0x004007ea: cdqe ': 1,
            '0x004007ec: movzx eax, byte ptr [rdx + rax]': 1,
            '0x004007f0: test al, al': 1,
            '0x004007f2: je 0x4007fb': 1,
            '0x004007f4: mov eax, 0x58': 1,
            '0x004007f9: jmp 0x400800': 1,
            '0x004007fb: mov eax, 0x20': 1,
            '0x00400800: mov edi, eax': 1,
            '0x00400802: call 0x400580': 1,
            '0x00400807: add dword ptr [rbp - 0xc], 1': 1,
            '0x0040080b: cmp dword ptr [rbp - 0xc], 0x1d': 1,
            '0x0040080f: jle 0x4007ca': 1,
            '0x00400811: mov edi, 0x7c': 1,
            '0x00400816: call 0x400580': 1,
            '0x0040081b: mov edi, 0xa': 1,
            '0x00400820: call 0x400580': 1,
            '0x00400825: add dword ptr [rbp - 8], 1': 1,
            '0x00400829: cmp dword ptr [rbp - 8], 0xe': 1,
            '0x0040082d: jle 0x4007b7': 1,
            '0x0040082f: mov dword ptr [rbp - 0x10], 0': 1,
            '0x00400836: jmp 0x400846': 1,
            '0x00400838: mov edi, 0x2d': 1,
            '0x0040083d: call 0x400580': 1,
            '0x00400842: add dword ptr [rbp - 0x10], 1': 1,
            '0x00400846: cmp dword ptr [rbp - 0x10], 0x1f': 1,
            '0x0040084a: jle 0x400838': 1,
            '0x0040084c: mov edi, 0xa': 1,
            '0x00400851: call 0x400580': 1,
            '0x00400856: nop ': 1,
            '0x00400857: leave ': 1,
            '0x00400858: ret ': 1,
            '0x00400859: push rbp': 1,
            '0x0040085a: mov rbp, rsp': 1,
            '0x0040085d: mov qword ptr [rbp - 0x18], rdi': 1,
            '0x00400861: mov dword ptr [rbp - 0x1c], esi': 1,
            '0x00400864: mov dword ptr [rbp - 0x20], edx': 1,
            '0x00400867: mov dword ptr [rbp - 4], 0': 1,
            '0x0040086e: cmp dword ptr [rbp - 0x1c], 0': 1,
            '0x00400872: jle 0x4008ac': 1,
            '0x00400874: cmp dword ptr [rbp - 0x20], 0': 1,
            '0x00400878: jle 0x4008ac': 1,
            '0x0040087a: mov eax, dword ptr [rbp - 0x1c]': 1,
            '0x0040087d: movsxd rdx, eax': 1,
            '0x00400880: mov rax, rdx': 1,
            '0x00400883: shl rax, 4': 1,
            '0x00400887: sub rax, rdx': 1,
            '0x0040088a: add rax, rax': 1,
            '0x0040088d: lea rdx, [rax - 0x1e]': 1,
            '0x00400891: mov rax, qword ptr [rbp - 0x18]': 1,
            '0x00400895: add rdx, rax': 1,
            '0x00400898: mov eax, dword ptr [rbp - 0x20]': 1,
            '0x0040089b: sub eax, 1': 1,
            '0x0040089e: cdqe ': 1,
            '0x004008a0: movzx eax, byte ptr [rdx + rax]': 1,
            '0x004008a4: test al, al': 1,
            '0x004008a6: je 0x4008ac': 1,
            '0x004008a8: add dword ptr [rbp - 4], 1': 1,
            '0x004008ac: cmp dword ptr [rbp - 0x1c], 0': 1,
            '0x004008b0: jle 0x4008e1': 1,
            '0x004008b2: mov eax, dword ptr [rbp - 0x1c]': 1,
            '0x004008b5: movsxd rdx, eax': 1,
            '0x004008b8: mov rax, rdx': 1,
            '0x004008bb: shl rax, 4': 1,
            '0x004008bf: sub rax, rdx': 1,
            '0x004008c2: add rax, rax': 1,
            '0x004008c5: lea rdx, [rax - 0x1e]': 1,
            '0x004008c9: mov rax, qword ptr [rbp - 0x18]': 1,
            '0x004008cd: add rdx, rax': 1,
            '0x004008d0: mov eax, dword ptr [rbp - 0x20]': 1,
            '0x004008d3: cdqe ': 1,
            '0x004008d5: movzx eax, byte ptr [rdx + rax]': 1,
            '0x004008d9: test al, al': 1,
            '0x004008db: je 0x4008e1': 1,
            '0x004008dd: add dword ptr [rbp - 4], 1': 1,
            '0x004008e1: cmp dword ptr [rbp - 0x1c], 0': 1,
            '0x004008e5: jle 0x40091f': 1,
            '0x004008e7: cmp dword ptr [rbp - 0x20], 0x1c': 1,
            '0x004008eb: jg 0x40091f': 1,
            '0x004008ed: mov eax, dword ptr [rbp - 0x1c]': 1,
            '0x004008f0: movsxd rdx, eax': 1,
            '0x004008f3: mov rax, rdx': 1,
            '0x004008f6: shl rax, 4': 1,
            '0x004008fa: sub rax, rdx': 1,
            '0x004008fd: add rax, rax': 1,
            '0x00400900: lea rdx, [rax - 0x1e]': 1,
            '0x00400904: mov rax, qword ptr [rbp - 0x18]': 1,
            '0x00400908: add rdx, rax': 1,
            '0x0040090b: mov eax, dword ptr [rbp - 0x20]': 1,
            '0x0040090e: add eax, 1': 1,
            '0x00400911: cdqe ': 1,
            '0x00400913: movzx eax, byte ptr [rdx + rax]': 1,
            '0x00400917: test al, al': 1,
            '0x00400919: je 0x40091f': 1,
            '0x0040091b: add dword ptr [rbp - 4], 1': 1,
            '0x0040091f: cmp dword ptr [rbp - 0x20], 0': 1,
            '0x00400923: jle 0x400956': 1,
            '0x00400925: mov eax, dword ptr [rbp - 0x1c]': 1,
            '0x00400928: movsxd rdx, eax': 1,
            '0x0040092b: mov rax, rdx': 1,
            '0x0040092e: shl rax, 4': 1,
            '0x00400932: sub rax, rdx': 1,
            '0x00400935: add rax, rax': 1,
            '0x00400938: mov rdx, rax': 1,
            '0x0040093b: mov rax, qword ptr [rbp - 0x18]': 1,
            '0x0040093f: add rdx, rax': 1,
            '0x00400942: mov eax, dword ptr [rbp - 0x20]': 1,
            '0x00400945: sub eax, 1': 1,
            '0x00400948: cdqe ': 1,
            '0x0040094a: movzx eax, byte ptr [rdx + rax]': 1,
            '0x0040094e: test al, al': 1,
            '0x00400950: je 0x400956': 1,
            '0x00400952: add dword ptr [rbp - 4], 1': 1,
            '0x00400956: cmp dword ptr [rbp - 0x20], 0x1c': 1,
            '0x0040095a: jg 0x40098d': 1,
            '0x0040095c: mov eax, dword ptr [rbp - 0x1c]': 1,
            '0x0040095f: movsxd rdx, eax': 1,
            '0x00400962: mov rax, rdx': 1,
            '0x00400965: shl rax, 4': 1,
            '0x00400969: sub rax, rdx': 1,
            '0x0040096c: add rax, rax': 1,
            '0x0040096f: mov rdx, rax': 1,
            '0x00400972: mov rax, qword ptr [rbp - 0x18]': 1,
            '0x00400976: add rdx, rax': 1,
            '0x00400979: mov eax, dword ptr [rbp - 0x20]': 1,
            '0x0040097c: add eax, 1': 1,
            '0x0040097f: cdqe ': 1,
            '0x00400981: movzx eax, byte ptr [rdx + rax]': 1,
            '0x00400985: test al, al': 1,
            '0x00400987: je 0x40098d': 1,
            '0x00400989: add dword ptr [rbp - 4], 1': 1,
            '0x0040098d: cmp dword ptr [rbp - 0x1c], 0xd': 1,
            '0x00400991: jg 0x4009cd': 1,
            '0x00400993: cmp dword ptr [rbp - 0x20], 0': 1,
            '0x00400997: jle 0x4009cd': 1,
            '0x00400999: mov eax, dword ptr [rbp - 0x1c]': 1,
            '0x0040099c: cdqe ': 1,
            '0x0040099e: lea rdx, [rax + 1]': 1,
            '0x004009a2: mov rax, rdx': 1,
            '0x004009a5: shl rax, 4': 1,
            '0x004009a9: sub rax, rdx': 1,
            '0x004009ac: add rax, rax': 1,
            '0x004009af: mov rdx, rax': 1,
            '0x004009b2: mov rax, qword ptr [rbp - 0x18]': 1,
            '0x004009b6: add rdx, rax': 1,
            '0x004009b9: mov eax, dword ptr [rbp - 0x20]': 1,
            '0x004009bc: sub eax, 1': 1,
            '0x004009bf: cdqe ': 1,
            '0x004009c1: movzx eax, byte ptr [rdx + rax]': 1,
            '0x004009c5: test al, al': 1,
            '0x004009c7: je 0x4009cd': 1,
            '0x004009c9: add dword ptr [rbp - 4], 1': 1,
            '0x004009cd: cmp dword ptr [rbp - 0x1c], 0xd': 1,
            '0x004009d1: jg 0x400a04': 1,
            '0x004009d3: mov eax, dword ptr [rbp - 0x1c]': 1,
            '0x004009d6: cdqe ': 1,
            '0x004009d8: lea rdx, [rax + 1]': 1,
            '0x004009dc: mov rax, rdx': 1,
            '0x004009df: shl rax, 4': 1,
            '0x004009e3: sub rax, rdx': 1,
            '0x004009e6: add rax, rax': 1,
            '0x004009e9: mov rdx, rax': 1,
            '0x004009ec: mov rax, qword ptr [rbp - 0x18]': 1,
            '0x004009f0: add rdx, rax': 1,
            '0x004009f3: mov eax, dword ptr [rbp - 0x20]': 1,
            '0x004009f6: cdqe ': 1,
            '0x004009f8: movzx eax, byte ptr [rdx + rax]': 1,
            '0x004009fc: test al, al': 1,
            '0x004009fe: je 0x400a04': 1,
            '0x00400a00: add dword ptr [rbp - 4], 1': 1,
            '0x00400a04: cmp dword ptr [rbp - 0x1c], 0xd': 1,
            '0x00400a08: jg 0x400a44': 1,
            '0x00400a0a: cmp dword ptr [rbp - 0x20], 0x1c': 1,
            '0x00400a0e: jg 0x400a44': 1,
            '0x00400a10: mov eax, dword ptr [rbp - 0x1c]': 1,
            '0x00400a13: cdqe ': 1,
            '0x00400a15: lea rdx, [rax + 1]': 1,
            '0x00400a19: mov rax, rdx': 1,
            '0x00400a1c: shl rax, 4': 1,
            '0x00400a20: sub rax, rdx': 1,
            '0x00400a23: add rax, rax': 1,
            '0x00400a26: mov rdx, rax': 1,
            '0x00400a29: mov rax, qword ptr [rbp - 0x18]': 1,
            '0x00400a2d: add rdx, rax': 1,
            '0x00400a30: mov eax, dword ptr [rbp - 0x20]': 1,
            '0x00400a33: add eax, 1': 1,
            '0x00400a36: cdqe ': 1,
            '0x00400a38: movzx eax, byte ptr [rdx + rax]': 1,
            '0x00400a3c: test al, al': 1,
            '0x00400a3e: je 0x400a44': 1,
            '0x00400a40: add dword ptr [rbp - 4], 1': 1,
            '0x00400a44: mov eax, dword ptr [rbp - 4]': 1,
            '0x00400a47: pop rbp': 1,
            '0x00400a48: ret ': 1,
            '0x00400a49: push rbp': 1,
            '0x00400a4a: mov rbp, rsp': 1,
            '0x00400a4d: sub rsp, 0x3b0': 1,
            '0x00400a54: mov edi, 0x3039': 1,
            '0x00400a59: call 0x4005b0': 1,
            '0x00400a5e: mov dword ptr [rbp - 4], 0': 1,
            '0x00400a65: jmp 0x400ace': 1,
            '0x00400a67: mov dword ptr [rbp - 8], 0': 1,
            '0x00400a6e: jmp 0x400ac4': 1,
            '0x00400a70: call 0x4005c0': 1,
            '0x00400a75: cvtsi2ss xmm0, eax': 1,
            '0x00400a79: movss xmm1, dword ptr [rip + 0x2e7]': 1,
            '0x00400a81: divss xmm0, xmm1': 1,
            '0x00400a85: movaps xmm1, xmm0': 1,
            '0x00400a88: movss xmm0, dword ptr [rip + 0x2dc]': 1,
            '0x00400a90: comiss xmm0, xmm1': 1,
            '0x00400a93: seta al': 1,
            '0x00400a96: mov esi, eax': 1,
            '0x00400a98: mov eax, dword ptr [rbp - 8]': 1,
            '0x00400a9b: movsxd rcx, eax': 1,
            '0x00400a9e: mov eax, dword ptr [rbp - 4]': 1,
            '0x00400aa1: movsxd rdx, eax': 1,
            '0x00400aa4: mov rax, rdx': 1,
            '0x00400aa7: shl rax, 4': 1,
            '0x00400aab: sub rax, rdx': 1,
            '0x00400aae: add rax, rax': 1,
            '0x00400ab1: add rax, rbp': 1,
            '0x00400ab4: add rax, rcx': 1,
            '0x00400ab7: sub rax, 0x1e0': 1,
            '0x00400abd: mov byte ptr [rax], sil': 1,
            '0x00400ac0: add dword ptr [rbp - 8], 1': 1,
            '0x00400ac4: cmp dword ptr [rbp - 8], 0x1d': 1,
            '0x00400ac8: jle 0x400a70': 1,
            '0x00400aca: add dword ptr [rbp - 4], 1': 1,
            '0x00400ace: cmp dword ptr [rbp - 4], 0xe': 1,
            '0x00400ad2: jle 0x400a67': 1,
            '0x00400ad4: lea rax, [rbp - 0x1e0]': 1,
            '0x00400adb: mov rdi, rax': 1,
            '0x00400ade: call 0x400771': 1,
            '0x00400ae3: mov dword ptr [rbp - 0xc], 0': 1,
            '0x00400aea: jmp 0x400c37': 1,
            '0x00400aef: mov dword ptr [rbp - 0x10], 0': 1,
            '0x00400af6: jmp 0x400c29': 1,
            '0x00400afb: mov edx, dword ptr [rbp - 0x10]': 1,
            '0x00400afe: mov ecx, dword ptr [rbp - 0xc]': 1,
            '0x00400b01: lea rax, [rbp - 0x1e0]': 1,
            '0x00400b08: mov esi, ecx': 1,
            '0x00400b0a: mov rdi, rax': 1,
            '0x00400b0d: call 0x400859': 1,
            '0x00400b12: mov dword ptr [rbp - 0x1c], eax': 1,
            '0x00400b15: mov eax, dword ptr [rbp - 0x10]': 1,
            '0x00400b18: movsxd rcx, eax': 1,
            '0x00400b1b: mov eax, dword ptr [rbp - 0xc]': 1,
            '0x00400b1e: movsxd rdx, eax': 1,
            '0x00400b21: mov rax, rdx': 1,
            '0x00400b24: shl rax, 4': 1,
            '0x00400b28: sub rax, rdx': 1,
            '0x00400b2b: add rax, rax': 1,
            '0x00400b2e: add rax, rbp': 1,
            '0x00400b31: add rax, rcx': 1,
            '0x00400b34: sub rax, 0x1e0': 1,
            '0x00400b3a: movzx eax, byte ptr [rax]': 1,
            '0x00400b3d: test al, al': 1,
            '0x00400b3f: je 0x400b7a': 1,
            '0x00400b41: cmp dword ptr [rbp - 0x1c], 1': 1,
            '0x00400b45: jle 0x400b4d': 1,
            '0x00400b47: cmp dword ptr [rbp - 0x1c], 3': 1,
            '0x00400b4b: jle 0x400b7a': 1,
            '0x00400b4d: mov eax, dword ptr [rbp - 0x10]': 1,
            '0x00400b50: movsxd rcx, eax': 1,
            '0x00400b53: mov eax, dword ptr [rbp - 0xc]': 1,
            '0x00400b56: movsxd rdx, eax': 1,
            '0x00400b59: mov rax, rdx': 1,
            '0x00400b5c: shl rax, 4': 1,
            '0x00400b60: sub rax, rdx': 1,
            '0x00400b63: add rax, rax': 1,
            '0x00400b66: add rax, rbp': 1,
            '0x00400b69: add rax, rcx': 1,
            '0x00400b6c: sub rax, 0x3b0': 1,
            '0x00400b72: mov byte ptr [rax], 0': 1,
            '0x00400b75: jmp 0x400c25': 1,
            '0x00400b7a: mov eax, dword ptr [rbp - 0x10]': 1,
            '0x00400b7d: movsxd rcx, eax': 1,
            '0x00400b80: mov eax, dword ptr [rbp - 0xc]': 1,
            '0x00400b83: movsxd rdx, eax': 1,
            '0x00400b86: mov rax, rdx': 1,
            '0x00400b89: shl rax, 4': 1,
            '0x00400b8d: sub rax, rdx': 1,
            '0x00400b90: add rax, rax': 1,
            '0x00400b93: add rax, rbp': 1,
            '0x00400b96: add rax, rcx': 1,
            '0x00400b99: sub rax, 0x1e0': 1,
            '0x00400b9f: movzx eax, byte ptr [rax]': 1,
            '0x00400ba2: test al, al': 1,
            '0x00400ba4: jne 0x400bd6': 1,
            '0x00400ba6: cmp dword ptr [rbp - 0x1c], 3': 1,
            '0x00400baa: jne 0x400bd6': 1,
            '0x00400bac: mov eax, dword ptr [rbp - 0x10]': 1,
            '0x00400baf: movsxd rcx, eax': 1,
            '0x00400bb2: mov eax, dword ptr [rbp - 0xc]': 1,
            '0x00400bb5: movsxd rdx, eax': 1,
            '0x00400bb8: mov rax, rdx': 1,
            '0x00400bbb: shl rax, 4': 1,
            '0x00400bbf: sub rax, rdx': 1,
            '0x00400bc2: add rax, rax': 1,
            '0x00400bc5: add rax, rbp': 1,
            '0x00400bc8: add rax, rcx': 1,
            '0x00400bcb: sub rax, 0x3b0': 1,
            '0x00400bd1: mov byte ptr [rax], 1': 1,
            '0x00400bd4: jmp 0x400c25': 1,
            '0x00400bd6: mov eax, dword ptr [rbp - 0x10]': 1,
            '0x00400bd9: movsxd rcx, eax': 1,
            '0x00400bdc: mov eax, dword ptr [rbp - 0xc]': 1,
            '0x00400bdf: movsxd rdx, eax': 1,
            '0x00400be2: mov rax, rdx': 1,
            '0x00400be5: shl rax, 4': 1,
            '0x00400be9: sub rax, rdx': 1,
            '0x00400bec: add rax, rax': 1,
            '0x00400bef: add rax, rbp': 1,
            '0x00400bf2: add rax, rcx': 1,
            '0x00400bf5: sub rax, 0x1e0': 1,
            '0x00400bfb: movzx ecx, byte ptr [rax]': 1,
            '0x00400bfe: mov eax, dword ptr [rbp - 0x10]': 1,
            '0x00400c01: movsxd rsi, eax': 1,
            '0x00400c04: mov eax, dword ptr [rbp - 0xc]': 1,
            '0x00400c07: movsxd rdx, eax': 1,
            '0x00400c0a: mov rax, rdx': 1,
            '0x00400c0d: shl rax, 4': 1,
            '0x00400c11: sub rax, rdx': 1,
            '0x00400c14: add rax, rax': 1,
            '0x00400c17: add rax, rbp': 1,
            '0x00400c1a: add rax, rsi': 1,
            '0x00400c1d: sub rax, 0x3b0': 1,
            '0x00400c23: mov byte ptr [rax], cl': 1,
            '0x00400c25: add dword ptr [rbp - 0x10], 1': 1,
            '0x00400c29: cmp dword ptr [rbp - 0x10], 0x1d': 1,
            '0x00400c2d: jle 0x400afb': 1,
            '0x00400c33: add dword ptr [rbp - 0xc], 1': 1,
            '0x00400c37: cmp dword ptr [rbp - 0xc], 0xe': 1,
            '0x00400c3b: jle 0x400aef': 1,
            '0x00400c41: mov dword ptr [rbp - 0x14], 0': 1,
            '0x00400c48: jmp 0x400cb0': 1,
            '0x00400c4a: mov dword ptr [rbp - 0x18], 0': 1,
            '0x00400c51: jmp 0x400ca6': 1,
            '0x00400c53: mov eax, dword ptr [rbp - 0x18]': 1,
            '0x00400c56: movsxd rcx, eax': 1,
            '0x00400c59: mov eax, dword ptr [rbp - 0x14]': 1,
            '0x00400c5c: movsxd rdx, eax': 1,
            '0x00400c5f: mov rax, rdx': 1,
            '0x00400c62: shl rax, 4': 1,
            '0x00400c66: sub rax, rdx': 1,
            '0x00400c69: add rax, rax': 1,
            '0x00400c6c: add rax, rbp': 1,
            '0x00400c6f: add rax, rcx': 1,
            '0x00400c72: sub rax, 0x3b0': 1,
            '0x00400c78: movzx ecx, byte ptr [rax]': 1,
            '0x00400c7b: mov eax, dword ptr [rbp - 0x18]': 1,
            '0x00400c7e: movsxd rsi, eax': 1,
            '0x00400c81: mov eax, dword ptr [rbp - 0x14]': 1,
            '0x00400c84: movsxd rdx, eax': 1,
            '0x00400c87: mov rax, rdx': 1,
            '0x00400c8a: shl rax, 4': 1,
            '0x00400c8e: sub rax, rdx': 1,
            '0x00400c91: add rax, rax': 1,
            '0x00400c94: add rax, rbp': 1,
            '0x00400c97: add rax, rsi': 1,
            '0x00400c9a: sub rax, 0x1e0': 1,
            '0x00400ca0: mov byte ptr [rax], cl': 1,
            '0x00400ca2: add dword ptr [rbp - 0x18], 1': 1,
            '0x00400ca6: cmp dword ptr [rbp - 0x18], 0x1d': 1,
            '0x00400caa: jle 0x400c53': 1,
            '0x00400cac: add dword ptr [rbp - 0x14], 1': 1,
            '0x00400cb0: cmp dword ptr [rbp - 0x14], 0xe': 1,
            '0x00400cb4: jle 0x400c4a': 1,
            '0x00400cb6: mov edi, 0x1f4': 1,
            '0x00400cbb: call 0x4006b6': 1,
            '0x00400cc0: jmp 0x400ad4': 1,
            '0x00400cd0: endbr64 ': 1,
            '0x00400cd4: push r15': 1,
            '0x00400cd6: mov r15, rdx': 1,
            '0x00400cd9: push r14': 1,
            '0x00400cdb: mov r14, rsi': 1,
            '0x00400cde: push r13': 1,
            '0x00400ce0: mov r13d, edi': 1,
            '0x00400ce3: push r12': 1,
            '0x00400ce5: lea r12, [rip + 0x201114]': 1,
            '0x00400cec: push rbp': 1,
            '0x00400ced: lea rbp, [rip + 0x201114]': 1,
            '0x00400cf4: push rbx': 1,
            '0x00400cf5: sub rbp, r12': 1,
            '0x00400cf8: sub rsp, 8': 1,
            '0x00400cfc: call 0x400550': 1,
            '0x00400d01: sar rbp, 3': 1,
            '0x00400d05: je 0x400d26': 1,
            '0x00400d07: xor ebx, ebx': 1,
            '0x00400d09: nop dword ptr [rax]': 1,
            '0x00400d10: mov rdx, r15': 1,
            '0x00400d13: mov rsi, r14': 1,
            '0x00400d16: mov edi, r13d': 1,
            '0x00400d19: call qword ptr [r12 + rbx*8]': 1,
            '0x00400d1d: add rbx, 1': 1,
            '0x00400d21: cmp rbp, rbx': 1,
            '0x00400d24: jne 0x400d10': 1,
            '0x00400d26: add rsp, 8': 1,
            '0x00400d2a: pop rbx': 1,
            '0x00400d2b: pop rbp': 1,
            '0x00400d2c: pop r12': 1,
            '0x00400d2e: pop r13': 1,
            '0x00400d30: pop r14': 1,
            '0x00400d32: pop r15': 1,
            '0x00400d34: ret ': 1,
            '0x00400d40: endbr64 ': 1,
            '0x00400d44: ret ': 1,
            '0x00400d48: endbr64 ': 1,
            '0x00400d4c: sub rsp, 8': 1,
            '0x00400d50: add rsp, 8': 1,
            '0x00400d54: ret ': 1,
        },
    }
    
    return {
        'blocks': __auto_blocks,
        'file': os.path.basename(__file__),
        'inputs': _load_smda(),
        'cfg': __auto_cfg,
        'functions': __auto_functions,
        'expected': expected,
    }
    

def _load_smda():
    """Loads the Conways GOL example using angr"""
    smda = get_module('smda.Disassembler', raise_err=True)
    filepath = os.path.join(os.path.dirname(__file__), 'gol.compiled')
    proj = smda.Disassembler().disassembleFile(filepath)
    return [proj]
