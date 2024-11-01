"""Conway's GOL binary analyzed with angr

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

    # Create the cfg object. This cfg has 14 functions, 56 basic blocks, 65 edges, and 219 lines of assembly.
    metadata = {'some': 'cfg-level', 134: ('metadata', True, None), (1, 2, 3): 'apples'}
    __auto_cfg = CFG(metadata=metadata) if build_level in ['cfg'] else FakeCFG(metadata=metadata)

    # Building all functions. Dictionary maps integer address to CFGFunction() object
    __auto_functions = {
        4195664: func_type(parent_cfg=__auto_cfg, address=4195664, name='_init', is_extern_function=False, metadata={}),
        4195696: func_type(parent_cfg=__auto_cfg, address=4195696, name='sub_400570', is_extern_function=False, metadata={}),
        4195708: func_type(parent_cfg=__auto_cfg, address=4195708, name='sub_40057c', is_extern_function=False, metadata={}),
        4195712: func_type(parent_cfg=__auto_cfg, address=4195712, name='putchar', is_extern_function=True, metadata={}),
        4195728: func_type(parent_cfg=__auto_cfg, address=4195728, name='__errno_location', is_extern_function=True, metadata={}),
        4195744: func_type(parent_cfg=__auto_cfg, address=4195744, name='nanosleep', is_extern_function=True, metadata={}),
        4195760: func_type(parent_cfg=__auto_cfg, address=4195760, name='srand', is_extern_function=True, metadata={}),
        4195776: func_type(parent_cfg=__auto_cfg, address=4195776, name='rand', is_extern_function=True, metadata={}),
        4195792: func_type(parent_cfg=__auto_cfg, address=4195792, name='_start', is_extern_function=False, metadata={}),
        4195838: func_type(parent_cfg=__auto_cfg, address=4195838, name='sub_4005fe', is_extern_function=False, metadata={}),
        4195839: func_type(parent_cfg=__auto_cfg, address=4195839, name='.annobin_static_reloc.c_end', is_extern_function=False, metadata={}),
        4195840: func_type(parent_cfg=__auto_cfg, address=4195840, name='_dl_relocate_static_pie', is_extern_function=False, metadata={}),
        4195845: func_type(parent_cfg=__auto_cfg, address=4195845, name='.annobin__dl_relocate_static_pie.end', is_extern_function=False, metadata={}),
        4195856: func_type(parent_cfg=__auto_cfg, address=4195856, name='deregister_tm_clones', is_extern_function=False, metadata={}),
        4195897: func_type(parent_cfg=__auto_cfg, address=4195897, name='sub_400639', is_extern_function=False, metadata={}),
        4195904: func_type(parent_cfg=__auto_cfg, address=4195904, name='register_tm_clones', is_extern_function=False, metadata={}),
        4195961: func_type(parent_cfg=__auto_cfg, address=4195961, name='sub_400679', is_extern_function=False, metadata={}),
        4195968: func_type(parent_cfg=__auto_cfg, address=4195968, name='__do_global_dtors_aux', is_extern_function=False, metadata={}),
        4195999: func_type(parent_cfg=__auto_cfg, address=4195999, name='sub_40069f', is_extern_function=False, metadata={}),
        4196001: func_type(parent_cfg=__auto_cfg, address=4196001, name='sub_4006a1', is_extern_function=False, metadata={}),
        4196016: func_type(parent_cfg=__auto_cfg, address=4196016, name='frame_dummy', is_extern_function=False, metadata={}),
        4196022: func_type(parent_cfg=__auto_cfg, address=4196022, name='msleep', is_extern_function=False, metadata={}),
        4196209: func_type(parent_cfg=__auto_cfg, address=4196209, name='print_board', is_extern_function=False, metadata={}),
        4196441: func_type(parent_cfg=__auto_cfg, address=4196441, name='getNeighbors', is_extern_function=False, metadata={}),
        4196937: func_type(parent_cfg=__auto_cfg, address=4196937, name='main', is_extern_function=False, metadata={}),
        4197573: func_type(parent_cfg=__auto_cfg, address=4197573, name='sub_400cc5', is_extern_function=False, metadata={}),
        4197584: func_type(parent_cfg=__auto_cfg, address=4197584, name='__libc_csu_init', is_extern_function=False, metadata={}),
        4197685: func_type(parent_cfg=__auto_cfg, address=4197685, name='.annobin___libc_csu_fini.start', is_extern_function=False, metadata={}),
        4197696: func_type(parent_cfg=__auto_cfg, address=4197696, name='__libc_csu_fini', is_extern_function=False, metadata={}),
        4197704: func_type(parent_cfg=__auto_cfg, address=4197704, name='_fini', is_extern_function=False, metadata={}),
        7340032: func_type(parent_cfg=__auto_cfg, address=7340032, name='__libc_start_main', is_extern_function=False, metadata={}),
        7340040: func_type(parent_cfg=__auto_cfg, address=7340040, name='putchar', is_extern_function=False, metadata={}),
        7340048: func_type(parent_cfg=__auto_cfg, address=7340048, name='__errno_location', is_extern_function=False, metadata={}),
        7340056: func_type(parent_cfg=__auto_cfg, address=7340056, name='nanosleep', is_extern_function=False, metadata={}),
        7340064: func_type(parent_cfg=__auto_cfg, address=7340064, name='srand', is_extern_function=False, metadata={}),
        7340072: func_type(parent_cfg=__auto_cfg, address=7340072, name='rand', is_extern_function=False, metadata={}),
        8392784: func_type(parent_cfg=__auto_cfg, address=8392784, name='UnresolvableJumpTarget', is_extern_function=False, metadata={}),
        8392792: func_type(parent_cfg=__auto_cfg, address=8392792, name='UnresolvableCallTarget', is_extern_function=False, metadata={}),
    }

    # Building basic blocks. Dictionary maps integer address to CFGBasicBlock() object
    __auto_blocks = {
        4195664: CFGBasicBlock(parent_function=__auto_functions[4195664], address=4195664, asm_memory_addresses=[4195664, 4195668, 4195672, 4195679, 4195682], metadata={}, asm_lines=[
            '0x400550:\tendbr64\t',
            '0x400554:\tsub\trsp, 8',
            '0x400558:\tmov\trax, qword ptr [rip + 0x201a91]',
            '0x40055f:\ttest\trax, rax',
            '0x400562:\tje\t0x400566',
        ]),
        4195684: CFGBasicBlock(parent_function=__auto_functions[4195664], address=4195684, asm_memory_addresses=[4195684], metadata={}, asm_lines=[
            '0x400564:\tcall\trax',
        ]),
        4195686: CFGBasicBlock(parent_function=__auto_functions[4195664], address=4195686, asm_memory_addresses=[4195686, 4195690], metadata={}, asm_lines=[
            '0x400566:\tadd\trsp, 8',
            '0x40056a:\tret\t',
        ]),
        4195696: CFGBasicBlock(parent_function=__auto_functions[4195696], address=4195696, asm_memory_addresses=[4195696, 4195702], metadata={}, asm_lines=[
            '0x400570:\tpush\tqword ptr [rip + 0x201a92]',
            '0x400576:\tjmp\tqword ptr [rip + 0x201a94]',
        ]),
        4195708: CFGBasicBlock(parent_function=__auto_functions[4195708], address=4195708, asm_memory_addresses=[4195708], metadata={}, asm_lines=[
            '0x40057c:\tnop\tdword ptr [rax]',
        ]),
        4195712: CFGBasicBlock(parent_function=__auto_functions[4195712], address=4195712, asm_memory_addresses=[4195712], metadata={}, asm_lines=[
            '0x400580:\tjmp\tqword ptr [rip + 0x201a92]',
        ]),
        4195728: CFGBasicBlock(parent_function=__auto_functions[4195728], address=4195728, asm_memory_addresses=[4195728], metadata={}, asm_lines=[
            '0x400590:\tjmp\tqword ptr [rip + 0x201a8a]',
        ]),
        4195744: CFGBasicBlock(parent_function=__auto_functions[4195744], address=4195744, asm_memory_addresses=[4195744], metadata={}, asm_lines=[
            '0x4005a0:\tjmp\tqword ptr [rip + 0x201a82]',
        ]),
        4195760: CFGBasicBlock(parent_function=__auto_functions[4195760], address=4195760, asm_memory_addresses=[4195760], metadata={}, asm_lines=[
            '0x4005b0:\tjmp\tqword ptr [rip + 0x201a7a]',
        ]),
        4195776: CFGBasicBlock(parent_function=__auto_functions[4195776], address=4195776, asm_memory_addresses=[4195776], metadata={}, asm_lines=[
            '0x4005c0:\tjmp\tqword ptr [rip + 0x201a72]',
        ]),
        4195792: CFGBasicBlock(parent_function=__auto_functions[4195792], address=4195792, asm_memory_addresses=[4195792, 4195796, 4195798, 4195801, 4195802, 4195805, 4195809, 4195810, 4195811, 4195818, 4195825, 4195832], metadata={}, asm_lines=[
            '0x4005d0:\tendbr64\t',
            '0x4005d4:\txor\tebp, ebp',
            '0x4005d6:\tmov\tr9, rdx',
            '0x4005d9:\tpop\trsi',
            '0x4005da:\tmov\trdx, rsp',
            '0x4005dd:\tand\trsp, 0xfffffffffffffff0',
            '0x4005e1:\tpush\trax',
            '0x4005e2:\tpush\trsp',
            '0x4005e3:\tmov\tr8, 0x400d40',
            '0x4005ea:\tmov\trcx, 0x400cd0',
            '0x4005f1:\tmov\trdi, 0x400a49',
            '0x4005f8:\tcall\tqword ptr [rip + 0x2019ea]',
        ]),
        4195838: CFGBasicBlock(parent_function=__auto_functions[4195838], address=4195838, asm_memory_addresses=[4195838], metadata={}, asm_lines=[
            '0x4005fe:\thlt\t',
        ]),
        4195839: CFGBasicBlock(parent_function=__auto_functions[4195839], address=4195839, asm_memory_addresses=[4195839], metadata={}, asm_lines=[
            '0x4005ff:\tnop\t',
        ]),
        4195840: CFGBasicBlock(parent_function=__auto_functions[4195840], address=4195840, asm_memory_addresses=[4195840, 4195844], metadata={}, asm_lines=[
            '0x400600:\tendbr64\t',
            '0x400604:\tret\t',
        ]),
        4195845: CFGBasicBlock(parent_function=__auto_functions[4195845], address=4195845, asm_memory_addresses=[4195845, 4195855], metadata={}, asm_lines=[
            '0x400605:\tnop\tword ptr cs:[rax + rax]',
            '0x40060f:\tnop\t',
        ]),
        4195856: CFGBasicBlock(parent_function=__auto_functions[4195856], address=4195856, asm_memory_addresses=[4195856, 4195863, 4195870, 4195873], metadata={}, asm_lines=[
            '0x400610:\tlea\trdi, [rip + 0x201a31]',
            '0x400617:\tlea\trax, [rip + 0x201a2a]',
            '0x40061e:\tcmp\trax, rdi',
            '0x400621:\tje\t0x400638',
        ]),
        4195875: CFGBasicBlock(parent_function=__auto_functions[4195856], address=4195875, asm_memory_addresses=[4195875, 4195882, 4195885], metadata={}, asm_lines=[
            '0x400623:\tmov\trax, qword ptr [rip + 0x2019b6]',
            '0x40062a:\ttest\trax, rax',
            '0x40062d:\tje\t0x400638',
        ]),
        4195887: CFGBasicBlock(parent_function=__auto_functions[4195856], address=4195887, asm_memory_addresses=[4195887], metadata={}, asm_lines=[
            '0x40062f:\tjmp\trax',
        ]),
        4195889: CFGBasicBlock(parent_function=__auto_functions[4195856], address=4195889, asm_memory_addresses=[4195889], metadata={}, asm_lines=[
            '0x400631:\tnop\tdword ptr [rax]',
        ]),
        4195896: CFGBasicBlock(parent_function=__auto_functions[4195856], address=4195896, asm_memory_addresses=[4195896], metadata={}, asm_lines=[
            '0x400638:\tret\t',
        ]),
        4195897: CFGBasicBlock(parent_function=__auto_functions[4195897], address=4195897, asm_memory_addresses=[4195897], metadata={}, asm_lines=[
            '0x400639:\tnop\tdword ptr [rax]',
        ]),
        4195904: CFGBasicBlock(parent_function=__auto_functions[4195904], address=4195904, asm_memory_addresses=[4195904, 4195911, 4195918, 4195921, 4195925, 4195928, 4195932, 4195935, 4195938], metadata={}, asm_lines=[
            '0x400640:\tlea\trdi, [rip + 0x201a01]',
            '0x400647:\tlea\trsi, [rip + 0x2019fa]',
            '0x40064e:\tsub\trsi, rdi',
            '0x400651:\tsar\trsi, 3',
            '0x400655:\tmov\trax, rsi',
            '0x400658:\tshr\trax, 0x3f',
            '0x40065c:\tadd\trsi, rax',
            '0x40065f:\tsar\trsi, 1',
            '0x400662:\tje\t0x400678',
        ]),
        4195940: CFGBasicBlock(parent_function=__auto_functions[4195904], address=4195940, asm_memory_addresses=[4195940, 4195947, 4195950], metadata={}, asm_lines=[
            '0x400664:\tmov\trax, qword ptr [rip + 0x20198d]',
            '0x40066b:\ttest\trax, rax',
            '0x40066e:\tje\t0x400678',
        ]),
        4195952: CFGBasicBlock(parent_function=__auto_functions[4195904], address=4195952, asm_memory_addresses=[4195952], metadata={}, asm_lines=[
            '0x400670:\tjmp\trax',
        ]),
        4195954: CFGBasicBlock(parent_function=__auto_functions[4195904], address=4195954, asm_memory_addresses=[4195954], metadata={}, asm_lines=[
            '0x400672:\tnop\tword ptr [rax + rax]',
        ]),
        4195960: CFGBasicBlock(parent_function=__auto_functions[4195904], address=4195960, asm_memory_addresses=[4195960], metadata={}, asm_lines=[
            '0x400678:\tret\t',
        ]),
        4195961: CFGBasicBlock(parent_function=__auto_functions[4195961], address=4195961, asm_memory_addresses=[4195961], metadata={}, asm_lines=[
            '0x400679:\tnop\tdword ptr [rax]',
        ]),
        4195968: CFGBasicBlock(parent_function=__auto_functions[4195968], address=4195968, asm_memory_addresses=[4195968, 4195972, 4195979], metadata={}, asm_lines=[
            '0x400680:\tendbr64\t',
            '0x400684:\tcmp\tbyte ptr [rip + 0x2019b9], 0',
            '0x40068b:\tjne\t0x4006a0',
        ]),
        4195981: CFGBasicBlock(parent_function=__auto_functions[4195968], address=4195981, asm_memory_addresses=[4195981, 4195982, 4195985], metadata={}, asm_lines=[
            '0x40068d:\tpush\trbp',
            '0x40068e:\tmov\trbp, rsp',
            '0x400691:\tcall\t0x400610',
        ]),
        4195990: CFGBasicBlock(parent_function=__auto_functions[4195968], address=4195990, asm_memory_addresses=[4195990, 4195997, 4195998], metadata={}, asm_lines=[
            '0x400696:\tmov\tbyte ptr [rip + 0x2019a7], 1',
            '0x40069d:\tpop\trbp',
            '0x40069e:\tret\t',
        ]),
        4195999: CFGBasicBlock(parent_function=__auto_functions[4195999], address=4195999, asm_memory_addresses=[4195999], metadata={}, asm_lines=[
            '0x40069f:\tnop\t',
        ]),
        4196000: CFGBasicBlock(parent_function=__auto_functions[4195968], address=4196000, asm_memory_addresses=[4196000], metadata={}, asm_lines=[
            '0x4006a0:\tret\t',
        ]),
        4196001: CFGBasicBlock(parent_function=__auto_functions[4196001], address=4196001, asm_memory_addresses=[4196001, 4196012], metadata={}, asm_lines=[
            '0x4006a1:\tnop\tword ptr cs:[rax + rax]',
            '0x4006ac:\tnop\tdword ptr [rax]',
        ]),
        4196016: CFGBasicBlock(parent_function=__auto_functions[4196016], address=4196016, asm_memory_addresses=[4196016, 4196020], metadata={}, asm_lines=[
            '0x4006b0:\tendbr64\t',
            '0x4006b4:\tjmp\t0x400640',
        ]),
        4196022: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196022, asm_memory_addresses=[4196022, 4196023, 4196026, 4196030, 4196034, 4196039], metadata={}, asm_lines=[
            '0x4006b6:\tpush\trbp',
            '0x4006b7:\tmov\trbp, rsp',
            '0x4006ba:\tsub\trsp, 0x30',
            '0x4006be:\tmov\tqword ptr [rbp - 0x28], rdi',
            '0x4006c2:\tcmp\tqword ptr [rbp - 0x28], 0',
            '0x4006c7:\tjns\t0x4006de',
        ]),
        4196041: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196041, asm_memory_addresses=[4196041], metadata={}, asm_lines=[
            '0x4006c9:\tcall\t0x400590',
        ]),
        4196046: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196046, asm_memory_addresses=[4196046, 4196052, 4196057], metadata={}, asm_lines=[
            '0x4006ce:\tmov\tdword ptr [rax], 0x16',
            '0x4006d4:\tmov\teax, 0xffffffff',
            '0x4006d9:\tjmp\t0x40076f',
        ]),
        4196062: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196062, asm_memory_addresses=[4196062, 4196066, 4196076, 4196079, 4196082, 4196086, 4196089, 4196093, 4196096, 4196099, 4196103, 4196107, 4196117, 4196120, 4196123, 4196127, 4196130, 4196134, 4196137, 4196140, 4196147, 4196150, 4196153, 4196160, 4196164, 4196168, 4196172, 4196175, 4196178], metadata={}, asm_lines=[
            '0x4006de:\tmov\trcx, qword ptr [rbp - 0x28]',
            '0x4006e2:\tmovabs\trdx, 0x20c49ba5e353f7cf',
            '0x4006ec:\tmov\trax, rcx',
            '0x4006ef:\timul\trdx',
            '0x4006f2:\tsar\trdx, 7',
            '0x4006f6:\tmov\trax, rcx',
            '0x4006f9:\tsar\trax, 0x3f',
            '0x4006fd:\tsub\trdx, rax',
            '0x400700:\tmov\trax, rdx',
            '0x400703:\tmov\tqword ptr [rbp - 0x20], rax',
            '0x400707:\tmov\trcx, qword ptr [rbp - 0x28]',
            '0x40070b:\tmovabs\trdx, 0x20c49ba5e353f7cf',
            '0x400715:\tmov\trax, rcx',
            '0x400718:\timul\trdx',
            '0x40071b:\tsar\trdx, 7',
            '0x40071f:\tmov\trax, rcx',
            '0x400722:\tsar\trax, 0x3f',
            '0x400726:\tsub\trdx, rax',
            '0x400729:\tmov\trax, rdx',
            '0x40072c:\timul\trax, rax, 0x3e8',
            '0x400733:\tsub\trcx, rax',
            '0x400736:\tmov\trax, rcx',
            '0x400739:\timul\trax, rax, 0xf4240',
            '0x400740:\tmov\tqword ptr [rbp - 0x18], rax',
            '0x400744:\tlea\trdx, [rbp - 0x20]',
            '0x400748:\tlea\trax, [rbp - 0x20]',
            '0x40074c:\tmov\trsi, rdx',
            '0x40074f:\tmov\trdi, rax',
            '0x400752:\tcall\t0x4005a0',
        ]),
        4196164: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196164, asm_memory_addresses=[4196164, 4196168, 4196172, 4196175, 4196178], metadata={}, asm_lines=[
            '0x400744:\tlea\trdx, [rbp - 0x20]',
            '0x400748:\tlea\trax, [rbp - 0x20]',
            '0x40074c:\tmov\trsi, rdx',
            '0x40074f:\tmov\trdi, rax',
            '0x400752:\tcall\t0x4005a0',
        ]),
        4196183: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196183, asm_memory_addresses=[4196183, 4196186, 4196190], metadata={}, asm_lines=[
            '0x400757:\tmov\tdword ptr [rbp - 4], eax',
            '0x40075a:\tcmp\tdword ptr [rbp - 4], 0',
            '0x40075e:\tje\t0x40076c',
        ]),
        4196192: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196192, asm_memory_addresses=[4196192], metadata={}, asm_lines=[
            '0x400760:\tcall\t0x400590',
        ]),
        4196197: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196197, asm_memory_addresses=[4196197, 4196199, 4196202], metadata={}, asm_lines=[
            '0x400765:\tmov\teax, dword ptr [rax]',
            '0x400767:\tcmp\teax, 4',
            '0x40076a:\tje\t0x400744',
        ]),
        4196204: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196204, asm_memory_addresses=[4196204, 4196207, 4196208], metadata={}, asm_lines=[
            '0x40076c:\tmov\teax, dword ptr [rbp - 4]',
            '0x40076f:\tleave\t',
            '0x400770:\tret\t',
        ]),
        4196207: CFGBasicBlock(parent_function=__auto_functions[4196022], address=4196207, asm_memory_addresses=[4196207, 4196208], metadata={}, asm_lines=[
            '0x40076f:\tleave\t',
            '0x400770:\tret\t',
        ]),
        4196209: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196209, asm_memory_addresses=[4196209, 4196210, 4196213, 4196217, 4196221, 4196226], metadata={}, asm_lines=[
            '0x400771:\tpush\trbp',
            '0x400772:\tmov\trbp, rsp',
            '0x400775:\tsub\trsp, 0x20',
            '0x400779:\tmov\tqword ptr [rbp - 0x18], rdi',
            '0x40077d:\tmov\tedi, 0xa',
            '0x400782:\tcall\t0x400580',
        ]),
        4196231: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196231, asm_memory_addresses=[4196231, 4196238], metadata={}, asm_lines=[
            '0x400787:\tmov\tdword ptr [rbp - 4], 0',
            '0x40078e:\tjmp\t0x40079e',
        ]),
        4196240: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196240, asm_memory_addresses=[4196240, 4196245], metadata={}, asm_lines=[
            '0x400790:\tmov\tedi, 0x2d',
            '0x400795:\tcall\t0x400580',
        ]),
        4196250: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196250, asm_memory_addresses=[4196250, 4196254, 4196258], metadata={}, asm_lines=[
            '0x40079a:\tadd\tdword ptr [rbp - 4], 1',
            '0x40079e:\tcmp\tdword ptr [rbp - 4], 0x1f',
            '0x4007a2:\tjle\t0x400790',
        ]),
        4196254: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196254, asm_memory_addresses=[4196254, 4196258], metadata={}, asm_lines=[
            '0x40079e:\tcmp\tdword ptr [rbp - 4], 0x1f',
            '0x4007a2:\tjle\t0x400790',
        ]),
        4196260: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196260, asm_memory_addresses=[4196260, 4196265], metadata={}, asm_lines=[
            '0x4007a4:\tmov\tedi, 0xa',
            '0x4007a9:\tcall\t0x400580',
        ]),
        4196270: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196270, asm_memory_addresses=[4196270, 4196277], metadata={}, asm_lines=[
            '0x4007ae:\tmov\tdword ptr [rbp - 8], 0',
            '0x4007b5:\tjmp\t0x400829',
        ]),
        4196279: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196279, asm_memory_addresses=[4196279, 4196284], metadata={}, asm_lines=[
            '0x4007b7:\tmov\tedi, 0x7c',
            '0x4007bc:\tcall\t0x400580',
        ]),
        4196289: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196289, asm_memory_addresses=[4196289, 4196296], metadata={}, asm_lines=[
            '0x4007c1:\tmov\tdword ptr [rbp - 0xc], 0',
            '0x4007c8:\tjmp\t0x40080b',
        ]),
        4196298: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196298, asm_memory_addresses=[4196298, 4196301, 4196304, 4196307, 4196311, 4196314, 4196317, 4196320, 4196324, 4196327, 4196330, 4196332, 4196336, 4196338], metadata={}, asm_lines=[
            '0x4007ca:\tmov\teax, dword ptr [rbp - 8]',
            '0x4007cd:\tmovsxd\trdx, eax',
            '0x4007d0:\tmov\trax, rdx',
            '0x4007d3:\tshl\trax, 4',
            '0x4007d7:\tsub\trax, rdx',
            '0x4007da:\tadd\trax, rax',
            '0x4007dd:\tmov\trdx, rax',
            '0x4007e0:\tmov\trax, qword ptr [rbp - 0x18]',
            '0x4007e4:\tadd\trdx, rax',
            '0x4007e7:\tmov\teax, dword ptr [rbp - 0xc]',
            '0x4007ea:\tcdqe\t',
            '0x4007ec:\tmovzx\teax, byte ptr [rdx + rax]',
            '0x4007f0:\ttest\tal, al',
            '0x4007f2:\tje\t0x4007fb',
        ]),
        4196340: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196340, asm_memory_addresses=[4196340, 4196345], metadata={}, asm_lines=[
            '0x4007f4:\tmov\teax, 0x58',
            '0x4007f9:\tjmp\t0x400800',
        ]),
        4196347: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196347, asm_memory_addresses=[4196347, 4196352, 4196354], metadata={}, asm_lines=[
            '0x4007fb:\tmov\teax, 0x20',
            '0x400800:\tmov\tedi, eax',
            '0x400802:\tcall\t0x400580',
        ]),
        4196352: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196352, asm_memory_addresses=[4196352, 4196354], metadata={}, asm_lines=[
            '0x400800:\tmov\tedi, eax',
            '0x400802:\tcall\t0x400580',
        ]),
        4196359: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196359, asm_memory_addresses=[4196359, 4196363, 4196367], metadata={}, asm_lines=[
            '0x400807:\tadd\tdword ptr [rbp - 0xc], 1',
            '0x40080b:\tcmp\tdword ptr [rbp - 0xc], 0x1d',
            '0x40080f:\tjle\t0x4007ca',
        ]),
        4196363: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196363, asm_memory_addresses=[4196363, 4196367], metadata={}, asm_lines=[
            '0x40080b:\tcmp\tdword ptr [rbp - 0xc], 0x1d',
            '0x40080f:\tjle\t0x4007ca',
        ]),
        4196369: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196369, asm_memory_addresses=[4196369, 4196374], metadata={}, asm_lines=[
            '0x400811:\tmov\tedi, 0x7c',
            '0x400816:\tcall\t0x400580',
        ]),
        4196379: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196379, asm_memory_addresses=[4196379, 4196384], metadata={}, asm_lines=[
            '0x40081b:\tmov\tedi, 0xa',
            '0x400820:\tcall\t0x400580',
        ]),
        4196389: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196389, asm_memory_addresses=[4196389, 4196393, 4196397], metadata={}, asm_lines=[
            '0x400825:\tadd\tdword ptr [rbp - 8], 1',
            '0x400829:\tcmp\tdword ptr [rbp - 8], 0xe',
            '0x40082d:\tjle\t0x4007b7',
        ]),
        4196393: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196393, asm_memory_addresses=[4196393, 4196397], metadata={}, asm_lines=[
            '0x400829:\tcmp\tdword ptr [rbp - 8], 0xe',
            '0x40082d:\tjle\t0x4007b7',
        ]),
        4196399: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196399, asm_memory_addresses=[4196399, 4196406], metadata={}, asm_lines=[
            '0x40082f:\tmov\tdword ptr [rbp - 0x10], 0',
            '0x400836:\tjmp\t0x400846',
        ]),
        4196408: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196408, asm_memory_addresses=[4196408, 4196413], metadata={}, asm_lines=[
            '0x400838:\tmov\tedi, 0x2d',
            '0x40083d:\tcall\t0x400580',
        ]),
        4196418: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196418, asm_memory_addresses=[4196418, 4196422, 4196426], metadata={}, asm_lines=[
            '0x400842:\tadd\tdword ptr [rbp - 0x10], 1',
            '0x400846:\tcmp\tdword ptr [rbp - 0x10], 0x1f',
            '0x40084a:\tjle\t0x400838',
        ]),
        4196422: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196422, asm_memory_addresses=[4196422, 4196426], metadata={}, asm_lines=[
            '0x400846:\tcmp\tdword ptr [rbp - 0x10], 0x1f',
            '0x40084a:\tjle\t0x400838',
        ]),
        4196428: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196428, asm_memory_addresses=[4196428, 4196433], metadata={}, asm_lines=[
            '0x40084c:\tmov\tedi, 0xa',
            '0x400851:\tcall\t0x400580',
        ]),
        4196438: CFGBasicBlock(parent_function=__auto_functions[4196209], address=4196438, asm_memory_addresses=[4196438, 4196439, 4196440], metadata={}, asm_lines=[
            '0x400856:\tnop\t',
            '0x400857:\tleave\t',
            '0x400858:\tret\t',
        ]),
        4196441: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196441, asm_memory_addresses=[4196441, 4196442, 4196445, 4196449, 4196452, 4196455, 4196462, 4196466], metadata={}, asm_lines=[
            '0x400859:\tpush\trbp',
            '0x40085a:\tmov\trbp, rsp',
            '0x40085d:\tmov\tqword ptr [rbp - 0x18], rdi',
            '0x400861:\tmov\tdword ptr [rbp - 0x1c], esi',
            '0x400864:\tmov\tdword ptr [rbp - 0x20], edx',
            '0x400867:\tmov\tdword ptr [rbp - 4], 0',
            '0x40086e:\tcmp\tdword ptr [rbp - 0x1c], 0',
            '0x400872:\tjle\t0x4008ac',
        ]),
        4196468: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196468, asm_memory_addresses=[4196468, 4196472], metadata={}, asm_lines=[
            '0x400874:\tcmp\tdword ptr [rbp - 0x20], 0',
            '0x400878:\tjle\t0x4008ac',
        ]),
        4196474: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196474, asm_memory_addresses=[4196474, 4196477, 4196480, 4196483, 4196487, 4196490, 4196493, 4196497, 4196501, 4196504, 4196507, 4196510, 4196512, 4196516, 4196518], metadata={}, asm_lines=[
            '0x40087a:\tmov\teax, dword ptr [rbp - 0x1c]',
            '0x40087d:\tmovsxd\trdx, eax',
            '0x400880:\tmov\trax, rdx',
            '0x400883:\tshl\trax, 4',
            '0x400887:\tsub\trax, rdx',
            '0x40088a:\tadd\trax, rax',
            '0x40088d:\tlea\trdx, [rax - 0x1e]',
            '0x400891:\tmov\trax, qword ptr [rbp - 0x18]',
            '0x400895:\tadd\trdx, rax',
            '0x400898:\tmov\teax, dword ptr [rbp - 0x20]',
            '0x40089b:\tsub\teax, 1',
            '0x40089e:\tcdqe\t',
            '0x4008a0:\tmovzx\teax, byte ptr [rdx + rax]',
            '0x4008a4:\ttest\tal, al',
            '0x4008a6:\tje\t0x4008ac',
        ]),
        4196520: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196520, asm_memory_addresses=[4196520, 4196524, 4196528], metadata={}, asm_lines=[
            '0x4008a8:\tadd\tdword ptr [rbp - 4], 1',
            '0x4008ac:\tcmp\tdword ptr [rbp - 0x1c], 0',
            '0x4008b0:\tjle\t0x4008e1',
        ]),
        4196524: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196524, asm_memory_addresses=[4196524, 4196528], metadata={}, asm_lines=[
            '0x4008ac:\tcmp\tdword ptr [rbp - 0x1c], 0',
            '0x4008b0:\tjle\t0x4008e1',
        ]),
        4196530: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196530, asm_memory_addresses=[4196530, 4196533, 4196536, 4196539, 4196543, 4196546, 4196549, 4196553, 4196557, 4196560, 4196563, 4196565, 4196569, 4196571], metadata={}, asm_lines=[
            '0x4008b2:\tmov\teax, dword ptr [rbp - 0x1c]',
            '0x4008b5:\tmovsxd\trdx, eax',
            '0x4008b8:\tmov\trax, rdx',
            '0x4008bb:\tshl\trax, 4',
            '0x4008bf:\tsub\trax, rdx',
            '0x4008c2:\tadd\trax, rax',
            '0x4008c5:\tlea\trdx, [rax - 0x1e]',
            '0x4008c9:\tmov\trax, qword ptr [rbp - 0x18]',
            '0x4008cd:\tadd\trdx, rax',
            '0x4008d0:\tmov\teax, dword ptr [rbp - 0x20]',
            '0x4008d3:\tcdqe\t',
            '0x4008d5:\tmovzx\teax, byte ptr [rdx + rax]',
            '0x4008d9:\ttest\tal, al',
            '0x4008db:\tje\t0x4008e1',
        ]),
        4196573: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196573, asm_memory_addresses=[4196573, 4196577, 4196581], metadata={}, asm_lines=[
            '0x4008dd:\tadd\tdword ptr [rbp - 4], 1',
            '0x4008e1:\tcmp\tdword ptr [rbp - 0x1c], 0',
            '0x4008e5:\tjle\t0x40091f',
        ]),
        4196577: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196577, asm_memory_addresses=[4196577, 4196581], metadata={}, asm_lines=[
            '0x4008e1:\tcmp\tdword ptr [rbp - 0x1c], 0',
            '0x4008e5:\tjle\t0x40091f',
        ]),
        4196583: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196583, asm_memory_addresses=[4196583, 4196587], metadata={}, asm_lines=[
            '0x4008e7:\tcmp\tdword ptr [rbp - 0x20], 0x1c',
            '0x4008eb:\tjg\t0x40091f',
        ]),
        4196589: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196589, asm_memory_addresses=[4196589, 4196592, 4196595, 4196598, 4196602, 4196605, 4196608, 4196612, 4196616, 4196619, 4196622, 4196625, 4196627, 4196631, 4196633], metadata={}, asm_lines=[
            '0x4008ed:\tmov\teax, dword ptr [rbp - 0x1c]',
            '0x4008f0:\tmovsxd\trdx, eax',
            '0x4008f3:\tmov\trax, rdx',
            '0x4008f6:\tshl\trax, 4',
            '0x4008fa:\tsub\trax, rdx',
            '0x4008fd:\tadd\trax, rax',
            '0x400900:\tlea\trdx, [rax - 0x1e]',
            '0x400904:\tmov\trax, qword ptr [rbp - 0x18]',
            '0x400908:\tadd\trdx, rax',
            '0x40090b:\tmov\teax, dword ptr [rbp - 0x20]',
            '0x40090e:\tadd\teax, 1',
            '0x400911:\tcdqe\t',
            '0x400913:\tmovzx\teax, byte ptr [rdx + rax]',
            '0x400917:\ttest\tal, al',
            '0x400919:\tje\t0x40091f',
        ]),
        4196635: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196635, asm_memory_addresses=[4196635, 4196639, 4196643], metadata={}, asm_lines=[
            '0x40091b:\tadd\tdword ptr [rbp - 4], 1',
            '0x40091f:\tcmp\tdword ptr [rbp - 0x20], 0',
            '0x400923:\tjle\t0x400956',
        ]),
        4196639: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196639, asm_memory_addresses=[4196639, 4196643], metadata={}, asm_lines=[
            '0x40091f:\tcmp\tdword ptr [rbp - 0x20], 0',
            '0x400923:\tjle\t0x400956',
        ]),
        4196645: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196645, asm_memory_addresses=[4196645, 4196648, 4196651, 4196654, 4196658, 4196661, 4196664, 4196667, 4196671, 4196674, 4196677, 4196680, 4196682, 4196686, 4196688], metadata={}, asm_lines=[
            '0x400925:\tmov\teax, dword ptr [rbp - 0x1c]',
            '0x400928:\tmovsxd\trdx, eax',
            '0x40092b:\tmov\trax, rdx',
            '0x40092e:\tshl\trax, 4',
            '0x400932:\tsub\trax, rdx',
            '0x400935:\tadd\trax, rax',
            '0x400938:\tmov\trdx, rax',
            '0x40093b:\tmov\trax, qword ptr [rbp - 0x18]',
            '0x40093f:\tadd\trdx, rax',
            '0x400942:\tmov\teax, dword ptr [rbp - 0x20]',
            '0x400945:\tsub\teax, 1',
            '0x400948:\tcdqe\t',
            '0x40094a:\tmovzx\teax, byte ptr [rdx + rax]',
            '0x40094e:\ttest\tal, al',
            '0x400950:\tje\t0x400956',
        ]),
        4196690: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196690, asm_memory_addresses=[4196690, 4196694, 4196698], metadata={}, asm_lines=[
            '0x400952:\tadd\tdword ptr [rbp - 4], 1',
            '0x400956:\tcmp\tdword ptr [rbp - 0x20], 0x1c',
            '0x40095a:\tjg\t0x40098d',
        ]),
        4196694: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196694, asm_memory_addresses=[4196694, 4196698], metadata={}, asm_lines=[
            '0x400956:\tcmp\tdword ptr [rbp - 0x20], 0x1c',
            '0x40095a:\tjg\t0x40098d',
        ]),
        4196700: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196700, asm_memory_addresses=[4196700, 4196703, 4196706, 4196709, 4196713, 4196716, 4196719, 4196722, 4196726, 4196729, 4196732, 4196735, 4196737, 4196741, 4196743], metadata={}, asm_lines=[
            '0x40095c:\tmov\teax, dword ptr [rbp - 0x1c]',
            '0x40095f:\tmovsxd\trdx, eax',
            '0x400962:\tmov\trax, rdx',
            '0x400965:\tshl\trax, 4',
            '0x400969:\tsub\trax, rdx',
            '0x40096c:\tadd\trax, rax',
            '0x40096f:\tmov\trdx, rax',
            '0x400972:\tmov\trax, qword ptr [rbp - 0x18]',
            '0x400976:\tadd\trdx, rax',
            '0x400979:\tmov\teax, dword ptr [rbp - 0x20]',
            '0x40097c:\tadd\teax, 1',
            '0x40097f:\tcdqe\t',
            '0x400981:\tmovzx\teax, byte ptr [rdx + rax]',
            '0x400985:\ttest\tal, al',
            '0x400987:\tje\t0x40098d',
        ]),
        4196745: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196745, asm_memory_addresses=[4196745, 4196749, 4196753], metadata={}, asm_lines=[
            '0x400989:\tadd\tdword ptr [rbp - 4], 1',
            '0x40098d:\tcmp\tdword ptr [rbp - 0x1c], 0xd',
            '0x400991:\tjg\t0x4009cd',
        ]),
        4196749: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196749, asm_memory_addresses=[4196749, 4196753], metadata={}, asm_lines=[
            '0x40098d:\tcmp\tdword ptr [rbp - 0x1c], 0xd',
            '0x400991:\tjg\t0x4009cd',
        ]),
        4196755: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196755, asm_memory_addresses=[4196755, 4196759], metadata={}, asm_lines=[
            '0x400993:\tcmp\tdword ptr [rbp - 0x20], 0',
            '0x400997:\tjle\t0x4009cd',
        ]),
        4196761: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196761, asm_memory_addresses=[4196761, 4196764, 4196766, 4196770, 4196773, 4196777, 4196780, 4196783, 4196786, 4196790, 4196793, 4196796, 4196799, 4196801, 4196805, 4196807], metadata={}, asm_lines=[
            '0x400999:\tmov\teax, dword ptr [rbp - 0x1c]',
            '0x40099c:\tcdqe\t',
            '0x40099e:\tlea\trdx, [rax + 1]',
            '0x4009a2:\tmov\trax, rdx',
            '0x4009a5:\tshl\trax, 4',
            '0x4009a9:\tsub\trax, rdx',
            '0x4009ac:\tadd\trax, rax',
            '0x4009af:\tmov\trdx, rax',
            '0x4009b2:\tmov\trax, qword ptr [rbp - 0x18]',
            '0x4009b6:\tadd\trdx, rax',
            '0x4009b9:\tmov\teax, dword ptr [rbp - 0x20]',
            '0x4009bc:\tsub\teax, 1',
            '0x4009bf:\tcdqe\t',
            '0x4009c1:\tmovzx\teax, byte ptr [rdx + rax]',
            '0x4009c5:\ttest\tal, al',
            '0x4009c7:\tje\t0x4009cd',
        ]),
        4196809: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196809, asm_memory_addresses=[4196809, 4196813, 4196817], metadata={}, asm_lines=[
            '0x4009c9:\tadd\tdword ptr [rbp - 4], 1',
            '0x4009cd:\tcmp\tdword ptr [rbp - 0x1c], 0xd',
            '0x4009d1:\tjg\t0x400a04',
        ]),
        4196813: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196813, asm_memory_addresses=[4196813, 4196817], metadata={}, asm_lines=[
            '0x4009cd:\tcmp\tdword ptr [rbp - 0x1c], 0xd',
            '0x4009d1:\tjg\t0x400a04',
        ]),
        4196819: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196819, asm_memory_addresses=[4196819, 4196822, 4196824, 4196828, 4196831, 4196835, 4196838, 4196841, 4196844, 4196848, 4196851, 4196854, 4196856, 4196860, 4196862], metadata={}, asm_lines=[
            '0x4009d3:\tmov\teax, dword ptr [rbp - 0x1c]',
            '0x4009d6:\tcdqe\t',
            '0x4009d8:\tlea\trdx, [rax + 1]',
            '0x4009dc:\tmov\trax, rdx',
            '0x4009df:\tshl\trax, 4',
            '0x4009e3:\tsub\trax, rdx',
            '0x4009e6:\tadd\trax, rax',
            '0x4009e9:\tmov\trdx, rax',
            '0x4009ec:\tmov\trax, qword ptr [rbp - 0x18]',
            '0x4009f0:\tadd\trdx, rax',
            '0x4009f3:\tmov\teax, dword ptr [rbp - 0x20]',
            '0x4009f6:\tcdqe\t',
            '0x4009f8:\tmovzx\teax, byte ptr [rdx + rax]',
            '0x4009fc:\ttest\tal, al',
            '0x4009fe:\tje\t0x400a04',
        ]),
        4196864: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196864, asm_memory_addresses=[4196864, 4196868, 4196872], metadata={}, asm_lines=[
            '0x400a00:\tadd\tdword ptr [rbp - 4], 1',
            '0x400a04:\tcmp\tdword ptr [rbp - 0x1c], 0xd',
            '0x400a08:\tjg\t0x400a44',
        ]),
        4196868: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196868, asm_memory_addresses=[4196868, 4196872], metadata={}, asm_lines=[
            '0x400a04:\tcmp\tdword ptr [rbp - 0x1c], 0xd',
            '0x400a08:\tjg\t0x400a44',
        ]),
        4196874: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196874, asm_memory_addresses=[4196874, 4196878], metadata={}, asm_lines=[
            '0x400a0a:\tcmp\tdword ptr [rbp - 0x20], 0x1c',
            '0x400a0e:\tjg\t0x400a44',
        ]),
        4196880: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196880, asm_memory_addresses=[4196880, 4196883, 4196885, 4196889, 4196892, 4196896, 4196899, 4196902, 4196905, 4196909, 4196912, 4196915, 4196918, 4196920, 4196924, 4196926], metadata={}, asm_lines=[
            '0x400a10:\tmov\teax, dword ptr [rbp - 0x1c]',
            '0x400a13:\tcdqe\t',
            '0x400a15:\tlea\trdx, [rax + 1]',
            '0x400a19:\tmov\trax, rdx',
            '0x400a1c:\tshl\trax, 4',
            '0x400a20:\tsub\trax, rdx',
            '0x400a23:\tadd\trax, rax',
            '0x400a26:\tmov\trdx, rax',
            '0x400a29:\tmov\trax, qword ptr [rbp - 0x18]',
            '0x400a2d:\tadd\trdx, rax',
            '0x400a30:\tmov\teax, dword ptr [rbp - 0x20]',
            '0x400a33:\tadd\teax, 1',
            '0x400a36:\tcdqe\t',
            '0x400a38:\tmovzx\teax, byte ptr [rdx + rax]',
            '0x400a3c:\ttest\tal, al',
            '0x400a3e:\tje\t0x400a44',
        ]),
        4196928: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196928, asm_memory_addresses=[4196928, 4196932, 4196935, 4196936], metadata={}, asm_lines=[
            '0x400a40:\tadd\tdword ptr [rbp - 4], 1',
            '0x400a44:\tmov\teax, dword ptr [rbp - 4]',
            '0x400a47:\tpop\trbp',
            '0x400a48:\tret\t',
        ]),
        4196932: CFGBasicBlock(parent_function=__auto_functions[4196441], address=4196932, asm_memory_addresses=[4196932, 4196935, 4196936], metadata={}, asm_lines=[
            '0x400a44:\tmov\teax, dword ptr [rbp - 4]',
            '0x400a47:\tpop\trbp',
            '0x400a48:\tret\t',
        ]),
        4196937: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4196937, asm_memory_addresses=[4196937, 4196938, 4196941, 4196948, 4196953], metadata={}, asm_lines=[
            '0x400a49:\tpush\trbp',
            '0x400a4a:\tmov\trbp, rsp',
            '0x400a4d:\tsub\trsp, 0x3b0',
            '0x400a54:\tmov\tedi, 0x3039',
            '0x400a59:\tcall\t0x4005b0',
        ]),
        4196958: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4196958, asm_memory_addresses=[4196958, 4196965], metadata={}, asm_lines=[
            '0x400a5e:\tmov\tdword ptr [rbp - 4], 0',
            '0x400a65:\tjmp\t0x400ace',
        ]),
        4196967: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4196967, asm_memory_addresses=[4196967, 4196974], metadata={}, asm_lines=[
            '0x400a67:\tmov\tdword ptr [rbp - 8], 0',
            '0x400a6e:\tjmp\t0x400ac4',
        ]),
        4196976: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4196976, asm_memory_addresses=[4196976], metadata={}, asm_lines=[
            '0x400a70:\tcall\t0x4005c0',
        ]),
        4196981: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4196981, asm_memory_addresses=[4196981, 4196985, 4196993, 4196997, 4197000, 4197008, 4197011, 4197014, 4197016, 4197019, 4197022, 4197025, 4197028, 4197031, 4197035, 4197038, 4197041, 4197044, 4197047, 4197053, 4197056, 4197060, 4197064], metadata={}, asm_lines=[
            '0x400a75:\tcvtsi2ss\txmm0, eax',
            '0x400a79:\tmovss\txmm1, dword ptr [rip + 0x2e7]',
            '0x400a81:\tdivss\txmm0, xmm1',
            '0x400a85:\tmovaps\txmm1, xmm0',
            '0x400a88:\tmovss\txmm0, dword ptr [rip + 0x2dc]',
            '0x400a90:\tcomiss\txmm0, xmm1',
            '0x400a93:\tseta\tal',
            '0x400a96:\tmov\tesi, eax',
            '0x400a98:\tmov\teax, dword ptr [rbp - 8]',
            '0x400a9b:\tmovsxd\trcx, eax',
            '0x400a9e:\tmov\teax, dword ptr [rbp - 4]',
            '0x400aa1:\tmovsxd\trdx, eax',
            '0x400aa4:\tmov\trax, rdx',
            '0x400aa7:\tshl\trax, 4',
            '0x400aab:\tsub\trax, rdx',
            '0x400aae:\tadd\trax, rax',
            '0x400ab1:\tadd\trax, rbp',
            '0x400ab4:\tadd\trax, rcx',
            '0x400ab7:\tsub\trax, 0x1e0',
            '0x400abd:\tmov\tbyte ptr [rax], sil',
            '0x400ac0:\tadd\tdword ptr [rbp - 8], 1',
            '0x400ac4:\tcmp\tdword ptr [rbp - 8], 0x1d',
            '0x400ac8:\tjle\t0x400a70',
        ]),
        4197060: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197060, asm_memory_addresses=[4197060, 4197064], metadata={}, asm_lines=[
            '0x400ac4:\tcmp\tdword ptr [rbp - 8], 0x1d',
            '0x400ac8:\tjle\t0x400a70',
        ]),
        4197066: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197066, asm_memory_addresses=[4197066, 4197070, 4197074], metadata={}, asm_lines=[
            '0x400aca:\tadd\tdword ptr [rbp - 4], 1',
            '0x400ace:\tcmp\tdword ptr [rbp - 4], 0xe',
            '0x400ad2:\tjle\t0x400a67',
        ]),
        4197070: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197070, asm_memory_addresses=[4197070, 4197074], metadata={}, asm_lines=[
            '0x400ace:\tcmp\tdword ptr [rbp - 4], 0xe',
            '0x400ad2:\tjle\t0x400a67',
        ]),
        4197076: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197076, asm_memory_addresses=[4197076, 4197083, 4197086], metadata={}, asm_lines=[
            '0x400ad4:\tlea\trax, [rbp - 0x1e0]',
            '0x400adb:\tmov\trdi, rax',
            '0x400ade:\tcall\t0x400771',
        ]),
        4197091: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197091, asm_memory_addresses=[4197091, 4197098], metadata={}, asm_lines=[
            '0x400ae3:\tmov\tdword ptr [rbp - 0xc], 0',
            '0x400aea:\tjmp\t0x400c37',
        ]),
        4197103: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197103, asm_memory_addresses=[4197103, 4197110], metadata={}, asm_lines=[
            '0x400aef:\tmov\tdword ptr [rbp - 0x10], 0',
            '0x400af6:\tjmp\t0x400c29',
        ]),
        4197115: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197115, asm_memory_addresses=[4197115, 4197118, 4197121, 4197128, 4197130, 4197133], metadata={}, asm_lines=[
            '0x400afb:\tmov\tedx, dword ptr [rbp - 0x10]',
            '0x400afe:\tmov\tecx, dword ptr [rbp - 0xc]',
            '0x400b01:\tlea\trax, [rbp - 0x1e0]',
            '0x400b08:\tmov\tesi, ecx',
            '0x400b0a:\tmov\trdi, rax',
            '0x400b0d:\tcall\t0x400859',
        ]),
        4197138: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197138, asm_memory_addresses=[4197138, 4197141, 4197144, 4197147, 4197150, 4197153, 4197156, 4197160, 4197163, 4197166, 4197169, 4197172, 4197178, 4197181, 4197183], metadata={}, asm_lines=[
            '0x400b12:\tmov\tdword ptr [rbp - 0x1c], eax',
            '0x400b15:\tmov\teax, dword ptr [rbp - 0x10]',
            '0x400b18:\tmovsxd\trcx, eax',
            '0x400b1b:\tmov\teax, dword ptr [rbp - 0xc]',
            '0x400b1e:\tmovsxd\trdx, eax',
            '0x400b21:\tmov\trax, rdx',
            '0x400b24:\tshl\trax, 4',
            '0x400b28:\tsub\trax, rdx',
            '0x400b2b:\tadd\trax, rax',
            '0x400b2e:\tadd\trax, rbp',
            '0x400b31:\tadd\trax, rcx',
            '0x400b34:\tsub\trax, 0x1e0',
            '0x400b3a:\tmovzx\teax, byte ptr [rax]',
            '0x400b3d:\ttest\tal, al',
            '0x400b3f:\tje\t0x400b7a',
        ]),
        4197185: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197185, asm_memory_addresses=[4197185, 4197189], metadata={}, asm_lines=[
            '0x400b41:\tcmp\tdword ptr [rbp - 0x1c], 1',
            '0x400b45:\tjle\t0x400b4d',
        ]),
        4197191: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197191, asm_memory_addresses=[4197191, 4197195], metadata={}, asm_lines=[
            '0x400b47:\tcmp\tdword ptr [rbp - 0x1c], 3',
            '0x400b4b:\tjle\t0x400b7a',
        ]),
        4197197: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197197, asm_memory_addresses=[4197197, 4197200, 4197203, 4197206, 4197209, 4197212, 4197216, 4197219, 4197222, 4197225, 4197228, 4197234, 4197237], metadata={}, asm_lines=[
            '0x400b4d:\tmov\teax, dword ptr [rbp - 0x10]',
            '0x400b50:\tmovsxd\trcx, eax',
            '0x400b53:\tmov\teax, dword ptr [rbp - 0xc]',
            '0x400b56:\tmovsxd\trdx, eax',
            '0x400b59:\tmov\trax, rdx',
            '0x400b5c:\tshl\trax, 4',
            '0x400b60:\tsub\trax, rdx',
            '0x400b63:\tadd\trax, rax',
            '0x400b66:\tadd\trax, rbp',
            '0x400b69:\tadd\trax, rcx',
            '0x400b6c:\tsub\trax, 0x3b0',
            '0x400b72:\tmov\tbyte ptr [rax], 0',
            '0x400b75:\tjmp\t0x400c25',
        ]),
        4197242: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197242, asm_memory_addresses=[4197242, 4197245, 4197248, 4197251, 4197254, 4197257, 4197261, 4197264, 4197267, 4197270, 4197273, 4197279, 4197282, 4197284], metadata={}, asm_lines=[
            '0x400b7a:\tmov\teax, dword ptr [rbp - 0x10]',
            '0x400b7d:\tmovsxd\trcx, eax',
            '0x400b80:\tmov\teax, dword ptr [rbp - 0xc]',
            '0x400b83:\tmovsxd\trdx, eax',
            '0x400b86:\tmov\trax, rdx',
            '0x400b89:\tshl\trax, 4',
            '0x400b8d:\tsub\trax, rdx',
            '0x400b90:\tadd\trax, rax',
            '0x400b93:\tadd\trax, rbp',
            '0x400b96:\tadd\trax, rcx',
            '0x400b99:\tsub\trax, 0x1e0',
            '0x400b9f:\tmovzx\teax, byte ptr [rax]',
            '0x400ba2:\ttest\tal, al',
            '0x400ba4:\tjne\t0x400bd6',
        ]),
        4197286: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197286, asm_memory_addresses=[4197286, 4197290], metadata={}, asm_lines=[
            '0x400ba6:\tcmp\tdword ptr [rbp - 0x1c], 3',
            '0x400baa:\tjne\t0x400bd6',
        ]),
        4197292: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197292, asm_memory_addresses=[4197292, 4197295, 4197298, 4197301, 4197304, 4197307, 4197311, 4197314, 4197317, 4197320, 4197323, 4197329, 4197332], metadata={}, asm_lines=[
            '0x400bac:\tmov\teax, dword ptr [rbp - 0x10]',
            '0x400baf:\tmovsxd\trcx, eax',
            '0x400bb2:\tmov\teax, dword ptr [rbp - 0xc]',
            '0x400bb5:\tmovsxd\trdx, eax',
            '0x400bb8:\tmov\trax, rdx',
            '0x400bbb:\tshl\trax, 4',
            '0x400bbf:\tsub\trax, rdx',
            '0x400bc2:\tadd\trax, rax',
            '0x400bc5:\tadd\trax, rbp',
            '0x400bc8:\tadd\trax, rcx',
            '0x400bcb:\tsub\trax, 0x3b0',
            '0x400bd1:\tmov\tbyte ptr [rax], 1',
            '0x400bd4:\tjmp\t0x400c25',
        ]),
        4197334: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197334, asm_memory_addresses=[4197334, 4197337, 4197340, 4197343, 4197346, 4197349, 4197353, 4197356, 4197359, 4197362, 4197365, 4197371, 4197374, 4197377, 4197380, 4197383, 4197386, 4197389, 4197393, 4197396, 4197399, 4197402, 4197405, 4197411, 4197413, 4197417, 4197421], metadata={}, asm_lines=[
            '0x400bd6:\tmov\teax, dword ptr [rbp - 0x10]',
            '0x400bd9:\tmovsxd\trcx, eax',
            '0x400bdc:\tmov\teax, dword ptr [rbp - 0xc]',
            '0x400bdf:\tmovsxd\trdx, eax',
            '0x400be2:\tmov\trax, rdx',
            '0x400be5:\tshl\trax, 4',
            '0x400be9:\tsub\trax, rdx',
            '0x400bec:\tadd\trax, rax',
            '0x400bef:\tadd\trax, rbp',
            '0x400bf2:\tadd\trax, rcx',
            '0x400bf5:\tsub\trax, 0x1e0',
            '0x400bfb:\tmovzx\tecx, byte ptr [rax]',
            '0x400bfe:\tmov\teax, dword ptr [rbp - 0x10]',
            '0x400c01:\tmovsxd\trsi, eax',
            '0x400c04:\tmov\teax, dword ptr [rbp - 0xc]',
            '0x400c07:\tmovsxd\trdx, eax',
            '0x400c0a:\tmov\trax, rdx',
            '0x400c0d:\tshl\trax, 4',
            '0x400c11:\tsub\trax, rdx',
            '0x400c14:\tadd\trax, rax',
            '0x400c17:\tadd\trax, rbp',
            '0x400c1a:\tadd\trax, rsi',
            '0x400c1d:\tsub\trax, 0x3b0',
            '0x400c23:\tmov\tbyte ptr [rax], cl',
            '0x400c25:\tadd\tdword ptr [rbp - 0x10], 1',
            '0x400c29:\tcmp\tdword ptr [rbp - 0x10], 0x1d',
            '0x400c2d:\tjle\t0x400afb',
        ]),
        4197413: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197413, asm_memory_addresses=[4197413, 4197417, 4197421], metadata={}, asm_lines=[
            '0x400c25:\tadd\tdword ptr [rbp - 0x10], 1',
            '0x400c29:\tcmp\tdword ptr [rbp - 0x10], 0x1d',
            '0x400c2d:\tjle\t0x400afb',
        ]),
        4197417: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197417, asm_memory_addresses=[4197417, 4197421], metadata={}, asm_lines=[
            '0x400c29:\tcmp\tdword ptr [rbp - 0x10], 0x1d',
            '0x400c2d:\tjle\t0x400afb',
        ]),
        4197427: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197427, asm_memory_addresses=[4197427, 4197431, 4197435], metadata={}, asm_lines=[
            '0x400c33:\tadd\tdword ptr [rbp - 0xc], 1',
            '0x400c37:\tcmp\tdword ptr [rbp - 0xc], 0xe',
            '0x400c3b:\tjle\t0x400aef',
        ]),
        4197431: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197431, asm_memory_addresses=[4197431, 4197435], metadata={}, asm_lines=[
            '0x400c37:\tcmp\tdword ptr [rbp - 0xc], 0xe',
            '0x400c3b:\tjle\t0x400aef',
        ]),
        4197441: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197441, asm_memory_addresses=[4197441, 4197448], metadata={}, asm_lines=[
            '0x400c41:\tmov\tdword ptr [rbp - 0x14], 0',
            '0x400c48:\tjmp\t0x400cb0',
        ]),
        4197450: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197450, asm_memory_addresses=[4197450, 4197457], metadata={}, asm_lines=[
            '0x400c4a:\tmov\tdword ptr [rbp - 0x18], 0',
            '0x400c51:\tjmp\t0x400ca6',
        ]),
        4197459: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197459, asm_memory_addresses=[4197459, 4197462, 4197465, 4197468, 4197471, 4197474, 4197478, 4197481, 4197484, 4197487, 4197490, 4197496, 4197499, 4197502, 4197505, 4197508, 4197511, 4197514, 4197518, 4197521, 4197524, 4197527, 4197530, 4197536, 4197538, 4197542, 4197546], metadata={}, asm_lines=[
            '0x400c53:\tmov\teax, dword ptr [rbp - 0x18]',
            '0x400c56:\tmovsxd\trcx, eax',
            '0x400c59:\tmov\teax, dword ptr [rbp - 0x14]',
            '0x400c5c:\tmovsxd\trdx, eax',
            '0x400c5f:\tmov\trax, rdx',
            '0x400c62:\tshl\trax, 4',
            '0x400c66:\tsub\trax, rdx',
            '0x400c69:\tadd\trax, rax',
            '0x400c6c:\tadd\trax, rbp',
            '0x400c6f:\tadd\trax, rcx',
            '0x400c72:\tsub\trax, 0x3b0',
            '0x400c78:\tmovzx\tecx, byte ptr [rax]',
            '0x400c7b:\tmov\teax, dword ptr [rbp - 0x18]',
            '0x400c7e:\tmovsxd\trsi, eax',
            '0x400c81:\tmov\teax, dword ptr [rbp - 0x14]',
            '0x400c84:\tmovsxd\trdx, eax',
            '0x400c87:\tmov\trax, rdx',
            '0x400c8a:\tshl\trax, 4',
            '0x400c8e:\tsub\trax, rdx',
            '0x400c91:\tadd\trax, rax',
            '0x400c94:\tadd\trax, rbp',
            '0x400c97:\tadd\trax, rsi',
            '0x400c9a:\tsub\trax, 0x1e0',
            '0x400ca0:\tmov\tbyte ptr [rax], cl',
            '0x400ca2:\tadd\tdword ptr [rbp - 0x18], 1',
            '0x400ca6:\tcmp\tdword ptr [rbp - 0x18], 0x1d',
            '0x400caa:\tjle\t0x400c53',
        ]),
        4197542: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197542, asm_memory_addresses=[4197542, 4197546], metadata={}, asm_lines=[
            '0x400ca6:\tcmp\tdword ptr [rbp - 0x18], 0x1d',
            '0x400caa:\tjle\t0x400c53',
        ]),
        4197548: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197548, asm_memory_addresses=[4197548, 4197552, 4197556], metadata={}, asm_lines=[
            '0x400cac:\tadd\tdword ptr [rbp - 0x14], 1',
            '0x400cb0:\tcmp\tdword ptr [rbp - 0x14], 0xe',
            '0x400cb4:\tjle\t0x400c4a',
        ]),
        4197552: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197552, asm_memory_addresses=[4197552, 4197556], metadata={}, asm_lines=[
            '0x400cb0:\tcmp\tdword ptr [rbp - 0x14], 0xe',
            '0x400cb4:\tjle\t0x400c4a',
        ]),
        4197558: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197558, asm_memory_addresses=[4197558, 4197563], metadata={}, asm_lines=[
            '0x400cb6:\tmov\tedi, 0x1f4',
            '0x400cbb:\tcall\t0x4006b6',
        ]),
        4197568: CFGBasicBlock(parent_function=__auto_functions[4196937], address=4197568, asm_memory_addresses=[4197568], metadata={}, asm_lines=[
            '0x400cc0:\tjmp\t0x400ad4',
        ]),
        4197573: CFGBasicBlock(parent_function=__auto_functions[4197573], address=4197573, asm_memory_addresses=[4197573, 4197583], metadata={}, asm_lines=[
            '0x400cc5:\tnop\tword ptr cs:[rax + rax]',
            '0x400ccf:\tnop\t',
        ]),
        4197584: CFGBasicBlock(parent_function=__auto_functions[4197584], address=4197584, asm_memory_addresses=[4197584, 4197588, 4197590, 4197593, 4197595, 4197598, 4197600, 4197603, 4197605, 4197612, 4197613, 4197620, 4197621, 4197624, 4197628], metadata={}, asm_lines=[
            '0x400cd0:\tendbr64\t',
            '0x400cd4:\tpush\tr15',
            '0x400cd6:\tmov\tr15, rdx',
            '0x400cd9:\tpush\tr14',
            '0x400cdb:\tmov\tr14, rsi',
            '0x400cde:\tpush\tr13',
            '0x400ce0:\tmov\tr13d, edi',
            '0x400ce3:\tpush\tr12',
            '0x400ce5:\tlea\tr12, [rip + 0x201114]',
            '0x400cec:\tpush\trbp',
            '0x400ced:\tlea\trbp, [rip + 0x201114]',
            '0x400cf4:\tpush\trbx',
            '0x400cf5:\tsub\trbp, r12',
            '0x400cf8:\tsub\trsp, 8',
            '0x400cfc:\tcall\t0x400550',
        ]),
        4197633: CFGBasicBlock(parent_function=__auto_functions[4197584], address=4197633, asm_memory_addresses=[4197633, 4197637], metadata={}, asm_lines=[
            '0x400d01:\tsar\trbp, 3',
            '0x400d05:\tje\t0x400d26',
        ]),
        4197639: CFGBasicBlock(parent_function=__auto_functions[4197584], address=4197639, asm_memory_addresses=[4197639, 4197641, 4197648, 4197651, 4197654, 4197657], metadata={}, asm_lines=[
            '0x400d07:\txor\tebx, ebx',
            '0x400d09:\tnop\tdword ptr [rax]',
            '0x400d10:\tmov\trdx, r15',
            '0x400d13:\tmov\trsi, r14',
            '0x400d16:\tmov\tedi, r13d',
            '0x400d19:\tcall\tqword ptr [r12 + rbx*8]',
        ]),
        4197648: CFGBasicBlock(parent_function=__auto_functions[4197584], address=4197648, asm_memory_addresses=[4197648, 4197651, 4197654, 4197657], metadata={}, asm_lines=[
            '0x400d10:\tmov\trdx, r15',
            '0x400d13:\tmov\trsi, r14',
            '0x400d16:\tmov\tedi, r13d',
            '0x400d19:\tcall\tqword ptr [r12 + rbx*8]',
        ]),
        4197661: CFGBasicBlock(parent_function=__auto_functions[4197584], address=4197661, asm_memory_addresses=[4197661, 4197665, 4197668], metadata={}, asm_lines=[
            '0x400d1d:\tadd\trbx, 1',
            '0x400d21:\tcmp\trbp, rbx',
            '0x400d24:\tjne\t0x400d10',
        ]),
        4197670: CFGBasicBlock(parent_function=__auto_functions[4197584], address=4197670, asm_memory_addresses=[4197670, 4197674, 4197675, 4197676, 4197678, 4197680, 4197682, 4197684], metadata={}, asm_lines=[
            '0x400d26:\tadd\trsp, 8',
            '0x400d2a:\tpop\trbx',
            '0x400d2b:\tpop\trbp',
            '0x400d2c:\tpop\tr12',
            '0x400d2e:\tpop\tr13',
            '0x400d30:\tpop\tr14',
            '0x400d32:\tpop\tr15',
            '0x400d34:\tret\t',
        ]),
        4197685: CFGBasicBlock(parent_function=__auto_functions[4197685], address=4197685, asm_memory_addresses=[4197685], metadata={}, asm_lines=[
            '0x400d35:\tnop\tword ptr cs:[rax + rax]',
        ]),
        4197696: CFGBasicBlock(parent_function=__auto_functions[4197696], address=4197696, asm_memory_addresses=[4197696, 4197700], metadata={}, asm_lines=[
            '0x400d40:\tendbr64\t',
            '0x400d44:\tret\t',
        ]),
        4197704: CFGBasicBlock(parent_function=__auto_functions[4197704], address=4197704, asm_memory_addresses=[4197704, 4197708, 4197712, 4197716], metadata={}, asm_lines=[
            '0x400d48:\tendbr64\t',
            '0x400d4c:\tsub\trsp, 8',
            '0x400d50:\tadd\trsp, 8',
            '0x400d54:\tret\t',
        ]),
        7340032: CFGBasicBlock(parent_function=__auto_functions[7340032], address=7340032, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        7340040: CFGBasicBlock(parent_function=__auto_functions[7340040], address=7340040, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        7340048: CFGBasicBlock(parent_function=__auto_functions[7340048], address=7340048, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        7340056: CFGBasicBlock(parent_function=__auto_functions[7340056], address=7340056, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        7340064: CFGBasicBlock(parent_function=__auto_functions[7340064], address=7340064, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        7340072: CFGBasicBlock(parent_function=__auto_functions[7340072], address=7340072, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        8392784: CFGBasicBlock(parent_function=__auto_functions[8392784], address=8392784, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        8392792: CFGBasicBlock(parent_function=__auto_functions[8392792], address=8392792, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
    }

    # Building all edges
    __auto_blocks[4195664].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195664], to_block=__auto_blocks[4195686], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4195664], to_block=__auto_blocks[4195684], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195684].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195684], to_block=__auto_blocks[8392792], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4195684], to_block=__auto_blocks[4195686], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195686].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195686], to_block=__auto_blocks[4197633], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195696].edges_out = set([
        
    ])

    __auto_blocks[4195708].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195708], to_block=__auto_blocks[4195712], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195712].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195712], to_block=__auto_blocks[7340040], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195728].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195728], to_block=__auto_blocks[7340048], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195744].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195744], to_block=__auto_blocks[7340056], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195760].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195760], to_block=__auto_blocks[7340064], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195776].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195776], to_block=__auto_blocks[7340072], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195792].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195792], to_block=__auto_blocks[7340032], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[4195838].edges_out = set([
        
    ])

    __auto_blocks[4195839].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195839], to_block=__auto_blocks[4195840], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195840].edges_out = set([
        
    ])

    __auto_blocks[4195845].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195845], to_block=__auto_blocks[4195856], edge_type=EdgeType.NORMAL),
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
        CFGEdge(from_block=__auto_blocks[4195887], to_block=__auto_blocks[8392784], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195889].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195889], to_block=__auto_blocks[4195896], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195896].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195896], to_block=__auto_blocks[4195990], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195897].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195897], to_block=__auto_blocks[4195904], edge_type=EdgeType.NORMAL),
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
        CFGEdge(from_block=__auto_blocks[4195952], to_block=__auto_blocks[8392784], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195954].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195954], to_block=__auto_blocks[4195960], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195960].edges_out = set([
        
    ])

    __auto_blocks[4195961].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195961], to_block=__auto_blocks[4195968], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195968].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195968], to_block=__auto_blocks[4195981], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4195968], to_block=__auto_blocks[4196000], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195981].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195981], to_block=__auto_blocks[4195856], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4195981], to_block=__auto_blocks[4195990], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4195990].edges_out = set([
        
    ])

    __auto_blocks[4195999].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4195999], to_block=__auto_blocks[4196000], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196000].edges_out = set([
        
    ])

    __auto_blocks[4196001].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196001], to_block=__auto_blocks[4196016], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196016].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196016], to_block=__auto_blocks[4195904], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196022].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196022], to_block=__auto_blocks[4196041], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196022], to_block=__auto_blocks[4196062], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196041].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196041], to_block=__auto_blocks[4195728], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4196041], to_block=__auto_blocks[4196046], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196046].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196046], to_block=__auto_blocks[4196207], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196062].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196062], to_block=__auto_blocks[4196164], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196164].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196164], to_block=__auto_blocks[4195744], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4196164], to_block=__auto_blocks[4196183], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196183].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196183], to_block=__auto_blocks[4196192], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196183], to_block=__auto_blocks[4196204], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196192].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196192], to_block=__auto_blocks[4195728], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4196192], to_block=__auto_blocks[4196197], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196197].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196197], to_block=__auto_blocks[4196204], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196197], to_block=__auto_blocks[4196164], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196204].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196204], to_block=__auto_blocks[4196207], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196207].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196207], to_block=__auto_blocks[4197568], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196209].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196209], to_block=__auto_blocks[4195712], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4196209], to_block=__auto_blocks[4196231], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196231].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196231], to_block=__auto_blocks[4196254], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196240].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196240], to_block=__auto_blocks[4195712], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4196240], to_block=__auto_blocks[4196250], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196250].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196250], to_block=__auto_blocks[4196254], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196254].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196254], to_block=__auto_blocks[4196260], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196254], to_block=__auto_blocks[4196240], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196260].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196260], to_block=__auto_blocks[4196270], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196260], to_block=__auto_blocks[4195712], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[4196270].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196270], to_block=__auto_blocks[4196393], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196279].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196279], to_block=__auto_blocks[4196289], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196279], to_block=__auto_blocks[4195712], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[4196289].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196289], to_block=__auto_blocks[4196363], edge_type=EdgeType.NORMAL),
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
        CFGEdge(from_block=__auto_blocks[4196352], to_block=__auto_blocks[4195712], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4196352], to_block=__auto_blocks[4196359], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196359].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196359], to_block=__auto_blocks[4196363], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196363].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196363], to_block=__auto_blocks[4196369], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196363], to_block=__auto_blocks[4196298], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196369].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196369], to_block=__auto_blocks[4195712], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4196369], to_block=__auto_blocks[4196379], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196379].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196379], to_block=__auto_blocks[4195712], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4196379], to_block=__auto_blocks[4196389], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196389].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196389], to_block=__auto_blocks[4196393], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196393].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196393], to_block=__auto_blocks[4196279], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196393], to_block=__auto_blocks[4196399], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196399].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196399], to_block=__auto_blocks[4196422], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196408].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196408], to_block=__auto_blocks[4195712], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4196408], to_block=__auto_blocks[4196418], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196418].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196418], to_block=__auto_blocks[4196422], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196422].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196422], to_block=__auto_blocks[4196428], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196422], to_block=__auto_blocks[4196408], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196428].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196428], to_block=__auto_blocks[4196438], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196428], to_block=__auto_blocks[4195712], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[4196438].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196438], to_block=__auto_blocks[4197091], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196441].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196441], to_block=__auto_blocks[4196524], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196441], to_block=__auto_blocks[4196468], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196468].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196468], to_block=__auto_blocks[4196524], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196468], to_block=__auto_blocks[4196474], edge_type=EdgeType.NORMAL),
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
        CFGEdge(from_block=__auto_blocks[4196874], to_block=__auto_blocks[4196880], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196874], to_block=__auto_blocks[4196932], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196880].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196880], to_block=__auto_blocks[4196932], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4196880], to_block=__auto_blocks[4196928], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196928].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196928], to_block=__auto_blocks[4196932], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196932].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196932], to_block=__auto_blocks[4197138], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196937].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196937], to_block=__auto_blocks[4195760], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4196937], to_block=__auto_blocks[4196958], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196958].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196958], to_block=__auto_blocks[4197070], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196967].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196967], to_block=__auto_blocks[4197060], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196976].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196976], to_block=__auto_blocks[4195776], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4196976], to_block=__auto_blocks[4196981], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4196981].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4196981], to_block=__auto_blocks[4197060], edge_type=EdgeType.NORMAL),
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
        CFGEdge(from_block=__auto_blocks[4197076], to_block=__auto_blocks[4196209], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4197076], to_block=__auto_blocks[4197091], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4197091].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197091], to_block=__auto_blocks[4197431], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4197103].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197103], to_block=__auto_blocks[4197417], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4197115].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197115], to_block=__auto_blocks[4196441], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4197115], to_block=__auto_blocks[4197138], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4197138].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197138], to_block=__auto_blocks[4197242], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197138], to_block=__auto_blocks[4197185], edge_type=EdgeType.NORMAL),
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
        CFGEdge(from_block=__auto_blocks[4197558], to_block=__auto_blocks[4197568], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197558], to_block=__auto_blocks[4196022], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[4197568].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197568], to_block=__auto_blocks[4197076], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4197573].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197573], to_block=__auto_blocks[4197584], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4197584].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197584], to_block=__auto_blocks[4195664], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4197584], to_block=__auto_blocks[4197633], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4197633].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197633], to_block=__auto_blocks[4197639], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197633], to_block=__auto_blocks[4197670], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4197639].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197639], to_block=__auto_blocks[4197648], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4197648].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197648], to_block=__auto_blocks[4197661], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197648], to_block=__auto_blocks[8392792], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[4197661].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197661], to_block=__auto_blocks[4197648], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4197661], to_block=__auto_blocks[4197670], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4197670].edges_out = set([
        
    ])

    __auto_blocks[4197685].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4197685], to_block=__auto_blocks[4197696], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4197696].edges_out = set([
        
    ])

    __auto_blocks[4197704].edges_out = set([
        
    ])

    __auto_blocks[7340032].edges_out = set([
        CFGEdge(from_block=__auto_blocks[7340032], to_block=__auto_blocks[4196937], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[7340032], to_block=__auto_blocks[4197696], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[7340032], to_block=__auto_blocks[4197584], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[7340040].edges_out = set([
        CFGEdge(from_block=__auto_blocks[7340040], to_block=__auto_blocks[4196270], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[7340040], to_block=__auto_blocks[4196438], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[7340040], to_block=__auto_blocks[4196231], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[7340040], to_block=__auto_blocks[4196389], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[7340040], to_block=__auto_blocks[4196289], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[7340040], to_block=__auto_blocks[4196250], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[7340040], to_block=__auto_blocks[4196418], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[7340040], to_block=__auto_blocks[4196359], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[7340040], to_block=__auto_blocks[4196379], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[7340048].edges_out = set([
        CFGEdge(from_block=__auto_blocks[7340048], to_block=__auto_blocks[4196046], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[7340048], to_block=__auto_blocks[4196197], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[7340056].edges_out = set([
        CFGEdge(from_block=__auto_blocks[7340056], to_block=__auto_blocks[4196183], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[7340064].edges_out = set([
        CFGEdge(from_block=__auto_blocks[7340064], to_block=__auto_blocks[4196958], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[7340072].edges_out = set([
        CFGEdge(from_block=__auto_blocks[7340072], to_block=__auto_blocks[4196981], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[8392784].edges_out = set([
        
    ])

    __auto_blocks[8392792].edges_out = set([
        
    ])

    # Set the edges_in on the blocks
    for b in __auto_blocks.values():
        for e in b.edges_out:
            e.to_block.edges_in.add(CFGEdge(b, e.to_block, e.edge_type))

    # Adding basic blocks to their associated functions
    __auto_functions[4195664].blocks = [
        __auto_blocks[4195664],
        __auto_blocks[4195686],
        __auto_blocks[4195684],
    ]

    __auto_functions[4195696].blocks = [
        __auto_blocks[4195696],
    ]

    __auto_functions[4195708].blocks = [
        __auto_blocks[4195708],
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

    __auto_functions[4195838].blocks = [
        __auto_blocks[4195838],
    ]

    __auto_functions[4195839].blocks = [
        __auto_blocks[4195839],
    ]

    __auto_functions[4195840].blocks = [
        __auto_blocks[4195840],
    ]

    __auto_functions[4195845].blocks = [
        __auto_blocks[4195845],
    ]

    __auto_functions[4195856].blocks = [
        __auto_blocks[4195856],
        __auto_blocks[4195896],
        __auto_blocks[4195875],
        __auto_blocks[4195887],
        __auto_blocks[4195889],
    ]

    __auto_functions[4195897].blocks = [
        __auto_blocks[4195897],
    ]

    __auto_functions[4195904].blocks = [
        __auto_blocks[4195904],
        __auto_blocks[4195960],
        __auto_blocks[4195940],
        __auto_blocks[4195952],
        __auto_blocks[4195954],
    ]

    __auto_functions[4195961].blocks = [
        __auto_blocks[4195961],
    ]

    __auto_functions[4195968].blocks = [
        __auto_blocks[4195968],
        __auto_blocks[4195981],
        __auto_blocks[4196000],
        __auto_blocks[4195990],
    ]

    __auto_functions[4195999].blocks = [
        __auto_blocks[4195999],
    ]

    __auto_functions[4196001].blocks = [
        __auto_blocks[4196001],
    ]

    __auto_functions[4196016].blocks = [
        __auto_blocks[4196016],
    ]

    __auto_functions[4196022].blocks = [
        __auto_blocks[4196022],
        __auto_blocks[4196041],
        __auto_blocks[4196062],
        __auto_blocks[4196046],
        __auto_blocks[4196183],
        __auto_blocks[4196207],
        __auto_blocks[4196204],
        __auto_blocks[4196192],
        __auto_blocks[4196197],
        __auto_blocks[4196164],
    ]

    __auto_functions[4196209].blocks = [
        __auto_blocks[4196209],
        __auto_blocks[4196231],
        __auto_blocks[4196254],
        __auto_blocks[4196240],
        __auto_blocks[4196260],
        __auto_blocks[4196250],
        __auto_blocks[4196270],
        __auto_blocks[4196393],
        __auto_blocks[4196279],
        __auto_blocks[4196399],
        __auto_blocks[4196289],
        __auto_blocks[4196422],
        __auto_blocks[4196363],
        __auto_blocks[4196408],
        __auto_blocks[4196428],
        __auto_blocks[4196298],
        __auto_blocks[4196369],
        __auto_blocks[4196418],
        __auto_blocks[4196438],
        __auto_blocks[4196347],
        __auto_blocks[4196340],
        __auto_blocks[4196379],
        __auto_blocks[4196359],
        __auto_blocks[4196352],
        __auto_blocks[4196389],
    ]

    __auto_functions[4196441].blocks = [
        __auto_blocks[4196441],
        __auto_blocks[4196524],
        __auto_blocks[4196468],
        __auto_blocks[4196577],
        __auto_blocks[4196530],
        __auto_blocks[4196474],
        __auto_blocks[4196639],
        __auto_blocks[4196583],
        __auto_blocks[4196573],
        __auto_blocks[4196520],
        __auto_blocks[4196694],
        __auto_blocks[4196645],
        __auto_blocks[4196589],
        __auto_blocks[4196700],
        __auto_blocks[4196749],
        __auto_blocks[4196690],
        __auto_blocks[4196635],
        __auto_blocks[4196745],
        __auto_blocks[4196755],
        __auto_blocks[4196813],
        __auto_blocks[4196761],
        __auto_blocks[4196819],
        __auto_blocks[4196868],
        __auto_blocks[4196809],
        __auto_blocks[4196864],
        __auto_blocks[4196874],
        __auto_blocks[4196932],
        __auto_blocks[4196880],
        __auto_blocks[4196928],
    ]

    __auto_functions[4196937].blocks = [
        __auto_blocks[4196937],
        __auto_blocks[4196958],
        __auto_blocks[4197070],
        __auto_blocks[4196967],
        __auto_blocks[4197076],
        __auto_blocks[4197060],
        __auto_blocks[4197091],
        __auto_blocks[4196976],
        __auto_blocks[4197066],
        __auto_blocks[4197431],
        __auto_blocks[4196981],
        __auto_blocks[4197103],
        __auto_blocks[4197441],
        __auto_blocks[4197417],
        __auto_blocks[4197552],
        __auto_blocks[4197115],
        __auto_blocks[4197427],
        __auto_blocks[4197450],
        __auto_blocks[4197558],
        __auto_blocks[4197138],
        __auto_blocks[4197542],
        __auto_blocks[4197568],
        __auto_blocks[4197242],
        __auto_blocks[4197185],
        __auto_blocks[4197459],
        __auto_blocks[4197548],
        __auto_blocks[4197286],
        __auto_blocks[4197334],
        __auto_blocks[4197197],
        __auto_blocks[4197191],
        __auto_blocks[4197292],
        __auto_blocks[4197413],
    ]

    __auto_functions[4197573].blocks = [
        __auto_blocks[4197573],
    ]

    __auto_functions[4197584].blocks = [
        __auto_blocks[4197584],
        __auto_blocks[4197633],
        __auto_blocks[4197670],
        __auto_blocks[4197639],
        __auto_blocks[4197661],
        __auto_blocks[4197648],
    ]

    __auto_functions[4197685].blocks = [
        __auto_blocks[4197685],
    ]

    __auto_functions[4197696].blocks = [
        __auto_blocks[4197696],
    ]

    __auto_functions[4197704].blocks = [
        __auto_blocks[4197704],
    ]

    __auto_functions[7340032].blocks = [
        __auto_blocks[7340032],
    ]

    __auto_functions[7340040].blocks = [
        __auto_blocks[7340040],
    ]

    __auto_functions[7340048].blocks = [
        __auto_blocks[7340048],
    ]

    __auto_functions[7340056].blocks = [
        __auto_blocks[7340056],
    ]

    __auto_functions[7340064].blocks = [
        __auto_blocks[7340064],
    ]

    __auto_functions[7340072].blocks = [
        __auto_blocks[7340072],
    ]

    __auto_functions[8392784].blocks = [
        __auto_blocks[8392784],
    ]

    __auto_functions[8392792].blocks = [
        __auto_blocks[8392792],
    ]

    expected = {
        'sorted_func_order': [4195664, 4195696, 4195708, 4195712, 4195728, 4195744, 4195760, 4195776, 4195792, 4195838, 4195839, 4195840, 4195845, 4195856, 4195897, 4195904, 4195961, 4195968, 4195999, 4196001, 4196016, 4196022, 4196209, 4196441, 4196937, 4197573, 4197584, 4197685, 4197696, 4197704, 7340032, 7340040, 7340048, 7340056, 7340064, 7340072, 8392784, 8392792],
        'sorted_block_order': [4195664, 4195684, 4195686, 4195696, 4195708, 4195712, 4195728, 4195744, 4195760, 4195776, 4195792, 4195838, 4195839, 4195840, 4195845, 4195856, 4195875, 4195887, 4195889, 4195896, 4195897, 4195904, 4195940, 4195952, 4195954, 4195960, 4195961, 4195968, 4195981, 4195990, 4195999, 4196000, 4196001, 4196016, 4196022, 4196041, 4196046, 4196062, 4196164, 4196183, 4196192, 4196197, 4196204, 4196207, 4196209, 4196231, 4196240, 4196250, 4196254, 4196260, 4196270, 4196279, 4196289, 4196298, 4196340, 4196347, 4196352, 4196359, 4196363, 4196369, 4196379, 4196389, 4196393, 4196399, 4196408, 4196418, 4196422, 4196428, 4196438, 4196441, 4196468, 4196474, 4196520, 4196524, 4196530, 4196573, 4196577, 4196583, 4196589, 4196635, 4196639, 4196645, 4196690, 4196694, 4196700, 4196745, 4196749, 4196755, 4196761, 4196809, 4196813, 4196819, 4196864, 4196868, 4196874, 4196880, 4196928, 4196932, 4196937, 4196958, 4196967, 4196976, 4196981, 4197060, 4197066, 4197070, 4197076, 4197091, 4197103, 4197115, 4197138, 4197185, 4197191, 4197197, 4197242, 4197286, 4197292, 4197334, 4197413, 4197417, 4197427, 4197431, 4197441, 4197450, 4197459, 4197542, 4197548, 4197552, 4197558, 4197568, 4197573, 4197584, 4197633, 4197639, 4197648, 4197661, 4197670, 4197685, 4197696, 4197704, 7340032, 7340040, 7340048, 7340056, 7340064, 7340072, 8392784, 8392792],
        'architecture': 'x86',
        'num_blocks': {4195664: 3, 4195696: 1, 4195708: 1, 4195712: 1, 4195728: 1, 4195744: 1, 4195760: 1, 4195776: 1, 4195792: 1, 4195838: 1, 4195839: 1, 4195840: 1, 4195845: 1, 4195856: 5, 4195897: 1, 4195904: 5, 4195961: 1, 4195968: 4, 4195999: 1, 4196001: 1, 4196016: 1, 4196022: 10, 4196209: 25, 4196441: 29, 4196937: 32, 4197573: 1, 4197584: 6, 4197685: 1, 4197696: 1, 4197704: 1, 7340032: 1, 7340040: 1, 7340048: 1, 7340056: 1, 7340064: 1, 7340072: 1, 8392784: 1, 8392792: 1},
        'num_asm_lines_per_block': {4195664: 5, 4195684: 1, 4195686: 2, 4195696: 2, 4195708: 1, 4195712: 1, 4195728: 1, 4195744: 1, 4195760: 1, 4195776: 1, 4195792: 12, 4195838: 1, 4195839: 1, 4195840: 2, 4195845: 2, 4195856: 4, 4195875: 3, 4195887: 1, 4195889: 1, 4195896: 1, 4195897: 1, 4195904: 9, 4195940: 3, 4195952: 1, 4195954: 1, 4195960: 1, 4195961: 1, 4195968: 3, 4195981: 3, 4195990: 3, 4195999: 1, 4196000: 1, 4196001: 2, 4196016: 2, 4196022: 6, 4196041: 1, 4196046: 3, 4196062: 29, 4196164: 5, 4196183: 3, 4196192: 1, 4196197: 3, 4196204: 3, 4196207: 2, 4196209: 6, 4196231: 2, 4196240: 2, 4196250: 3, 4196254: 2, 4196260: 2, 4196270: 2, 4196279: 2, 4196289: 2, 4196298: 14, 4196340: 2, 4196347: 3, 4196352: 2, 4196359: 3, 4196363: 2, 4196369: 2, 4196379: 2, 4196389: 3, 4196393: 2, 4196399: 2, 4196408: 2, 4196418: 3, 4196422: 2, 4196428: 2, 4196438: 3, 4196441: 8, 4196468: 2, 4196474: 15, 4196520: 3, 4196524: 2, 4196530: 14, 4196573: 3, 4196577: 2, 4196583: 2, 4196589: 15, 4196635: 3, 4196639: 2, 4196645: 15, 4196690: 3, 4196694: 2, 4196700: 15, 4196745: 3, 4196749: 2, 4196755: 2, 4196761: 16, 4196809: 3, 4196813: 2, 4196819: 15, 4196864: 3, 4196868: 2, 4196874: 2, 4196880: 16, 4196928: 4, 4196932: 3, 4196937: 5, 4196958: 2, 4196967: 2, 4196976: 1, 4196981: 23, 4197060: 2, 4197066: 3, 4197070: 2, 4197076: 3, 4197091: 2, 4197103: 2, 4197115: 6, 4197138: 15, 4197185: 2, 4197191: 2, 4197197: 13, 4197242: 14, 4197286: 2, 4197292: 13, 4197334: 27, 4197413: 3, 4197417: 2, 4197427: 3, 4197431: 2, 4197441: 2, 4197450: 2, 4197459: 27, 4197542: 2, 4197548: 3, 4197552: 2, 4197558: 2, 4197568: 1, 4197573: 2, 4197584: 15, 4197633: 2, 4197639: 6, 4197648: 4, 4197661: 3, 4197670: 8, 4197685: 1, 4197696: 2, 4197704: 4, 7340032: 0, 7340040: 0, 7340048: 0, 7340056: 0, 7340064: 0, 7340072: 0, 8392784: 0, 8392792: 0},
        'num_asm_lines_per_function': {4195664: 8, 4195696: 2, 4195708: 1, 4195712: 1, 4195728: 1, 4195744: 1, 4195760: 1, 4195776: 1, 4195792: 12, 4195838: 1, 4195839: 1, 4195840: 2, 4195845: 2, 4195856: 10, 4195897: 1, 4195904: 15, 4195961: 1, 4195968: 10, 4195999: 1, 4196001: 2, 4196016: 2, 4196022: 56, 4196209: 72, 4196441: 179, 4196937: 192, 4197573: 2, 4197584: 38, 4197685: 1, 4197696: 2, 4197704: 4, 7340032: 0, 7340040: 0, 7340048: 0, 7340056: 0, 7340064: 0, 7340072: 0, 8392784: 0, 8392792: 0},
        'num_functions': 38,
        'is_root_function': {4195664: False, 4195696: True, 4195708: True, 4195712: False, 4195728: False, 4195744: False, 4195760: False, 4195776: False, 4195792: True, 4195838: True, 4195839: True, 4195840: True, 4195845: True, 4195856: False, 4195897: True, 4195904: True, 4195961: True, 4195968: True, 4195999: True, 4196001: True, 4196016: True, 4196022: False, 4196209: False, 4196441: False, 4196937: False, 4197573: True, 4197584: False, 4197685: True, 4197696: False, 4197704: True, 7340032: False, 7340040: True, 7340048: True, 7340056: True, 7340064: True, 7340072: True, 8392784: True, 8392792: False},
        'is_recursive': {4195664: False, 4195696: False, 4195708: False, 4195712: False, 4195728: False, 4195744: False, 4195760: False, 4195776: False, 4195792: False, 4195838: False, 4195839: False, 4195840: False, 4195845: False, 4195856: False, 4195897: False, 4195904: False, 4195961: False, 4195968: False, 4195999: False, 4196001: False, 4196016: False, 4196022: False, 4196209: False, 4196441: False, 4196937: False, 4197573: False, 4197584: False, 4197685: False, 4197696: False, 4197704: False, 7340032: False, 7340040: False, 7340048: False, 7340056: False, 7340064: False, 7340072: False, 8392784: False, 8392792: False},
        'is_extern_function': {4195664: False, 4195696: False, 4195708: False, 4195712: True, 4195728: True, 4195744: True, 4195760: True, 4195776: True, 4195792: False, 4195838: False, 4195839: False, 4195840: False, 4195845: False, 4195856: False, 4195897: False, 4195904: False, 4195961: False, 4195968: False, 4195999: False, 4196001: False, 4196016: False, 4196022: False, 4196209: False, 4196441: False, 4196937: False, 4197573: False, 4197584: False, 4197685: False, 4197696: False, 4197704: False, 7340032: False, 7340040: False, 7340048: False, 7340056: False, 7340064: False, 7340072: False, 8392784: False, 8392792: False},
        'is_intern_function': {4195664: True, 4195696: True, 4195708: True, 4195712: False, 4195728: False, 4195744: False, 4195760: False, 4195776: False, 4195792: True, 4195838: True, 4195839: True, 4195840: True, 4195845: True, 4195856: True, 4195897: True, 4195904: True, 4195961: True, 4195968: True, 4195999: True, 4196001: True, 4196016: True, 4196022: True, 4196209: True, 4196441: True, 4196937: True, 4197573: True, 4197584: True, 4197685: True, 4197696: True, 4197704: True, 7340032: True, 7340040: True, 7340048: True, 7340056: True, 7340064: True, 7340072: True, 8392784: True, 8392792: True},
        'function_entry_block': {4195664: 4195664, 4195696: 4195696, 4195708: 4195708, 4195712: 4195712, 4195728: 4195728, 4195744: 4195744, 4195760: 4195760, 4195776: 4195776, 4195792: 4195792, 4195838: 4195838, 4195839: 4195839, 4195840: 4195840, 4195845: 4195845, 4195856: 4195856, 4195897: 4195897, 4195904: 4195904, 4195961: 4195961, 4195968: 4195968, 4195999: 4195999, 4196001: 4196001, 4196016: 4196016, 4196022: 4196022, 4196209: 4196209, 4196441: 4196441, 4196937: 4196937, 4197573: 4197573, 4197584: 4197584, 4197685: 4197685, 4197696: 4197696, 4197704: 4197704, 7340032: 7340032, 7340040: 7340040, 7340048: 7340048, 7340056: 7340056, 7340064: 7340064, 7340072: 7340072, 8392784: 8392784, 8392792: 8392792},
        'called_by': {4195664: {4197584}, 4195696: set(), 4195708: set(), 4195712: {4196352, 4196260, 4196428, 4196240, 4196209, 4196369, 4196279, 4196408, 4196379}, 4195728: {4196192, 4196041}, 4195744: {4196164}, 4195760: {4196937}, 4195776: {4196976}, 4195792: set(), 4195838: set(), 4195839: set(), 4195840: set(), 4195845: set(), 4195856: {4195981}, 4195897: set(), 4195904: set(), 4195961: set(), 4195968: set(), 4195999: set(), 4196001: set(), 4196016: set(), 4196022: {4197558}, 4196209: {4197076}, 4196441: {4197115}, 4196937: {7340032}, 4197573: set(), 4197584: {7340032}, 4197685: set(), 4197696: {7340032}, 4197704: set(), 7340032: {4195792}, 7340040: set(), 7340048: set(), 7340056: set(), 7340064: set(), 7340072: set(), 8392784: set(), 8392792: {4197648, 4195684}},
        'function_hashes': {4195664: 2160995126933485085, 4195696: 2163923611709850498, 4195708: 230877880744299600, 4195712: 2129857699332659036, 4195728: 1143045107736432371, 4195744: 1892387988510338414, 4195760: 976792914850716368, 4195776: 1398290322428979371, 4195792: 466102765393377934, 4195838: 1854748152323779301, 4195839: 1016891223247087265, 4195840: 27301026416917143, 4195845: 6889899995484940, 4195856: 1882895690983952382, 4195897: 1625410445650066448, 4195904: 961159074183556238, 4195961: 1658653837155300005, 4195968: 2149315199352528419, 4195999: 919839657399786341, 4196001: 1953663383943956501, 4196016: 2062754443646300329, 4196022: 1613948450472063327, 4196209: 2111862799725097010, 4196441: 923916102771861381, 4196937: 1507274468233787175, 4197573: 2186484097981916406, 4197584: 1336482863812718880, 4197685: 744335668214826006, 4197696: 52822271205752426, 4197704: 1979623380358093506, 7340032: 711019417788810660, 7340040: 1649690089919875361, 7340048: 689009185520196943, 7340056: 548676069374141313, 7340064: 197542682161233282, 7340072: 523019961456155531, 8392784: 1462142678643043579, 8392792: 2287933052588060455},
        'block_hashes': {4195664: 1062147447358754991, 4195684: 16913862596844181, 4195686: 356926153951250000, 4195696: 1532332052362679182, 4195708: 938980197669344835, 4195712: 1029049009690557779, 4195728: 852981731666002112, 4195744: 1922654529359698624, 4195760: 164937244786750440, 4195776: 1010818990484807965, 4195792: 1211597238194359108, 4195838: 1523449600097623170, 4195839: 848148685060457159, 4195840: 2050685517686135631, 4195845: 470113907578214219, 4195856: 1970808386170850639, 4195875: 1927179140948343306, 4195887: 2260526745684249278, 4195889: 571699871676338460, 4195896: 2074956984361735926, 4195897: 503460498619501803, 4195904: 118936980985038764, 4195940: 1414983352550209804, 4195952: 745992981794932260, 4195954: 697661527614581447, 4195960: 991824814932851764, 4195961: 1208324582096604077, 4195968: 2188611590987369275, 4195981: 646192208620703820, 4195990: 2236934074019218843, 4195999: 1793404573927060582, 4196000: 1737622900129786966, 4196001: 593838635142301715, 4196016: 643207840853972878, 4196022: 675469514788706720, 4196041: 1033000429920910650, 4196046: 575545792433991159, 4196062: 688363223192620856, 4196164: 25234203788517099, 4196183: 369066180901261157, 4196192: 847552345296835905, 4196197: 175459198962480872, 4196204: 657688540182482695, 4196207: 1120008836543990390, 4196209: 1639041206513733034, 4196231: 177463421326569139, 4196240: 1756039883886478274, 4196250: 2050412169842289922, 4196254: 756223457670923179, 4196260: 2034493221650933094, 4196270: 160880769534427300, 4196279: 709959523403226275, 4196289: 1537713322486905068, 4196298: 740000744901537501, 4196340: 1899106506941327857, 4196347: 597197405863448486, 4196352: 443999473442774035, 4196359: 1192149056279004234, 4196363: 1045137575408523033, 4196369: 1254583303648340389, 4196379: 429714355664904307, 4196389: 1808576220801956620, 4196393: 666281466260494683, 4196399: 1139854607986063489, 4196408: 2288734287033420346, 4196418: 1316869735350505550, 4196422: 794097444602773465, 4196428: 1333440131319459220, 4196438: 1766783682068958851, 4196441: 1766396973981749036, 4196468: 381736366189964585, 4196474: 1529256856748184621, 4196520: 1036562781401414361, 4196524: 509682986389842991, 4196530: 1670802051122808690, 4196573: 2043957295063214178, 4196577: 2176111969907506618, 4196583: 1290526680201123961, 4196589: 1276238641848388102, 4196635: 2305561297077747917, 4196639: 1793399223262521414, 4196645: 74765071564679529, 4196690: 2107622626086053383, 4196694: 698018199669816457, 4196700: 54259528660022726, 4196745: 2140364812293142069, 4196749: 2223061821699534582, 4196755: 464117375421202271, 4196761: 262463840690799878, 4196809: 465848286413664049, 4196813: 1158097438258622637, 4196819: 417920535158878912, 4196864: 708293419221927479, 4196868: 1928497003265124819, 4196874: 1492656115468754827, 4196880: 1516216161156065678, 4196928: 1600839077271468249, 4196932: 1014017063799881923, 4196937: 938860681075644808, 4196958: 948449379409883715, 4196967: 1802542816643739641, 4196976: 2147946364201262062, 4196981: 868587694998941252, 4197060: 399268734092033806, 4197066: 735236190376153115, 4197070: 399739659049138684, 4197076: 1514924096305293300, 4197091: 2180535862800043743, 4197103: 945188029011250889, 4197115: 926085648367822660, 4197138: 2044163257275392841, 4197185: 1711988716413084656, 4197191: 207991567730628682, 4197197: 864837686837108686, 4197242: 265913026178058618, 4197286: 670434368961384126, 4197292: 2090212456795479872, 4197334: 1878763590638893521, 4197413: 1613811116658947639, 4197417: 219987789537698202, 4197427: 1513609682217062208, 4197431: 772098826313660966, 4197441: 743444531225866234, 4197450: 75046107746219920, 4197459: 207692009546938654, 4197542: 1621976321645921887, 4197548: 744840713864783058, 4197552: 1694112318821912633, 4197558: 602542584694911554, 4197568: 258029163147404887, 4197573: 13316294786040203, 4197584: 2284215735838064091, 4197633: 288387514071178715, 4197639: 1515653302641637487, 4197648: 388129023282923677, 4197661: 1047218839571650461, 4197670: 405857084312441589, 4197685: 513472754198029172, 4197696: 2201050875224076548, 4197704: 325827652284708188, 7340032: 1759534603286467455, 7340040: 739154102070330226, 7340048: 467203206448029206, 7340056: 1063281793463299426, 7340064: 1185814405628052488, 7340072: 680443718829242946, 8392784: 2305493182983790158, 8392792: 1938495500537871973},
        'cfg_hash': 357272754354666949,
        'memcfg_hashes': {'base_norm-op': 2229598305193037971, 'base_norm-inst': 1871023009752101522, 'innereye-op': 2279097487526910978, 'innereye-inst': 499822139002604212, 'safe-op': 2065575377452871639, 'safe-inst': 63045045601888504, 'deepbindiff-op': 1182641413053122541, 'deepbindiff-inst': 484559110202926017, 'deepsemantic-op': 1110481179855274214, 'deepsemantic-inst': 1976766608904021484, 'compressed_stats-op': 1147144581830338726, 'compressed_stats-inst': 1728894721613007637, 'hpcdata-op': 908371875383517375, 'hpcdata-inst': 2111533707042269123},
        'metadata': {'some': 'cfg-level', 134: ('metadata', True, None), (1, 2, 3): 'apples'},
        'block_metadatas': {4195664: {}, 4195684: {}, 4195686: {}, 4195696: {}, 4195708: {}, 4195712: {}, 4195728: {}, 4195744: {}, 4195760: {}, 4195776: {}, 4195792: {}, 4195838: {}, 4195839: {}, 4195840: {}, 4195845: {}, 4195856: {}, 4195875: {}, 4195887: {}, 4195889: {}, 4195896: {}, 4195897: {}, 4195904: {}, 4195940: {}, 4195952: {}, 4195954: {}, 4195960: {}, 4195961: {}, 4195968: {}, 4195981: {}, 4195990: {}, 4195999: {}, 4196000: {}, 4196001: {}, 4196016: {}, 4196022: {}, 4196041: {}, 4196046: {}, 4196062: {}, 4196164: {}, 4196183: {}, 4196192: {}, 4196197: {}, 4196204: {}, 4196207: {}, 4196209: {}, 4196231: {}, 4196240: {}, 4196250: {}, 4196254: {}, 4196260: {}, 4196270: {}, 4196279: {}, 4196289: {}, 4196298: {}, 4196340: {}, 4196347: {}, 4196352: {}, 4196359: {}, 4196363: {}, 4196369: {}, 4196379: {}, 4196389: {}, 4196393: {}, 4196399: {}, 4196408: {}, 4196418: {}, 4196422: {}, 4196428: {}, 4196438: {}, 4196441: {}, 4196468: {}, 4196474: {}, 4196520: {}, 4196524: {}, 4196530: {}, 4196573: {}, 4196577: {}, 4196583: {}, 4196589: {}, 4196635: {}, 4196639: {}, 4196645: {}, 4196690: {}, 4196694: {}, 4196700: {}, 4196745: {}, 4196749: {}, 4196755: {}, 4196761: {}, 4196809: {}, 4196813: {}, 4196819: {}, 4196864: {}, 4196868: {}, 4196874: {}, 4196880: {}, 4196928: {}, 4196932: {}, 4196937: {}, 4196958: {}, 4196967: {}, 4196976: {}, 4196981: {}, 4197060: {}, 4197066: {}, 4197070: {}, 4197076: {}, 4197091: {}, 4197103: {}, 4197115: {}, 4197138: {}, 4197185: {}, 4197191: {}, 4197197: {}, 4197242: {}, 4197286: {}, 4197292: {}, 4197334: {}, 4197413: {}, 4197417: {}, 4197427: {}, 4197431: {}, 4197441: {}, 4197450: {}, 4197459: {}, 4197542: {}, 4197548: {}, 4197552: {}, 4197558: {}, 4197568: {}, 4197573: {}, 4197584: {}, 4197633: {}, 4197639: {}, 4197648: {}, 4197661: {}, 4197670: {}, 4197685: {}, 4197696: {}, 4197704: {}, 7340032: {}, 7340040: {}, 7340048: {}, 7340056: {}, 7340064: {}, 7340072: {}, 8392784: {}, 8392792: {}},
        'function_metadatas': {4195664: {}, 4195696: {}, 4195708: {}, 4195712: {}, 4195728: {}, 4195744: {}, 4195760: {}, 4195776: {}, 4195792: {}, 4195838: {}, 4195839: {}, 4195840: {}, 4195845: {}, 4195856: {}, 4195897: {}, 4195904: {}, 4195961: {}, 4195968: {}, 4195999: {}, 4196001: {}, 4196016: {}, 4196022: {}, 4196209: {}, 4196441: {}, 4196937: {}, 4197573: {}, 4197584: {}, 4197685: {}, 4197696: {}, 4197704: {}, 7340032: {}, 7340040: {}, 7340048: {}, 7340056: {}, 7340064: {}, 7340072: {}, 8392784: {}, 8392792: {}},
        'asm_counts_per_block': {
            4195664: {'0x400550:\tendbr64\t': 1, '0x400554:\tsub\trsp, 8': 1, '0x400558:\tmov\trax, qword ptr [rip + 0x201a91]': 1, '0x40055f:\ttest\trax, rax': 1, '0x400562:\tje\t0x400566': 1},
            4195684: {'0x400564:\tcall\trax': 1},
            4195686: {'0x400566:\tadd\trsp, 8': 1, '0x40056a:\tret\t': 1},
            4195696: {'0x400570:\tpush\tqword ptr [rip + 0x201a92]': 1, '0x400576:\tjmp\tqword ptr [rip + 0x201a94]': 1},
            4195708: {'0x40057c:\tnop\tdword ptr [rax]': 1},
            4195712: {'0x400580:\tjmp\tqword ptr [rip + 0x201a92]': 1},
            4195728: {'0x400590:\tjmp\tqword ptr [rip + 0x201a8a]': 1},
            4195744: {'0x4005a0:\tjmp\tqword ptr [rip + 0x201a82]': 1},
            4195760: {'0x4005b0:\tjmp\tqword ptr [rip + 0x201a7a]': 1},
            4195776: {'0x4005c0:\tjmp\tqword ptr [rip + 0x201a72]': 1},
            4195792: {'0x4005d0:\tendbr64\t': 1, '0x4005d4:\txor\tebp, ebp': 1, '0x4005d6:\tmov\tr9, rdx': 1, '0x4005d9:\tpop\trsi': 1, '0x4005da:\tmov\trdx, rsp': 1, '0x4005dd:\tand\trsp, 0xfffffffffffffff0': 1, '0x4005e1:\tpush\trax': 1, '0x4005e2:\tpush\trsp': 1, '0x4005e3:\tmov\tr8, 0x400d40': 1, '0x4005ea:\tmov\trcx, 0x400cd0': 1, '0x4005f1:\tmov\trdi, 0x400a49': 1, '0x4005f8:\tcall\tqword ptr [rip + 0x2019ea]': 1},
            4195838: {'0x4005fe:\thlt\t': 1},
            4195839: {'0x4005ff:\tnop\t': 1},
            4195840: {'0x400600:\tendbr64\t': 1, '0x400604:\tret\t': 1},
            4195845: {'0x400605:\tnop\tword ptr cs:[rax + rax]': 1, '0x40060f:\tnop\t': 1},
            4195856: {'0x400610:\tlea\trdi, [rip + 0x201a31]': 1, '0x400617:\tlea\trax, [rip + 0x201a2a]': 1, '0x40061e:\tcmp\trax, rdi': 1, '0x400621:\tje\t0x400638': 1},
            4195875: {'0x400623:\tmov\trax, qword ptr [rip + 0x2019b6]': 1, '0x40062a:\ttest\trax, rax': 1, '0x40062d:\tje\t0x400638': 1},
            4195887: {'0x40062f:\tjmp\trax': 1},
            4195889: {'0x400631:\tnop\tdword ptr [rax]': 1},
            4195896: {'0x400638:\tret\t': 1},
            4195897: {'0x400639:\tnop\tdword ptr [rax]': 1},
            4195904: {'0x400640:\tlea\trdi, [rip + 0x201a01]': 1, '0x400647:\tlea\trsi, [rip + 0x2019fa]': 1, '0x40064e:\tsub\trsi, rdi': 1, '0x400651:\tsar\trsi, 3': 1, '0x400655:\tmov\trax, rsi': 1, '0x400658:\tshr\trax, 0x3f': 1, '0x40065c:\tadd\trsi, rax': 1, '0x40065f:\tsar\trsi, 1': 1, '0x400662:\tje\t0x400678': 1},
            4195940: {'0x400664:\tmov\trax, qword ptr [rip + 0x20198d]': 1, '0x40066b:\ttest\trax, rax': 1, '0x40066e:\tje\t0x400678': 1},
            4195952: {'0x400670:\tjmp\trax': 1},
            4195954: {'0x400672:\tnop\tword ptr [rax + rax]': 1},
            4195960: {'0x400678:\tret\t': 1},
            4195961: {'0x400679:\tnop\tdword ptr [rax]': 1},
            4195968: {'0x400680:\tendbr64\t': 1, '0x400684:\tcmp\tbyte ptr [rip + 0x2019b9], 0': 1, '0x40068b:\tjne\t0x4006a0': 1},
            4195981: {'0x40068d:\tpush\trbp': 1, '0x40068e:\tmov\trbp, rsp': 1, '0x400691:\tcall\t0x400610': 1},
            4195990: {'0x400696:\tmov\tbyte ptr [rip + 0x2019a7], 1': 1, '0x40069d:\tpop\trbp': 1, '0x40069e:\tret\t': 1},
            4195999: {'0x40069f:\tnop\t': 1},
            4196000: {'0x4006a0:\tret\t': 1},
            4196001: {'0x4006a1:\tnop\tword ptr cs:[rax + rax]': 1, '0x4006ac:\tnop\tdword ptr [rax]': 1},
            4196016: {'0x4006b0:\tendbr64\t': 1, '0x4006b4:\tjmp\t0x400640': 1},
            4196022: {'0x4006b6:\tpush\trbp': 1, '0x4006b7:\tmov\trbp, rsp': 1, '0x4006ba:\tsub\trsp, 0x30': 1, '0x4006be:\tmov\tqword ptr [rbp - 0x28], rdi': 1, '0x4006c2:\tcmp\tqword ptr [rbp - 0x28], 0': 1, '0x4006c7:\tjns\t0x4006de': 1},
            4196041: {'0x4006c9:\tcall\t0x400590': 1},
            4196046: {'0x4006ce:\tmov\tdword ptr [rax], 0x16': 1, '0x4006d4:\tmov\teax, 0xffffffff': 1, '0x4006d9:\tjmp\t0x40076f': 1},
            4196062: {'0x4006de:\tmov\trcx, qword ptr [rbp - 0x28]': 1, '0x4006e2:\tmovabs\trdx, 0x20c49ba5e353f7cf': 1, '0x4006ec:\tmov\trax, rcx': 1, '0x4006ef:\timul\trdx': 1, '0x4006f2:\tsar\trdx, 7': 1, '0x4006f6:\tmov\trax, rcx': 1, '0x4006f9:\tsar\trax, 0x3f': 1, '0x4006fd:\tsub\trdx, rax': 1, '0x400700:\tmov\trax, rdx': 1, '0x400703:\tmov\tqword ptr [rbp - 0x20], rax': 1, '0x400707:\tmov\trcx, qword ptr [rbp - 0x28]': 1, '0x40070b:\tmovabs\trdx, 0x20c49ba5e353f7cf': 1, '0x400715:\tmov\trax, rcx': 1, '0x400718:\timul\trdx': 1, '0x40071b:\tsar\trdx, 7': 1, '0x40071f:\tmov\trax, rcx': 1, '0x400722:\tsar\trax, 0x3f': 1, '0x400726:\tsub\trdx, rax': 1, '0x400729:\tmov\trax, rdx': 1, '0x40072c:\timul\trax, rax, 0x3e8': 1, '0x400733:\tsub\trcx, rax': 1, '0x400736:\tmov\trax, rcx': 1, '0x400739:\timul\trax, rax, 0xf4240': 1, '0x400740:\tmov\tqword ptr [rbp - 0x18], rax': 1, '0x400744:\tlea\trdx, [rbp - 0x20]': 1, '0x400748:\tlea\trax, [rbp - 0x20]': 1, '0x40074c:\tmov\trsi, rdx': 1, '0x40074f:\tmov\trdi, rax': 1, '0x400752:\tcall\t0x4005a0': 1},
            4196164: {'0x400744:\tlea\trdx, [rbp - 0x20]': 1, '0x400748:\tlea\trax, [rbp - 0x20]': 1, '0x40074c:\tmov\trsi, rdx': 1, '0x40074f:\tmov\trdi, rax': 1, '0x400752:\tcall\t0x4005a0': 1},
            4196183: {'0x400757:\tmov\tdword ptr [rbp - 4], eax': 1, '0x40075a:\tcmp\tdword ptr [rbp - 4], 0': 1, '0x40075e:\tje\t0x40076c': 1},
            4196192: {'0x400760:\tcall\t0x400590': 1},
            4196197: {'0x400765:\tmov\teax, dword ptr [rax]': 1, '0x400767:\tcmp\teax, 4': 1, '0x40076a:\tje\t0x400744': 1},
            4196204: {'0x40076c:\tmov\teax, dword ptr [rbp - 4]': 1, '0x40076f:\tleave\t': 1, '0x400770:\tret\t': 1},
            4196207: {'0x40076f:\tleave\t': 1, '0x400770:\tret\t': 1},
            4196209: {'0x400771:\tpush\trbp': 1, '0x400772:\tmov\trbp, rsp': 1, '0x400775:\tsub\trsp, 0x20': 1, '0x400779:\tmov\tqword ptr [rbp - 0x18], rdi': 1, '0x40077d:\tmov\tedi, 0xa': 1, '0x400782:\tcall\t0x400580': 1},
            4196231: {'0x400787:\tmov\tdword ptr [rbp - 4], 0': 1, '0x40078e:\tjmp\t0x40079e': 1},
            4196240: {'0x400790:\tmov\tedi, 0x2d': 1, '0x400795:\tcall\t0x400580': 1},
            4196250: {'0x40079a:\tadd\tdword ptr [rbp - 4], 1': 1, '0x40079e:\tcmp\tdword ptr [rbp - 4], 0x1f': 1, '0x4007a2:\tjle\t0x400790': 1},
            4196254: {'0x40079e:\tcmp\tdword ptr [rbp - 4], 0x1f': 1, '0x4007a2:\tjle\t0x400790': 1},
            4196260: {'0x4007a4:\tmov\tedi, 0xa': 1, '0x4007a9:\tcall\t0x400580': 1},
            4196270: {'0x4007ae:\tmov\tdword ptr [rbp - 8], 0': 1, '0x4007b5:\tjmp\t0x400829': 1},
            4196279: {'0x4007b7:\tmov\tedi, 0x7c': 1, '0x4007bc:\tcall\t0x400580': 1},
            4196289: {'0x4007c1:\tmov\tdword ptr [rbp - 0xc], 0': 1, '0x4007c8:\tjmp\t0x40080b': 1},
            4196298: {'0x4007ca:\tmov\teax, dword ptr [rbp - 8]': 1, '0x4007cd:\tmovsxd\trdx, eax': 1, '0x4007d0:\tmov\trax, rdx': 1, '0x4007d3:\tshl\trax, 4': 1, '0x4007d7:\tsub\trax, rdx': 1, '0x4007da:\tadd\trax, rax': 1, '0x4007dd:\tmov\trdx, rax': 1, '0x4007e0:\tmov\trax, qword ptr [rbp - 0x18]': 1, '0x4007e4:\tadd\trdx, rax': 1, '0x4007e7:\tmov\teax, dword ptr [rbp - 0xc]': 1, '0x4007ea:\tcdqe\t': 1, '0x4007ec:\tmovzx\teax, byte ptr [rdx + rax]': 1, '0x4007f0:\ttest\tal, al': 1, '0x4007f2:\tje\t0x4007fb': 1},
            4196340: {'0x4007f4:\tmov\teax, 0x58': 1, '0x4007f9:\tjmp\t0x400800': 1},
            4196347: {'0x4007fb:\tmov\teax, 0x20': 1, '0x400800:\tmov\tedi, eax': 1, '0x400802:\tcall\t0x400580': 1},
            4196352: {'0x400800:\tmov\tedi, eax': 1, '0x400802:\tcall\t0x400580': 1},
            4196359: {'0x400807:\tadd\tdword ptr [rbp - 0xc], 1': 1, '0x40080b:\tcmp\tdword ptr [rbp - 0xc], 0x1d': 1, '0x40080f:\tjle\t0x4007ca': 1},
            4196363: {'0x40080b:\tcmp\tdword ptr [rbp - 0xc], 0x1d': 1, '0x40080f:\tjle\t0x4007ca': 1},
            4196369: {'0x400811:\tmov\tedi, 0x7c': 1, '0x400816:\tcall\t0x400580': 1},
            4196379: {'0x40081b:\tmov\tedi, 0xa': 1, '0x400820:\tcall\t0x400580': 1},
            4196389: {'0x400825:\tadd\tdword ptr [rbp - 8], 1': 1, '0x400829:\tcmp\tdword ptr [rbp - 8], 0xe': 1, '0x40082d:\tjle\t0x4007b7': 1},
            4196393: {'0x400829:\tcmp\tdword ptr [rbp - 8], 0xe': 1, '0x40082d:\tjle\t0x4007b7': 1},
            4196399: {'0x40082f:\tmov\tdword ptr [rbp - 0x10], 0': 1, '0x400836:\tjmp\t0x400846': 1},
            4196408: {'0x400838:\tmov\tedi, 0x2d': 1, '0x40083d:\tcall\t0x400580': 1},
            4196418: {'0x400842:\tadd\tdword ptr [rbp - 0x10], 1': 1, '0x400846:\tcmp\tdword ptr [rbp - 0x10], 0x1f': 1, '0x40084a:\tjle\t0x400838': 1},
            4196422: {'0x400846:\tcmp\tdword ptr [rbp - 0x10], 0x1f': 1, '0x40084a:\tjle\t0x400838': 1},
            4196428: {'0x40084c:\tmov\tedi, 0xa': 1, '0x400851:\tcall\t0x400580': 1},
            4196438: {'0x400856:\tnop\t': 1, '0x400857:\tleave\t': 1, '0x400858:\tret\t': 1},
            4196441: {'0x400859:\tpush\trbp': 1, '0x40085a:\tmov\trbp, rsp': 1, '0x40085d:\tmov\tqword ptr [rbp - 0x18], rdi': 1, '0x400861:\tmov\tdword ptr [rbp - 0x1c], esi': 1, '0x400864:\tmov\tdword ptr [rbp - 0x20], edx': 1, '0x400867:\tmov\tdword ptr [rbp - 4], 0': 1, '0x40086e:\tcmp\tdword ptr [rbp - 0x1c], 0': 1, '0x400872:\tjle\t0x4008ac': 1},
            4196468: {'0x400874:\tcmp\tdword ptr [rbp - 0x20], 0': 1, '0x400878:\tjle\t0x4008ac': 1},
            4196474: {'0x40087a:\tmov\teax, dword ptr [rbp - 0x1c]': 1, '0x40087d:\tmovsxd\trdx, eax': 1, '0x400880:\tmov\trax, rdx': 1, '0x400883:\tshl\trax, 4': 1, '0x400887:\tsub\trax, rdx': 1, '0x40088a:\tadd\trax, rax': 1, '0x40088d:\tlea\trdx, [rax - 0x1e]': 1, '0x400891:\tmov\trax, qword ptr [rbp - 0x18]': 1, '0x400895:\tadd\trdx, rax': 1, '0x400898:\tmov\teax, dword ptr [rbp - 0x20]': 1, '0x40089b:\tsub\teax, 1': 1, '0x40089e:\tcdqe\t': 1, '0x4008a0:\tmovzx\teax, byte ptr [rdx + rax]': 1, '0x4008a4:\ttest\tal, al': 1, '0x4008a6:\tje\t0x4008ac': 1},
            4196520: {'0x4008a8:\tadd\tdword ptr [rbp - 4], 1': 1, '0x4008ac:\tcmp\tdword ptr [rbp - 0x1c], 0': 1, '0x4008b0:\tjle\t0x4008e1': 1},
            4196524: {'0x4008ac:\tcmp\tdword ptr [rbp - 0x1c], 0': 1, '0x4008b0:\tjle\t0x4008e1': 1},
            4196530: {'0x4008b2:\tmov\teax, dword ptr [rbp - 0x1c]': 1, '0x4008b5:\tmovsxd\trdx, eax': 1, '0x4008b8:\tmov\trax, rdx': 1, '0x4008bb:\tshl\trax, 4': 1, '0x4008bf:\tsub\trax, rdx': 1, '0x4008c2:\tadd\trax, rax': 1, '0x4008c5:\tlea\trdx, [rax - 0x1e]': 1, '0x4008c9:\tmov\trax, qword ptr [rbp - 0x18]': 1, '0x4008cd:\tadd\trdx, rax': 1, '0x4008d0:\tmov\teax, dword ptr [rbp - 0x20]': 1, '0x4008d3:\tcdqe\t': 1, '0x4008d5:\tmovzx\teax, byte ptr [rdx + rax]': 1, '0x4008d9:\ttest\tal, al': 1, '0x4008db:\tje\t0x4008e1': 1},
            4196573: {'0x4008dd:\tadd\tdword ptr [rbp - 4], 1': 1, '0x4008e1:\tcmp\tdword ptr [rbp - 0x1c], 0': 1, '0x4008e5:\tjle\t0x40091f': 1},
            4196577: {'0x4008e1:\tcmp\tdword ptr [rbp - 0x1c], 0': 1, '0x4008e5:\tjle\t0x40091f': 1},
            4196583: {'0x4008e7:\tcmp\tdword ptr [rbp - 0x20], 0x1c': 1, '0x4008eb:\tjg\t0x40091f': 1},
            4196589: {'0x4008ed:\tmov\teax, dword ptr [rbp - 0x1c]': 1, '0x4008f0:\tmovsxd\trdx, eax': 1, '0x4008f3:\tmov\trax, rdx': 1, '0x4008f6:\tshl\trax, 4': 1, '0x4008fa:\tsub\trax, rdx': 1, '0x4008fd:\tadd\trax, rax': 1, '0x400900:\tlea\trdx, [rax - 0x1e]': 1, '0x400904:\tmov\trax, qword ptr [rbp - 0x18]': 1, '0x400908:\tadd\trdx, rax': 1, '0x40090b:\tmov\teax, dword ptr [rbp - 0x20]': 1, '0x40090e:\tadd\teax, 1': 1, '0x400911:\tcdqe\t': 1, '0x400913:\tmovzx\teax, byte ptr [rdx + rax]': 1, '0x400917:\ttest\tal, al': 1, '0x400919:\tje\t0x40091f': 1},
            4196635: {'0x40091b:\tadd\tdword ptr [rbp - 4], 1': 1, '0x40091f:\tcmp\tdword ptr [rbp - 0x20], 0': 1, '0x400923:\tjle\t0x400956': 1},
            4196639: {'0x40091f:\tcmp\tdword ptr [rbp - 0x20], 0': 1, '0x400923:\tjle\t0x400956': 1},
            4196645: {'0x400925:\tmov\teax, dword ptr [rbp - 0x1c]': 1, '0x400928:\tmovsxd\trdx, eax': 1, '0x40092b:\tmov\trax, rdx': 1, '0x40092e:\tshl\trax, 4': 1, '0x400932:\tsub\trax, rdx': 1, '0x400935:\tadd\trax, rax': 1, '0x400938:\tmov\trdx, rax': 1, '0x40093b:\tmov\trax, qword ptr [rbp - 0x18]': 1, '0x40093f:\tadd\trdx, rax': 1, '0x400942:\tmov\teax, dword ptr [rbp - 0x20]': 1, '0x400945:\tsub\teax, 1': 1, '0x400948:\tcdqe\t': 1, '0x40094a:\tmovzx\teax, byte ptr [rdx + rax]': 1, '0x40094e:\ttest\tal, al': 1, '0x400950:\tje\t0x400956': 1},
            4196690: {'0x400952:\tadd\tdword ptr [rbp - 4], 1': 1, '0x400956:\tcmp\tdword ptr [rbp - 0x20], 0x1c': 1, '0x40095a:\tjg\t0x40098d': 1},
            4196694: {'0x400956:\tcmp\tdword ptr [rbp - 0x20], 0x1c': 1, '0x40095a:\tjg\t0x40098d': 1},
            4196700: {'0x40095c:\tmov\teax, dword ptr [rbp - 0x1c]': 1, '0x40095f:\tmovsxd\trdx, eax': 1, '0x400962:\tmov\trax, rdx': 1, '0x400965:\tshl\trax, 4': 1, '0x400969:\tsub\trax, rdx': 1, '0x40096c:\tadd\trax, rax': 1, '0x40096f:\tmov\trdx, rax': 1, '0x400972:\tmov\trax, qword ptr [rbp - 0x18]': 1, '0x400976:\tadd\trdx, rax': 1, '0x400979:\tmov\teax, dword ptr [rbp - 0x20]': 1, '0x40097c:\tadd\teax, 1': 1, '0x40097f:\tcdqe\t': 1, '0x400981:\tmovzx\teax, byte ptr [rdx + rax]': 1, '0x400985:\ttest\tal, al': 1, '0x400987:\tje\t0x40098d': 1},
            4196745: {'0x400989:\tadd\tdword ptr [rbp - 4], 1': 1, '0x40098d:\tcmp\tdword ptr [rbp - 0x1c], 0xd': 1, '0x400991:\tjg\t0x4009cd': 1},
            4196749: {'0x40098d:\tcmp\tdword ptr [rbp - 0x1c], 0xd': 1, '0x400991:\tjg\t0x4009cd': 1},
            4196755: {'0x400993:\tcmp\tdword ptr [rbp - 0x20], 0': 1, '0x400997:\tjle\t0x4009cd': 1},
            4196761: {'0x400999:\tmov\teax, dword ptr [rbp - 0x1c]': 1, '0x40099c:\tcdqe\t': 1, '0x40099e:\tlea\trdx, [rax + 1]': 1, '0x4009a2:\tmov\trax, rdx': 1, '0x4009a5:\tshl\trax, 4': 1, '0x4009a9:\tsub\trax, rdx': 1, '0x4009ac:\tadd\trax, rax': 1, '0x4009af:\tmov\trdx, rax': 1, '0x4009b2:\tmov\trax, qword ptr [rbp - 0x18]': 1, '0x4009b6:\tadd\trdx, rax': 1, '0x4009b9:\tmov\teax, dword ptr [rbp - 0x20]': 1, '0x4009bc:\tsub\teax, 1': 1, '0x4009bf:\tcdqe\t': 1, '0x4009c1:\tmovzx\teax, byte ptr [rdx + rax]': 1, '0x4009c5:\ttest\tal, al': 1, '0x4009c7:\tje\t0x4009cd': 1},
            4196809: {'0x4009c9:\tadd\tdword ptr [rbp - 4], 1': 1, '0x4009cd:\tcmp\tdword ptr [rbp - 0x1c], 0xd': 1, '0x4009d1:\tjg\t0x400a04': 1},
            4196813: {'0x4009cd:\tcmp\tdword ptr [rbp - 0x1c], 0xd': 1, '0x4009d1:\tjg\t0x400a04': 1},
            4196819: {'0x4009d3:\tmov\teax, dword ptr [rbp - 0x1c]': 1, '0x4009d6:\tcdqe\t': 1, '0x4009d8:\tlea\trdx, [rax + 1]': 1, '0x4009dc:\tmov\trax, rdx': 1, '0x4009df:\tshl\trax, 4': 1, '0x4009e3:\tsub\trax, rdx': 1, '0x4009e6:\tadd\trax, rax': 1, '0x4009e9:\tmov\trdx, rax': 1, '0x4009ec:\tmov\trax, qword ptr [rbp - 0x18]': 1, '0x4009f0:\tadd\trdx, rax': 1, '0x4009f3:\tmov\teax, dword ptr [rbp - 0x20]': 1, '0x4009f6:\tcdqe\t': 1, '0x4009f8:\tmovzx\teax, byte ptr [rdx + rax]': 1, '0x4009fc:\ttest\tal, al': 1, '0x4009fe:\tje\t0x400a04': 1},
            4196864: {'0x400a00:\tadd\tdword ptr [rbp - 4], 1': 1, '0x400a04:\tcmp\tdword ptr [rbp - 0x1c], 0xd': 1, '0x400a08:\tjg\t0x400a44': 1},
            4196868: {'0x400a04:\tcmp\tdword ptr [rbp - 0x1c], 0xd': 1, '0x400a08:\tjg\t0x400a44': 1},
            4196874: {'0x400a0a:\tcmp\tdword ptr [rbp - 0x20], 0x1c': 1, '0x400a0e:\tjg\t0x400a44': 1},
            4196880: {'0x400a10:\tmov\teax, dword ptr [rbp - 0x1c]': 1, '0x400a13:\tcdqe\t': 1, '0x400a15:\tlea\trdx, [rax + 1]': 1, '0x400a19:\tmov\trax, rdx': 1, '0x400a1c:\tshl\trax, 4': 1, '0x400a20:\tsub\trax, rdx': 1, '0x400a23:\tadd\trax, rax': 1, '0x400a26:\tmov\trdx, rax': 1, '0x400a29:\tmov\trax, qword ptr [rbp - 0x18]': 1, '0x400a2d:\tadd\trdx, rax': 1, '0x400a30:\tmov\teax, dword ptr [rbp - 0x20]': 1, '0x400a33:\tadd\teax, 1': 1, '0x400a36:\tcdqe\t': 1, '0x400a38:\tmovzx\teax, byte ptr [rdx + rax]': 1, '0x400a3c:\ttest\tal, al': 1, '0x400a3e:\tje\t0x400a44': 1},
            4196928: {'0x400a40:\tadd\tdword ptr [rbp - 4], 1': 1, '0x400a44:\tmov\teax, dword ptr [rbp - 4]': 1, '0x400a47:\tpop\trbp': 1, '0x400a48:\tret\t': 1},
            4196932: {'0x400a44:\tmov\teax, dword ptr [rbp - 4]': 1, '0x400a47:\tpop\trbp': 1, '0x400a48:\tret\t': 1},
            4196937: {'0x400a49:\tpush\trbp': 1, '0x400a4a:\tmov\trbp, rsp': 1, '0x400a4d:\tsub\trsp, 0x3b0': 1, '0x400a54:\tmov\tedi, 0x3039': 1, '0x400a59:\tcall\t0x4005b0': 1},
            4196958: {'0x400a5e:\tmov\tdword ptr [rbp - 4], 0': 1, '0x400a65:\tjmp\t0x400ace': 1},
            4196967: {'0x400a67:\tmov\tdword ptr [rbp - 8], 0': 1, '0x400a6e:\tjmp\t0x400ac4': 1},
            4196976: {'0x400a70:\tcall\t0x4005c0': 1},
            4196981: {'0x400a75:\tcvtsi2ss\txmm0, eax': 1, '0x400a79:\tmovss\txmm1, dword ptr [rip + 0x2e7]': 1, '0x400a81:\tdivss\txmm0, xmm1': 1, '0x400a85:\tmovaps\txmm1, xmm0': 1, '0x400a88:\tmovss\txmm0, dword ptr [rip + 0x2dc]': 1, '0x400a90:\tcomiss\txmm0, xmm1': 1, '0x400a93:\tseta\tal': 1, '0x400a96:\tmov\tesi, eax': 1, '0x400a98:\tmov\teax, dword ptr [rbp - 8]': 1, '0x400a9b:\tmovsxd\trcx, eax': 1, '0x400a9e:\tmov\teax, dword ptr [rbp - 4]': 1, '0x400aa1:\tmovsxd\trdx, eax': 1, '0x400aa4:\tmov\trax, rdx': 1, '0x400aa7:\tshl\trax, 4': 1, '0x400aab:\tsub\trax, rdx': 1, '0x400aae:\tadd\trax, rax': 1, '0x400ab1:\tadd\trax, rbp': 1, '0x400ab4:\tadd\trax, rcx': 1, '0x400ab7:\tsub\trax, 0x1e0': 1, '0x400abd:\tmov\tbyte ptr [rax], sil': 1, '0x400ac0:\tadd\tdword ptr [rbp - 8], 1': 1, '0x400ac4:\tcmp\tdword ptr [rbp - 8], 0x1d': 1, '0x400ac8:\tjle\t0x400a70': 1},
            4197060: {'0x400ac4:\tcmp\tdword ptr [rbp - 8], 0x1d': 1, '0x400ac8:\tjle\t0x400a70': 1},
            4197066: {'0x400aca:\tadd\tdword ptr [rbp - 4], 1': 1, '0x400ace:\tcmp\tdword ptr [rbp - 4], 0xe': 1, '0x400ad2:\tjle\t0x400a67': 1},
            4197070: {'0x400ace:\tcmp\tdword ptr [rbp - 4], 0xe': 1, '0x400ad2:\tjle\t0x400a67': 1},
            4197076: {'0x400ad4:\tlea\trax, [rbp - 0x1e0]': 1, '0x400adb:\tmov\trdi, rax': 1, '0x400ade:\tcall\t0x400771': 1},
            4197091: {'0x400ae3:\tmov\tdword ptr [rbp - 0xc], 0': 1, '0x400aea:\tjmp\t0x400c37': 1},
            4197103: {'0x400aef:\tmov\tdword ptr [rbp - 0x10], 0': 1, '0x400af6:\tjmp\t0x400c29': 1},
            4197115: {'0x400afb:\tmov\tedx, dword ptr [rbp - 0x10]': 1, '0x400afe:\tmov\tecx, dword ptr [rbp - 0xc]': 1, '0x400b01:\tlea\trax, [rbp - 0x1e0]': 1, '0x400b08:\tmov\tesi, ecx': 1, '0x400b0a:\tmov\trdi, rax': 1, '0x400b0d:\tcall\t0x400859': 1},
            4197138: {'0x400b12:\tmov\tdword ptr [rbp - 0x1c], eax': 1, '0x400b15:\tmov\teax, dword ptr [rbp - 0x10]': 1, '0x400b18:\tmovsxd\trcx, eax': 1, '0x400b1b:\tmov\teax, dword ptr [rbp - 0xc]': 1, '0x400b1e:\tmovsxd\trdx, eax': 1, '0x400b21:\tmov\trax, rdx': 1, '0x400b24:\tshl\trax, 4': 1, '0x400b28:\tsub\trax, rdx': 1, '0x400b2b:\tadd\trax, rax': 1, '0x400b2e:\tadd\trax, rbp': 1, '0x400b31:\tadd\trax, rcx': 1, '0x400b34:\tsub\trax, 0x1e0': 1, '0x400b3a:\tmovzx\teax, byte ptr [rax]': 1, '0x400b3d:\ttest\tal, al': 1, '0x400b3f:\tje\t0x400b7a': 1},
            4197185: {'0x400b41:\tcmp\tdword ptr [rbp - 0x1c], 1': 1, '0x400b45:\tjle\t0x400b4d': 1},
            4197191: {'0x400b47:\tcmp\tdword ptr [rbp - 0x1c], 3': 1, '0x400b4b:\tjle\t0x400b7a': 1},
            4197197: {'0x400b4d:\tmov\teax, dword ptr [rbp - 0x10]': 1, '0x400b50:\tmovsxd\trcx, eax': 1, '0x400b53:\tmov\teax, dword ptr [rbp - 0xc]': 1, '0x400b56:\tmovsxd\trdx, eax': 1, '0x400b59:\tmov\trax, rdx': 1, '0x400b5c:\tshl\trax, 4': 1, '0x400b60:\tsub\trax, rdx': 1, '0x400b63:\tadd\trax, rax': 1, '0x400b66:\tadd\trax, rbp': 1, '0x400b69:\tadd\trax, rcx': 1, '0x400b6c:\tsub\trax, 0x3b0': 1, '0x400b72:\tmov\tbyte ptr [rax], 0': 1, '0x400b75:\tjmp\t0x400c25': 1},
            4197242: {'0x400b7a:\tmov\teax, dword ptr [rbp - 0x10]': 1, '0x400b7d:\tmovsxd\trcx, eax': 1, '0x400b80:\tmov\teax, dword ptr [rbp - 0xc]': 1, '0x400b83:\tmovsxd\trdx, eax': 1, '0x400b86:\tmov\trax, rdx': 1, '0x400b89:\tshl\trax, 4': 1, '0x400b8d:\tsub\trax, rdx': 1, '0x400b90:\tadd\trax, rax': 1, '0x400b93:\tadd\trax, rbp': 1, '0x400b96:\tadd\trax, rcx': 1, '0x400b99:\tsub\trax, 0x1e0': 1, '0x400b9f:\tmovzx\teax, byte ptr [rax]': 1, '0x400ba2:\ttest\tal, al': 1, '0x400ba4:\tjne\t0x400bd6': 1},
            4197286: {'0x400ba6:\tcmp\tdword ptr [rbp - 0x1c], 3': 1, '0x400baa:\tjne\t0x400bd6': 1},
            4197292: {'0x400bac:\tmov\teax, dword ptr [rbp - 0x10]': 1, '0x400baf:\tmovsxd\trcx, eax': 1, '0x400bb2:\tmov\teax, dword ptr [rbp - 0xc]': 1, '0x400bb5:\tmovsxd\trdx, eax': 1, '0x400bb8:\tmov\trax, rdx': 1, '0x400bbb:\tshl\trax, 4': 1, '0x400bbf:\tsub\trax, rdx': 1, '0x400bc2:\tadd\trax, rax': 1, '0x400bc5:\tadd\trax, rbp': 1, '0x400bc8:\tadd\trax, rcx': 1, '0x400bcb:\tsub\trax, 0x3b0': 1, '0x400bd1:\tmov\tbyte ptr [rax], 1': 1, '0x400bd4:\tjmp\t0x400c25': 1},
            4197334: {'0x400bd6:\tmov\teax, dword ptr [rbp - 0x10]': 1, '0x400bd9:\tmovsxd\trcx, eax': 1, '0x400bdc:\tmov\teax, dword ptr [rbp - 0xc]': 1, '0x400bdf:\tmovsxd\trdx, eax': 1, '0x400be2:\tmov\trax, rdx': 1, '0x400be5:\tshl\trax, 4': 1, '0x400be9:\tsub\trax, rdx': 1, '0x400bec:\tadd\trax, rax': 1, '0x400bef:\tadd\trax, rbp': 1, '0x400bf2:\tadd\trax, rcx': 1, '0x400bf5:\tsub\trax, 0x1e0': 1, '0x400bfb:\tmovzx\tecx, byte ptr [rax]': 1, '0x400bfe:\tmov\teax, dword ptr [rbp - 0x10]': 1, '0x400c01:\tmovsxd\trsi, eax': 1, '0x400c04:\tmov\teax, dword ptr [rbp - 0xc]': 1, '0x400c07:\tmovsxd\trdx, eax': 1, '0x400c0a:\tmov\trax, rdx': 1, '0x400c0d:\tshl\trax, 4': 1, '0x400c11:\tsub\trax, rdx': 1, '0x400c14:\tadd\trax, rax': 1, '0x400c17:\tadd\trax, rbp': 1, '0x400c1a:\tadd\trax, rsi': 1, '0x400c1d:\tsub\trax, 0x3b0': 1, '0x400c23:\tmov\tbyte ptr [rax], cl': 1, '0x400c25:\tadd\tdword ptr [rbp - 0x10], 1': 1, '0x400c29:\tcmp\tdword ptr [rbp - 0x10], 0x1d': 1, '0x400c2d:\tjle\t0x400afb': 1},
            4197413: {'0x400c25:\tadd\tdword ptr [rbp - 0x10], 1': 1, '0x400c29:\tcmp\tdword ptr [rbp - 0x10], 0x1d': 1, '0x400c2d:\tjle\t0x400afb': 1},
            4197417: {'0x400c29:\tcmp\tdword ptr [rbp - 0x10], 0x1d': 1, '0x400c2d:\tjle\t0x400afb': 1},
            4197427: {'0x400c33:\tadd\tdword ptr [rbp - 0xc], 1': 1, '0x400c37:\tcmp\tdword ptr [rbp - 0xc], 0xe': 1, '0x400c3b:\tjle\t0x400aef': 1},
            4197431: {'0x400c37:\tcmp\tdword ptr [rbp - 0xc], 0xe': 1, '0x400c3b:\tjle\t0x400aef': 1},
            4197441: {'0x400c41:\tmov\tdword ptr [rbp - 0x14], 0': 1, '0x400c48:\tjmp\t0x400cb0': 1},
            4197450: {'0x400c4a:\tmov\tdword ptr [rbp - 0x18], 0': 1, '0x400c51:\tjmp\t0x400ca6': 1},
            4197459: {'0x400c53:\tmov\teax, dword ptr [rbp - 0x18]': 1, '0x400c56:\tmovsxd\trcx, eax': 1, '0x400c59:\tmov\teax, dword ptr [rbp - 0x14]': 1, '0x400c5c:\tmovsxd\trdx, eax': 1, '0x400c5f:\tmov\trax, rdx': 1, '0x400c62:\tshl\trax, 4': 1, '0x400c66:\tsub\trax, rdx': 1, '0x400c69:\tadd\trax, rax': 1, '0x400c6c:\tadd\trax, rbp': 1, '0x400c6f:\tadd\trax, rcx': 1, '0x400c72:\tsub\trax, 0x3b0': 1, '0x400c78:\tmovzx\tecx, byte ptr [rax]': 1, '0x400c7b:\tmov\teax, dword ptr [rbp - 0x18]': 1, '0x400c7e:\tmovsxd\trsi, eax': 1, '0x400c81:\tmov\teax, dword ptr [rbp - 0x14]': 1, '0x400c84:\tmovsxd\trdx, eax': 1, '0x400c87:\tmov\trax, rdx': 1, '0x400c8a:\tshl\trax, 4': 1, '0x400c8e:\tsub\trax, rdx': 1, '0x400c91:\tadd\trax, rax': 1, '0x400c94:\tadd\trax, rbp': 1, '0x400c97:\tadd\trax, rsi': 1, '0x400c9a:\tsub\trax, 0x1e0': 1, '0x400ca0:\tmov\tbyte ptr [rax], cl': 1, '0x400ca2:\tadd\tdword ptr [rbp - 0x18], 1': 1, '0x400ca6:\tcmp\tdword ptr [rbp - 0x18], 0x1d': 1, '0x400caa:\tjle\t0x400c53': 1},
            4197542: {'0x400ca6:\tcmp\tdword ptr [rbp - 0x18], 0x1d': 1, '0x400caa:\tjle\t0x400c53': 1},
            4197548: {'0x400cac:\tadd\tdword ptr [rbp - 0x14], 1': 1, '0x400cb0:\tcmp\tdword ptr [rbp - 0x14], 0xe': 1, '0x400cb4:\tjle\t0x400c4a': 1},
            4197552: {'0x400cb0:\tcmp\tdword ptr [rbp - 0x14], 0xe': 1, '0x400cb4:\tjle\t0x400c4a': 1},
            4197558: {'0x400cb6:\tmov\tedi, 0x1f4': 1, '0x400cbb:\tcall\t0x4006b6': 1},
            4197568: {'0x400cc0:\tjmp\t0x400ad4': 1},
            4197573: {'0x400cc5:\tnop\tword ptr cs:[rax + rax]': 1, '0x400ccf:\tnop\t': 1},
            4197584: {'0x400cd0:\tendbr64\t': 1, '0x400cd4:\tpush\tr15': 1, '0x400cd6:\tmov\tr15, rdx': 1, '0x400cd9:\tpush\tr14': 1, '0x400cdb:\tmov\tr14, rsi': 1, '0x400cde:\tpush\tr13': 1, '0x400ce0:\tmov\tr13d, edi': 1, '0x400ce3:\tpush\tr12': 1, '0x400ce5:\tlea\tr12, [rip + 0x201114]': 1, '0x400cec:\tpush\trbp': 1, '0x400ced:\tlea\trbp, [rip + 0x201114]': 1, '0x400cf4:\tpush\trbx': 1, '0x400cf5:\tsub\trbp, r12': 1, '0x400cf8:\tsub\trsp, 8': 1, '0x400cfc:\tcall\t0x400550': 1},
            4197633: {'0x400d01:\tsar\trbp, 3': 1, '0x400d05:\tje\t0x400d26': 1},
            4197639: {'0x400d07:\txor\tebx, ebx': 1, '0x400d09:\tnop\tdword ptr [rax]': 1, '0x400d10:\tmov\trdx, r15': 1, '0x400d13:\tmov\trsi, r14': 1, '0x400d16:\tmov\tedi, r13d': 1, '0x400d19:\tcall\tqword ptr [r12 + rbx*8]': 1},
            4197648: {'0x400d10:\tmov\trdx, r15': 1, '0x400d13:\tmov\trsi, r14': 1, '0x400d16:\tmov\tedi, r13d': 1, '0x400d19:\tcall\tqword ptr [r12 + rbx*8]': 1},
            4197661: {'0x400d1d:\tadd\trbx, 1': 1, '0x400d21:\tcmp\trbp, rbx': 1, '0x400d24:\tjne\t0x400d10': 1},
            4197670: {'0x400d26:\tadd\trsp, 8': 1, '0x400d2a:\tpop\trbx': 1, '0x400d2b:\tpop\trbp': 1, '0x400d2c:\tpop\tr12': 1, '0x400d2e:\tpop\tr13': 1, '0x400d30:\tpop\tr14': 1, '0x400d32:\tpop\tr15': 1, '0x400d34:\tret\t': 1},
            4197685: {'0x400d35:\tnop\tword ptr cs:[rax + rax]': 1},
            4197696: {'0x400d40:\tendbr64\t': 1, '0x400d44:\tret\t': 1},
            4197704: {'0x400d48:\tendbr64\t': 1, '0x400d4c:\tsub\trsp, 8': 1, '0x400d50:\tadd\trsp, 8': 1, '0x400d54:\tret\t': 1},
            7340032: {},
            7340040: {},
            7340048: {},
            7340056: {},
            7340064: {},
            7340072: {},
            8392784: {},
            8392792: {},
        },
        'asm_counts_per_function': {
            4195664: {
                '0x400550:\tendbr64\t': 1,
                '0x400554:\tsub\trsp, 8': 1,
                '0x400558:\tmov\trax, qword ptr [rip + 0x201a91]': 1,
                '0x40055f:\ttest\trax, rax': 1,
                '0x400562:\tje\t0x400566': 1,
                '0x400566:\tadd\trsp, 8': 1,
                '0x40056a:\tret\t': 1,
                '0x400564:\tcall\trax': 1,
            },
            4195696: {
                '0x400570:\tpush\tqword ptr [rip + 0x201a92]': 1,
                '0x400576:\tjmp\tqword ptr [rip + 0x201a94]': 1,
            },
            4195708: {
                '0x40057c:\tnop\tdword ptr [rax]': 1,
            },
            4195712: {
                '0x400580:\tjmp\tqword ptr [rip + 0x201a92]': 1,
            },
            4195728: {
                '0x400590:\tjmp\tqword ptr [rip + 0x201a8a]': 1,
            },
            4195744: {
                '0x4005a0:\tjmp\tqword ptr [rip + 0x201a82]': 1,
            },
            4195760: {
                '0x4005b0:\tjmp\tqword ptr [rip + 0x201a7a]': 1,
            },
            4195776: {
                '0x4005c0:\tjmp\tqword ptr [rip + 0x201a72]': 1,
            },
            4195792: {
                '0x4005d0:\tendbr64\t': 1,
                '0x4005d4:\txor\tebp, ebp': 1,
                '0x4005d6:\tmov\tr9, rdx': 1,
                '0x4005d9:\tpop\trsi': 1,
                '0x4005da:\tmov\trdx, rsp': 1,
                '0x4005dd:\tand\trsp, 0xfffffffffffffff0': 1,
                '0x4005e1:\tpush\trax': 1,
                '0x4005e2:\tpush\trsp': 1,
                '0x4005e3:\tmov\tr8, 0x400d40': 1,
                '0x4005ea:\tmov\trcx, 0x400cd0': 1,
                '0x4005f1:\tmov\trdi, 0x400a49': 1,
                '0x4005f8:\tcall\tqword ptr [rip + 0x2019ea]': 1,
            },
            4195838: {
                '0x4005fe:\thlt\t': 1,
            },
            4195839: {
                '0x4005ff:\tnop\t': 1,
            },
            4195840: {
                '0x400600:\tendbr64\t': 1,
                '0x400604:\tret\t': 1,
            },
            4195845: {
                '0x400605:\tnop\tword ptr cs:[rax + rax]': 1,
                '0x40060f:\tnop\t': 1,
            },
            4195856: {
                '0x400610:\tlea\trdi, [rip + 0x201a31]': 1,
                '0x400617:\tlea\trax, [rip + 0x201a2a]': 1,
                '0x40061e:\tcmp\trax, rdi': 1,
                '0x400621:\tje\t0x400638': 1,
                '0x400638:\tret\t': 1,
                '0x400623:\tmov\trax, qword ptr [rip + 0x2019b6]': 1,
                '0x40062a:\ttest\trax, rax': 1,
                '0x40062d:\tje\t0x400638': 1,
                '0x40062f:\tjmp\trax': 1,
                '0x400631:\tnop\tdword ptr [rax]': 1,
            },
            4195897: {
                '0x400639:\tnop\tdword ptr [rax]': 1,
            },
            4195904: {
                '0x400640:\tlea\trdi, [rip + 0x201a01]': 1,
                '0x400647:\tlea\trsi, [rip + 0x2019fa]': 1,
                '0x40064e:\tsub\trsi, rdi': 1,
                '0x400651:\tsar\trsi, 3': 1,
                '0x400655:\tmov\trax, rsi': 1,
                '0x400658:\tshr\trax, 0x3f': 1,
                '0x40065c:\tadd\trsi, rax': 1,
                '0x40065f:\tsar\trsi, 1': 1,
                '0x400662:\tje\t0x400678': 1,
                '0x400678:\tret\t': 1,
                '0x400664:\tmov\trax, qword ptr [rip + 0x20198d]': 1,
                '0x40066b:\ttest\trax, rax': 1,
                '0x40066e:\tje\t0x400678': 1,
                '0x400670:\tjmp\trax': 1,
                '0x400672:\tnop\tword ptr [rax + rax]': 1,
            },
            4195961: {
                '0x400679:\tnop\tdword ptr [rax]': 1,
            },
            4195968: {
                '0x400680:\tendbr64\t': 1,
                '0x400684:\tcmp\tbyte ptr [rip + 0x2019b9], 0': 1,
                '0x40068b:\tjne\t0x4006a0': 1,
                '0x40068d:\tpush\trbp': 1,
                '0x40068e:\tmov\trbp, rsp': 1,
                '0x400691:\tcall\t0x400610': 1,
                '0x4006a0:\tret\t': 1,
                '0x400696:\tmov\tbyte ptr [rip + 0x2019a7], 1': 1,
                '0x40069d:\tpop\trbp': 1,
                '0x40069e:\tret\t': 1,
            },
            4195999: {
                '0x40069f:\tnop\t': 1,
            },
            4196001: {
                '0x4006a1:\tnop\tword ptr cs:[rax + rax]': 1,
                '0x4006ac:\tnop\tdword ptr [rax]': 1,
            },
            4196016: {
                '0x4006b0:\tendbr64\t': 1,
                '0x4006b4:\tjmp\t0x400640': 1,
            },
            4196022: {
                '0x4006b6:\tpush\trbp': 1,
                '0x4006b7:\tmov\trbp, rsp': 1,
                '0x4006ba:\tsub\trsp, 0x30': 1,
                '0x4006be:\tmov\tqword ptr [rbp - 0x28], rdi': 1,
                '0x4006c2:\tcmp\tqword ptr [rbp - 0x28], 0': 1,
                '0x4006c7:\tjns\t0x4006de': 1,
                '0x4006c9:\tcall\t0x400590': 1,
                '0x4006de:\tmov\trcx, qword ptr [rbp - 0x28]': 1,
                '0x4006e2:\tmovabs\trdx, 0x20c49ba5e353f7cf': 1,
                '0x4006ec:\tmov\trax, rcx': 1,
                '0x4006ef:\timul\trdx': 1,
                '0x4006f2:\tsar\trdx, 7': 1,
                '0x4006f6:\tmov\trax, rcx': 1,
                '0x4006f9:\tsar\trax, 0x3f': 1,
                '0x4006fd:\tsub\trdx, rax': 1,
                '0x400700:\tmov\trax, rdx': 1,
                '0x400703:\tmov\tqword ptr [rbp - 0x20], rax': 1,
                '0x400707:\tmov\trcx, qword ptr [rbp - 0x28]': 1,
                '0x40070b:\tmovabs\trdx, 0x20c49ba5e353f7cf': 1,
                '0x400715:\tmov\trax, rcx': 1,
                '0x400718:\timul\trdx': 1,
                '0x40071b:\tsar\trdx, 7': 1,
                '0x40071f:\tmov\trax, rcx': 1,
                '0x400722:\tsar\trax, 0x3f': 1,
                '0x400726:\tsub\trdx, rax': 1,
                '0x400729:\tmov\trax, rdx': 1,
                '0x40072c:\timul\trax, rax, 0x3e8': 1,
                '0x400733:\tsub\trcx, rax': 1,
                '0x400736:\tmov\trax, rcx': 1,
                '0x400739:\timul\trax, rax, 0xf4240': 1,
                '0x400740:\tmov\tqword ptr [rbp - 0x18], rax': 1,
                '0x400744:\tlea\trdx, [rbp - 0x20]': 2,
                '0x400748:\tlea\trax, [rbp - 0x20]': 2,
                '0x40074c:\tmov\trsi, rdx': 2,
                '0x40074f:\tmov\trdi, rax': 2,
                '0x400752:\tcall\t0x4005a0': 2,
                '0x4006ce:\tmov\tdword ptr [rax], 0x16': 1,
                '0x4006d4:\tmov\teax, 0xffffffff': 1,
                '0x4006d9:\tjmp\t0x40076f': 1,
                '0x400757:\tmov\tdword ptr [rbp - 4], eax': 1,
                '0x40075a:\tcmp\tdword ptr [rbp - 4], 0': 1,
                '0x40075e:\tje\t0x40076c': 1,
                '0x40076f:\tleave\t': 2,
                '0x400770:\tret\t': 2,
                '0x40076c:\tmov\teax, dword ptr [rbp - 4]': 1,
                '0x400760:\tcall\t0x400590': 1,
                '0x400765:\tmov\teax, dword ptr [rax]': 1,
                '0x400767:\tcmp\teax, 4': 1,
                '0x40076a:\tje\t0x400744': 1,
            },
            4196209: {
                '0x400771:\tpush\trbp': 1,
                '0x400772:\tmov\trbp, rsp': 1,
                '0x400775:\tsub\trsp, 0x20': 1,
                '0x400779:\tmov\tqword ptr [rbp - 0x18], rdi': 1,
                '0x40077d:\tmov\tedi, 0xa': 1,
                '0x400782:\tcall\t0x400580': 1,
                '0x400787:\tmov\tdword ptr [rbp - 4], 0': 1,
                '0x40078e:\tjmp\t0x40079e': 1,
                '0x40079e:\tcmp\tdword ptr [rbp - 4], 0x1f': 2,
                '0x4007a2:\tjle\t0x400790': 2,
                '0x400790:\tmov\tedi, 0x2d': 1,
                '0x400795:\tcall\t0x400580': 1,
                '0x4007a4:\tmov\tedi, 0xa': 1,
                '0x4007a9:\tcall\t0x400580': 1,
                '0x40079a:\tadd\tdword ptr [rbp - 4], 1': 1,
                '0x4007ae:\tmov\tdword ptr [rbp - 8], 0': 1,
                '0x4007b5:\tjmp\t0x400829': 1,
                '0x400829:\tcmp\tdword ptr [rbp - 8], 0xe': 2,
                '0x40082d:\tjle\t0x4007b7': 2,
                '0x4007b7:\tmov\tedi, 0x7c': 1,
                '0x4007bc:\tcall\t0x400580': 1,
                '0x40082f:\tmov\tdword ptr [rbp - 0x10], 0': 1,
                '0x400836:\tjmp\t0x400846': 1,
                '0x4007c1:\tmov\tdword ptr [rbp - 0xc], 0': 1,
                '0x4007c8:\tjmp\t0x40080b': 1,
                '0x400846:\tcmp\tdword ptr [rbp - 0x10], 0x1f': 2,
                '0x40084a:\tjle\t0x400838': 2,
                '0x40080b:\tcmp\tdword ptr [rbp - 0xc], 0x1d': 2,
                '0x40080f:\tjle\t0x4007ca': 2,
                '0x400838:\tmov\tedi, 0x2d': 1,
                '0x40083d:\tcall\t0x400580': 1,
                '0x40084c:\tmov\tedi, 0xa': 1,
                '0x400851:\tcall\t0x400580': 1,
                '0x4007ca:\tmov\teax, dword ptr [rbp - 8]': 1,
                '0x4007cd:\tmovsxd\trdx, eax': 1,
                '0x4007d0:\tmov\trax, rdx': 1,
                '0x4007d3:\tshl\trax, 4': 1,
                '0x4007d7:\tsub\trax, rdx': 1,
                '0x4007da:\tadd\trax, rax': 1,
                '0x4007dd:\tmov\trdx, rax': 1,
                '0x4007e0:\tmov\trax, qword ptr [rbp - 0x18]': 1,
                '0x4007e4:\tadd\trdx, rax': 1,
                '0x4007e7:\tmov\teax, dword ptr [rbp - 0xc]': 1,
                '0x4007ea:\tcdqe\t': 1,
                '0x4007ec:\tmovzx\teax, byte ptr [rdx + rax]': 1,
                '0x4007f0:\ttest\tal, al': 1,
                '0x4007f2:\tje\t0x4007fb': 1,
                '0x400811:\tmov\tedi, 0x7c': 1,
                '0x400816:\tcall\t0x400580': 1,
                '0x400842:\tadd\tdword ptr [rbp - 0x10], 1': 1,
                '0x400856:\tnop\t': 1,
                '0x400857:\tleave\t': 1,
                '0x400858:\tret\t': 1,
                '0x4007fb:\tmov\teax, 0x20': 1,
                '0x400800:\tmov\tedi, eax': 2,
                '0x400802:\tcall\t0x400580': 2,
                '0x4007f4:\tmov\teax, 0x58': 1,
                '0x4007f9:\tjmp\t0x400800': 1,
                '0x40081b:\tmov\tedi, 0xa': 1,
                '0x400820:\tcall\t0x400580': 1,
                '0x400807:\tadd\tdword ptr [rbp - 0xc], 1': 1,
                '0x400825:\tadd\tdword ptr [rbp - 8], 1': 1,
            },
            4196441: {
                '0x400859:\tpush\trbp': 1,
                '0x40085a:\tmov\trbp, rsp': 1,
                '0x40085d:\tmov\tqword ptr [rbp - 0x18], rdi': 1,
                '0x400861:\tmov\tdword ptr [rbp - 0x1c], esi': 1,
                '0x400864:\tmov\tdword ptr [rbp - 0x20], edx': 1,
                '0x400867:\tmov\tdword ptr [rbp - 4], 0': 1,
                '0x40086e:\tcmp\tdword ptr [rbp - 0x1c], 0': 1,
                '0x400872:\tjle\t0x4008ac': 1,
                '0x4008ac:\tcmp\tdword ptr [rbp - 0x1c], 0': 2,
                '0x4008b0:\tjle\t0x4008e1': 2,
                '0x400874:\tcmp\tdword ptr [rbp - 0x20], 0': 1,
                '0x400878:\tjle\t0x4008ac': 1,
                '0x4008e1:\tcmp\tdword ptr [rbp - 0x1c], 0': 2,
                '0x4008e5:\tjle\t0x40091f': 2,
                '0x4008b2:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
                '0x4008b5:\tmovsxd\trdx, eax': 1,
                '0x4008b8:\tmov\trax, rdx': 1,
                '0x4008bb:\tshl\trax, 4': 1,
                '0x4008bf:\tsub\trax, rdx': 1,
                '0x4008c2:\tadd\trax, rax': 1,
                '0x4008c5:\tlea\trdx, [rax - 0x1e]': 1,
                '0x4008c9:\tmov\trax, qword ptr [rbp - 0x18]': 1,
                '0x4008cd:\tadd\trdx, rax': 1,
                '0x4008d0:\tmov\teax, dword ptr [rbp - 0x20]': 1,
                '0x4008d3:\tcdqe\t': 1,
                '0x4008d5:\tmovzx\teax, byte ptr [rdx + rax]': 1,
                '0x4008d9:\ttest\tal, al': 1,
                '0x4008db:\tje\t0x4008e1': 1,
                '0x40087a:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
                '0x40087d:\tmovsxd\trdx, eax': 1,
                '0x400880:\tmov\trax, rdx': 1,
                '0x400883:\tshl\trax, 4': 1,
                '0x400887:\tsub\trax, rdx': 1,
                '0x40088a:\tadd\trax, rax': 1,
                '0x40088d:\tlea\trdx, [rax - 0x1e]': 1,
                '0x400891:\tmov\trax, qword ptr [rbp - 0x18]': 1,
                '0x400895:\tadd\trdx, rax': 1,
                '0x400898:\tmov\teax, dword ptr [rbp - 0x20]': 1,
                '0x40089b:\tsub\teax, 1': 1,
                '0x40089e:\tcdqe\t': 1,
                '0x4008a0:\tmovzx\teax, byte ptr [rdx + rax]': 1,
                '0x4008a4:\ttest\tal, al': 1,
                '0x4008a6:\tje\t0x4008ac': 1,
                '0x40091f:\tcmp\tdword ptr [rbp - 0x20], 0': 2,
                '0x400923:\tjle\t0x400956': 2,
                '0x4008e7:\tcmp\tdword ptr [rbp - 0x20], 0x1c': 1,
                '0x4008eb:\tjg\t0x40091f': 1,
                '0x4008dd:\tadd\tdword ptr [rbp - 4], 1': 1,
                '0x4008a8:\tadd\tdword ptr [rbp - 4], 1': 1,
                '0x400956:\tcmp\tdword ptr [rbp - 0x20], 0x1c': 2,
                '0x40095a:\tjg\t0x40098d': 2,
                '0x400925:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
                '0x400928:\tmovsxd\trdx, eax': 1,
                '0x40092b:\tmov\trax, rdx': 1,
                '0x40092e:\tshl\trax, 4': 1,
                '0x400932:\tsub\trax, rdx': 1,
                '0x400935:\tadd\trax, rax': 1,
                '0x400938:\tmov\trdx, rax': 1,
                '0x40093b:\tmov\trax, qword ptr [rbp - 0x18]': 1,
                '0x40093f:\tadd\trdx, rax': 1,
                '0x400942:\tmov\teax, dword ptr [rbp - 0x20]': 1,
                '0x400945:\tsub\teax, 1': 1,
                '0x400948:\tcdqe\t': 1,
                '0x40094a:\tmovzx\teax, byte ptr [rdx + rax]': 1,
                '0x40094e:\ttest\tal, al': 1,
                '0x400950:\tje\t0x400956': 1,
                '0x4008ed:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
                '0x4008f0:\tmovsxd\trdx, eax': 1,
                '0x4008f3:\tmov\trax, rdx': 1,
                '0x4008f6:\tshl\trax, 4': 1,
                '0x4008fa:\tsub\trax, rdx': 1,
                '0x4008fd:\tadd\trax, rax': 1,
                '0x400900:\tlea\trdx, [rax - 0x1e]': 1,
                '0x400904:\tmov\trax, qword ptr [rbp - 0x18]': 1,
                '0x400908:\tadd\trdx, rax': 1,
                '0x40090b:\tmov\teax, dword ptr [rbp - 0x20]': 1,
                '0x40090e:\tadd\teax, 1': 1,
                '0x400911:\tcdqe\t': 1,
                '0x400913:\tmovzx\teax, byte ptr [rdx + rax]': 1,
                '0x400917:\ttest\tal, al': 1,
                '0x400919:\tje\t0x40091f': 1,
                '0x40095c:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
                '0x40095f:\tmovsxd\trdx, eax': 1,
                '0x400962:\tmov\trax, rdx': 1,
                '0x400965:\tshl\trax, 4': 1,
                '0x400969:\tsub\trax, rdx': 1,
                '0x40096c:\tadd\trax, rax': 1,
                '0x40096f:\tmov\trdx, rax': 1,
                '0x400972:\tmov\trax, qword ptr [rbp - 0x18]': 1,
                '0x400976:\tadd\trdx, rax': 1,
                '0x400979:\tmov\teax, dword ptr [rbp - 0x20]': 1,
                '0x40097c:\tadd\teax, 1': 1,
                '0x40097f:\tcdqe\t': 1,
                '0x400981:\tmovzx\teax, byte ptr [rdx + rax]': 1,
                '0x400985:\ttest\tal, al': 1,
                '0x400987:\tje\t0x40098d': 1,
                '0x40098d:\tcmp\tdword ptr [rbp - 0x1c], 0xd': 2,
                '0x400991:\tjg\t0x4009cd': 2,
                '0x400952:\tadd\tdword ptr [rbp - 4], 1': 1,
                '0x40091b:\tadd\tdword ptr [rbp - 4], 1': 1,
                '0x400989:\tadd\tdword ptr [rbp - 4], 1': 1,
                '0x400993:\tcmp\tdword ptr [rbp - 0x20], 0': 1,
                '0x400997:\tjle\t0x4009cd': 1,
                '0x4009cd:\tcmp\tdword ptr [rbp - 0x1c], 0xd': 2,
                '0x4009d1:\tjg\t0x400a04': 2,
                '0x400999:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
                '0x40099c:\tcdqe\t': 1,
                '0x40099e:\tlea\trdx, [rax + 1]': 1,
                '0x4009a2:\tmov\trax, rdx': 1,
                '0x4009a5:\tshl\trax, 4': 1,
                '0x4009a9:\tsub\trax, rdx': 1,
                '0x4009ac:\tadd\trax, rax': 1,
                '0x4009af:\tmov\trdx, rax': 1,
                '0x4009b2:\tmov\trax, qword ptr [rbp - 0x18]': 1,
                '0x4009b6:\tadd\trdx, rax': 1,
                '0x4009b9:\tmov\teax, dword ptr [rbp - 0x20]': 1,
                '0x4009bc:\tsub\teax, 1': 1,
                '0x4009bf:\tcdqe\t': 1,
                '0x4009c1:\tmovzx\teax, byte ptr [rdx + rax]': 1,
                '0x4009c5:\ttest\tal, al': 1,
                '0x4009c7:\tje\t0x4009cd': 1,
                '0x4009d3:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
                '0x4009d6:\tcdqe\t': 1,
                '0x4009d8:\tlea\trdx, [rax + 1]': 1,
                '0x4009dc:\tmov\trax, rdx': 1,
                '0x4009df:\tshl\trax, 4': 1,
                '0x4009e3:\tsub\trax, rdx': 1,
                '0x4009e6:\tadd\trax, rax': 1,
                '0x4009e9:\tmov\trdx, rax': 1,
                '0x4009ec:\tmov\trax, qword ptr [rbp - 0x18]': 1,
                '0x4009f0:\tadd\trdx, rax': 1,
                '0x4009f3:\tmov\teax, dword ptr [rbp - 0x20]': 1,
                '0x4009f6:\tcdqe\t': 1,
                '0x4009f8:\tmovzx\teax, byte ptr [rdx + rax]': 1,
                '0x4009fc:\ttest\tal, al': 1,
                '0x4009fe:\tje\t0x400a04': 1,
                '0x400a04:\tcmp\tdword ptr [rbp - 0x1c], 0xd': 2,
                '0x400a08:\tjg\t0x400a44': 2,
                '0x4009c9:\tadd\tdword ptr [rbp - 4], 1': 1,
                '0x400a00:\tadd\tdword ptr [rbp - 4], 1': 1,
                '0x400a0a:\tcmp\tdword ptr [rbp - 0x20], 0x1c': 1,
                '0x400a0e:\tjg\t0x400a44': 1,
                '0x400a44:\tmov\teax, dword ptr [rbp - 4]': 2,
                '0x400a47:\tpop\trbp': 2,
                '0x400a48:\tret\t': 2,
                '0x400a10:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
                '0x400a13:\tcdqe\t': 1,
                '0x400a15:\tlea\trdx, [rax + 1]': 1,
                '0x400a19:\tmov\trax, rdx': 1,
                '0x400a1c:\tshl\trax, 4': 1,
                '0x400a20:\tsub\trax, rdx': 1,
                '0x400a23:\tadd\trax, rax': 1,
                '0x400a26:\tmov\trdx, rax': 1,
                '0x400a29:\tmov\trax, qword ptr [rbp - 0x18]': 1,
                '0x400a2d:\tadd\trdx, rax': 1,
                '0x400a30:\tmov\teax, dword ptr [rbp - 0x20]': 1,
                '0x400a33:\tadd\teax, 1': 1,
                '0x400a36:\tcdqe\t': 1,
                '0x400a38:\tmovzx\teax, byte ptr [rdx + rax]': 1,
                '0x400a3c:\ttest\tal, al': 1,
                '0x400a3e:\tje\t0x400a44': 1,
                '0x400a40:\tadd\tdword ptr [rbp - 4], 1': 1,
            },
            4196937: {
                '0x400a49:\tpush\trbp': 1,
                '0x400a4a:\tmov\trbp, rsp': 1,
                '0x400a4d:\tsub\trsp, 0x3b0': 1,
                '0x400a54:\tmov\tedi, 0x3039': 1,
                '0x400a59:\tcall\t0x4005b0': 1,
                '0x400a5e:\tmov\tdword ptr [rbp - 4], 0': 1,
                '0x400a65:\tjmp\t0x400ace': 1,
                '0x400ace:\tcmp\tdword ptr [rbp - 4], 0xe': 2,
                '0x400ad2:\tjle\t0x400a67': 2,
                '0x400a67:\tmov\tdword ptr [rbp - 8], 0': 1,
                '0x400a6e:\tjmp\t0x400ac4': 1,
                '0x400ad4:\tlea\trax, [rbp - 0x1e0]': 1,
                '0x400adb:\tmov\trdi, rax': 1,
                '0x400ade:\tcall\t0x400771': 1,
                '0x400ac4:\tcmp\tdword ptr [rbp - 8], 0x1d': 2,
                '0x400ac8:\tjle\t0x400a70': 2,
                '0x400ae3:\tmov\tdword ptr [rbp - 0xc], 0': 1,
                '0x400aea:\tjmp\t0x400c37': 1,
                '0x400a70:\tcall\t0x4005c0': 1,
                '0x400aca:\tadd\tdword ptr [rbp - 4], 1': 1,
                '0x400c37:\tcmp\tdword ptr [rbp - 0xc], 0xe': 2,
                '0x400c3b:\tjle\t0x400aef': 2,
                '0x400a75:\tcvtsi2ss\txmm0, eax': 1,
                '0x400a79:\tmovss\txmm1, dword ptr [rip + 0x2e7]': 1,
                '0x400a81:\tdivss\txmm0, xmm1': 1,
                '0x400a85:\tmovaps\txmm1, xmm0': 1,
                '0x400a88:\tmovss\txmm0, dword ptr [rip + 0x2dc]': 1,
                '0x400a90:\tcomiss\txmm0, xmm1': 1,
                '0x400a93:\tseta\tal': 1,
                '0x400a96:\tmov\tesi, eax': 1,
                '0x400a98:\tmov\teax, dword ptr [rbp - 8]': 1,
                '0x400a9b:\tmovsxd\trcx, eax': 1,
                '0x400a9e:\tmov\teax, dword ptr [rbp - 4]': 1,
                '0x400aa1:\tmovsxd\trdx, eax': 1,
                '0x400aa4:\tmov\trax, rdx': 1,
                '0x400aa7:\tshl\trax, 4': 1,
                '0x400aab:\tsub\trax, rdx': 1,
                '0x400aae:\tadd\trax, rax': 1,
                '0x400ab1:\tadd\trax, rbp': 1,
                '0x400ab4:\tadd\trax, rcx': 1,
                '0x400ab7:\tsub\trax, 0x1e0': 1,
                '0x400abd:\tmov\tbyte ptr [rax], sil': 1,
                '0x400ac0:\tadd\tdword ptr [rbp - 8], 1': 1,
                '0x400aef:\tmov\tdword ptr [rbp - 0x10], 0': 1,
                '0x400af6:\tjmp\t0x400c29': 1,
                '0x400c41:\tmov\tdword ptr [rbp - 0x14], 0': 1,
                '0x400c48:\tjmp\t0x400cb0': 1,
                '0x400c29:\tcmp\tdword ptr [rbp - 0x10], 0x1d': 3,
                '0x400c2d:\tjle\t0x400afb': 3,
                '0x400cb0:\tcmp\tdword ptr [rbp - 0x14], 0xe': 2,
                '0x400cb4:\tjle\t0x400c4a': 2,
                '0x400afb:\tmov\tedx, dword ptr [rbp - 0x10]': 1,
                '0x400afe:\tmov\tecx, dword ptr [rbp - 0xc]': 1,
                '0x400b01:\tlea\trax, [rbp - 0x1e0]': 1,
                '0x400b08:\tmov\tesi, ecx': 1,
                '0x400b0a:\tmov\trdi, rax': 1,
                '0x400b0d:\tcall\t0x400859': 1,
                '0x400c33:\tadd\tdword ptr [rbp - 0xc], 1': 1,
                '0x400c4a:\tmov\tdword ptr [rbp - 0x18], 0': 1,
                '0x400c51:\tjmp\t0x400ca6': 1,
                '0x400cb6:\tmov\tedi, 0x1f4': 1,
                '0x400cbb:\tcall\t0x4006b6': 1,
                '0x400b12:\tmov\tdword ptr [rbp - 0x1c], eax': 1,
                '0x400b15:\tmov\teax, dword ptr [rbp - 0x10]': 1,
                '0x400b18:\tmovsxd\trcx, eax': 1,
                '0x400b1b:\tmov\teax, dword ptr [rbp - 0xc]': 1,
                '0x400b1e:\tmovsxd\trdx, eax': 1,
                '0x400b21:\tmov\trax, rdx': 1,
                '0x400b24:\tshl\trax, 4': 1,
                '0x400b28:\tsub\trax, rdx': 1,
                '0x400b2b:\tadd\trax, rax': 1,
                '0x400b2e:\tadd\trax, rbp': 1,
                '0x400b31:\tadd\trax, rcx': 1,
                '0x400b34:\tsub\trax, 0x1e0': 1,
                '0x400b3a:\tmovzx\teax, byte ptr [rax]': 1,
                '0x400b3d:\ttest\tal, al': 1,
                '0x400b3f:\tje\t0x400b7a': 1,
                '0x400ca6:\tcmp\tdword ptr [rbp - 0x18], 0x1d': 2,
                '0x400caa:\tjle\t0x400c53': 2,
                '0x400cc0:\tjmp\t0x400ad4': 1,
                '0x400b7a:\tmov\teax, dword ptr [rbp - 0x10]': 1,
                '0x400b7d:\tmovsxd\trcx, eax': 1,
                '0x400b80:\tmov\teax, dword ptr [rbp - 0xc]': 1,
                '0x400b83:\tmovsxd\trdx, eax': 1,
                '0x400b86:\tmov\trax, rdx': 1,
                '0x400b89:\tshl\trax, 4': 1,
                '0x400b8d:\tsub\trax, rdx': 1,
                '0x400b90:\tadd\trax, rax': 1,
                '0x400b93:\tadd\trax, rbp': 1,
                '0x400b96:\tadd\trax, rcx': 1,
                '0x400b99:\tsub\trax, 0x1e0': 1,
                '0x400b9f:\tmovzx\teax, byte ptr [rax]': 1,
                '0x400ba2:\ttest\tal, al': 1,
                '0x400ba4:\tjne\t0x400bd6': 1,
                '0x400b41:\tcmp\tdword ptr [rbp - 0x1c], 1': 1,
                '0x400b45:\tjle\t0x400b4d': 1,
                '0x400c53:\tmov\teax, dword ptr [rbp - 0x18]': 1,
                '0x400c56:\tmovsxd\trcx, eax': 1,
                '0x400c59:\tmov\teax, dword ptr [rbp - 0x14]': 1,
                '0x400c5c:\tmovsxd\trdx, eax': 1,
                '0x400c5f:\tmov\trax, rdx': 1,
                '0x400c62:\tshl\trax, 4': 1,
                '0x400c66:\tsub\trax, rdx': 1,
                '0x400c69:\tadd\trax, rax': 1,
                '0x400c6c:\tadd\trax, rbp': 1,
                '0x400c6f:\tadd\trax, rcx': 1,
                '0x400c72:\tsub\trax, 0x3b0': 1,
                '0x400c78:\tmovzx\tecx, byte ptr [rax]': 1,
                '0x400c7b:\tmov\teax, dword ptr [rbp - 0x18]': 1,
                '0x400c7e:\tmovsxd\trsi, eax': 1,
                '0x400c81:\tmov\teax, dword ptr [rbp - 0x14]': 1,
                '0x400c84:\tmovsxd\trdx, eax': 1,
                '0x400c87:\tmov\trax, rdx': 1,
                '0x400c8a:\tshl\trax, 4': 1,
                '0x400c8e:\tsub\trax, rdx': 1,
                '0x400c91:\tadd\trax, rax': 1,
                '0x400c94:\tadd\trax, rbp': 1,
                '0x400c97:\tadd\trax, rsi': 1,
                '0x400c9a:\tsub\trax, 0x1e0': 1,
                '0x400ca0:\tmov\tbyte ptr [rax], cl': 1,
                '0x400ca2:\tadd\tdword ptr [rbp - 0x18], 1': 1,
                '0x400cac:\tadd\tdword ptr [rbp - 0x14], 1': 1,
                '0x400ba6:\tcmp\tdword ptr [rbp - 0x1c], 3': 1,
                '0x400baa:\tjne\t0x400bd6': 1,
                '0x400bd6:\tmov\teax, dword ptr [rbp - 0x10]': 1,
                '0x400bd9:\tmovsxd\trcx, eax': 1,
                '0x400bdc:\tmov\teax, dword ptr [rbp - 0xc]': 1,
                '0x400bdf:\tmovsxd\trdx, eax': 1,
                '0x400be2:\tmov\trax, rdx': 1,
                '0x400be5:\tshl\trax, 4': 1,
                '0x400be9:\tsub\trax, rdx': 1,
                '0x400bec:\tadd\trax, rax': 1,
                '0x400bef:\tadd\trax, rbp': 1,
                '0x400bf2:\tadd\trax, rcx': 1,
                '0x400bf5:\tsub\trax, 0x1e0': 1,
                '0x400bfb:\tmovzx\tecx, byte ptr [rax]': 1,
                '0x400bfe:\tmov\teax, dword ptr [rbp - 0x10]': 1,
                '0x400c01:\tmovsxd\trsi, eax': 1,
                '0x400c04:\tmov\teax, dword ptr [rbp - 0xc]': 1,
                '0x400c07:\tmovsxd\trdx, eax': 1,
                '0x400c0a:\tmov\trax, rdx': 1,
                '0x400c0d:\tshl\trax, 4': 1,
                '0x400c11:\tsub\trax, rdx': 1,
                '0x400c14:\tadd\trax, rax': 1,
                '0x400c17:\tadd\trax, rbp': 1,
                '0x400c1a:\tadd\trax, rsi': 1,
                '0x400c1d:\tsub\trax, 0x3b0': 1,
                '0x400c23:\tmov\tbyte ptr [rax], cl': 1,
                '0x400c25:\tadd\tdword ptr [rbp - 0x10], 1': 2,
                '0x400b4d:\tmov\teax, dword ptr [rbp - 0x10]': 1,
                '0x400b50:\tmovsxd\trcx, eax': 1,
                '0x400b53:\tmov\teax, dword ptr [rbp - 0xc]': 1,
                '0x400b56:\tmovsxd\trdx, eax': 1,
                '0x400b59:\tmov\trax, rdx': 1,
                '0x400b5c:\tshl\trax, 4': 1,
                '0x400b60:\tsub\trax, rdx': 1,
                '0x400b63:\tadd\trax, rax': 1,
                '0x400b66:\tadd\trax, rbp': 1,
                '0x400b69:\tadd\trax, rcx': 1,
                '0x400b6c:\tsub\trax, 0x3b0': 1,
                '0x400b72:\tmov\tbyte ptr [rax], 0': 1,
                '0x400b75:\tjmp\t0x400c25': 1,
                '0x400b47:\tcmp\tdword ptr [rbp - 0x1c], 3': 1,
                '0x400b4b:\tjle\t0x400b7a': 1,
                '0x400bac:\tmov\teax, dword ptr [rbp - 0x10]': 1,
                '0x400baf:\tmovsxd\trcx, eax': 1,
                '0x400bb2:\tmov\teax, dword ptr [rbp - 0xc]': 1,
                '0x400bb5:\tmovsxd\trdx, eax': 1,
                '0x400bb8:\tmov\trax, rdx': 1,
                '0x400bbb:\tshl\trax, 4': 1,
                '0x400bbf:\tsub\trax, rdx': 1,
                '0x400bc2:\tadd\trax, rax': 1,
                '0x400bc5:\tadd\trax, rbp': 1,
                '0x400bc8:\tadd\trax, rcx': 1,
                '0x400bcb:\tsub\trax, 0x3b0': 1,
                '0x400bd1:\tmov\tbyte ptr [rax], 1': 1,
                '0x400bd4:\tjmp\t0x400c25': 1,
            },
            4197573: {
                '0x400cc5:\tnop\tword ptr cs:[rax + rax]': 1,
                '0x400ccf:\tnop\t': 1,
            },
            4197584: {
                '0x400cd0:\tendbr64\t': 1,
                '0x400cd4:\tpush\tr15': 1,
                '0x400cd6:\tmov\tr15, rdx': 1,
                '0x400cd9:\tpush\tr14': 1,
                '0x400cdb:\tmov\tr14, rsi': 1,
                '0x400cde:\tpush\tr13': 1,
                '0x400ce0:\tmov\tr13d, edi': 1,
                '0x400ce3:\tpush\tr12': 1,
                '0x400ce5:\tlea\tr12, [rip + 0x201114]': 1,
                '0x400cec:\tpush\trbp': 1,
                '0x400ced:\tlea\trbp, [rip + 0x201114]': 1,
                '0x400cf4:\tpush\trbx': 1,
                '0x400cf5:\tsub\trbp, r12': 1,
                '0x400cf8:\tsub\trsp, 8': 1,
                '0x400cfc:\tcall\t0x400550': 1,
                '0x400d01:\tsar\trbp, 3': 1,
                '0x400d05:\tje\t0x400d26': 1,
                '0x400d26:\tadd\trsp, 8': 1,
                '0x400d2a:\tpop\trbx': 1,
                '0x400d2b:\tpop\trbp': 1,
                '0x400d2c:\tpop\tr12': 1,
                '0x400d2e:\tpop\tr13': 1,
                '0x400d30:\tpop\tr14': 1,
                '0x400d32:\tpop\tr15': 1,
                '0x400d34:\tret\t': 1,
                '0x400d07:\txor\tebx, ebx': 1,
                '0x400d09:\tnop\tdword ptr [rax]': 1,
                '0x400d10:\tmov\trdx, r15': 2,
                '0x400d13:\tmov\trsi, r14': 2,
                '0x400d16:\tmov\tedi, r13d': 2,
                '0x400d19:\tcall\tqword ptr [r12 + rbx*8]': 2,
                '0x400d1d:\tadd\trbx, 1': 1,
                '0x400d21:\tcmp\trbp, rbx': 1,
                '0x400d24:\tjne\t0x400d10': 1,
            },
            4197685: {
                '0x400d35:\tnop\tword ptr cs:[rax + rax]': 1,
            },
            4197696: {
                '0x400d40:\tendbr64\t': 1,
                '0x400d44:\tret\t': 1,
            },
            4197704: {
                '0x400d48:\tendbr64\t': 1,
                '0x400d4c:\tsub\trsp, 8': 1,
                '0x400d50:\tadd\trsp, 8': 1,
                '0x400d54:\tret\t': 1,
            },
            7340032: {
                
            },
            7340040: {
                
            },
            7340048: {
                
            },
            7340056: {
                
            },
            7340064: {
                
            },
            7340072: {
                
            },
            8392784: {
                
            },
            8392792: {
                
            },
        },
        'asm_counts': {
            '0x400550:\tendbr64\t': 1,
            '0x400554:\tsub\trsp, 8': 1,
            '0x400558:\tmov\trax, qword ptr [rip + 0x201a91]': 1,
            '0x40055f:\ttest\trax, rax': 1,
            '0x400562:\tje\t0x400566': 1,
            '0x400566:\tadd\trsp, 8': 1,
            '0x40056a:\tret\t': 1,
            '0x400564:\tcall\trax': 1,
            '0x400570:\tpush\tqword ptr [rip + 0x201a92]': 1,
            '0x400576:\tjmp\tqword ptr [rip + 0x201a94]': 1,
            '0x40057c:\tnop\tdword ptr [rax]': 1,
            '0x400580:\tjmp\tqword ptr [rip + 0x201a92]': 1,
            '0x400590:\tjmp\tqword ptr [rip + 0x201a8a]': 1,
            '0x4005a0:\tjmp\tqword ptr [rip + 0x201a82]': 1,
            '0x4005b0:\tjmp\tqword ptr [rip + 0x201a7a]': 1,
            '0x4005c0:\tjmp\tqword ptr [rip + 0x201a72]': 1,
            '0x4005d0:\tendbr64\t': 1,
            '0x4005d4:\txor\tebp, ebp': 1,
            '0x4005d6:\tmov\tr9, rdx': 1,
            '0x4005d9:\tpop\trsi': 1,
            '0x4005da:\tmov\trdx, rsp': 1,
            '0x4005dd:\tand\trsp, 0xfffffffffffffff0': 1,
            '0x4005e1:\tpush\trax': 1,
            '0x4005e2:\tpush\trsp': 1,
            '0x4005e3:\tmov\tr8, 0x400d40': 1,
            '0x4005ea:\tmov\trcx, 0x400cd0': 1,
            '0x4005f1:\tmov\trdi, 0x400a49': 1,
            '0x4005f8:\tcall\tqword ptr [rip + 0x2019ea]': 1,
            '0x4005fe:\thlt\t': 1,
            '0x4005ff:\tnop\t': 1,
            '0x400600:\tendbr64\t': 1,
            '0x400604:\tret\t': 1,
            '0x400605:\tnop\tword ptr cs:[rax + rax]': 1,
            '0x40060f:\tnop\t': 1,
            '0x400610:\tlea\trdi, [rip + 0x201a31]': 1,
            '0x400617:\tlea\trax, [rip + 0x201a2a]': 1,
            '0x40061e:\tcmp\trax, rdi': 1,
            '0x400621:\tje\t0x400638': 1,
            '0x400638:\tret\t': 1,
            '0x400623:\tmov\trax, qword ptr [rip + 0x2019b6]': 1,
            '0x40062a:\ttest\trax, rax': 1,
            '0x40062d:\tje\t0x400638': 1,
            '0x40062f:\tjmp\trax': 1,
            '0x400631:\tnop\tdword ptr [rax]': 1,
            '0x400639:\tnop\tdword ptr [rax]': 1,
            '0x400640:\tlea\trdi, [rip + 0x201a01]': 1,
            '0x400647:\tlea\trsi, [rip + 0x2019fa]': 1,
            '0x40064e:\tsub\trsi, rdi': 1,
            '0x400651:\tsar\trsi, 3': 1,
            '0x400655:\tmov\trax, rsi': 1,
            '0x400658:\tshr\trax, 0x3f': 1,
            '0x40065c:\tadd\trsi, rax': 1,
            '0x40065f:\tsar\trsi, 1': 1,
            '0x400662:\tje\t0x400678': 1,
            '0x400678:\tret\t': 1,
            '0x400664:\tmov\trax, qword ptr [rip + 0x20198d]': 1,
            '0x40066b:\ttest\trax, rax': 1,
            '0x40066e:\tje\t0x400678': 1,
            '0x400670:\tjmp\trax': 1,
            '0x400672:\tnop\tword ptr [rax + rax]': 1,
            '0x400679:\tnop\tdword ptr [rax]': 1,
            '0x400680:\tendbr64\t': 1,
            '0x400684:\tcmp\tbyte ptr [rip + 0x2019b9], 0': 1,
            '0x40068b:\tjne\t0x4006a0': 1,
            '0x40068d:\tpush\trbp': 1,
            '0x40068e:\tmov\trbp, rsp': 1,
            '0x400691:\tcall\t0x400610': 1,
            '0x4006a0:\tret\t': 1,
            '0x400696:\tmov\tbyte ptr [rip + 0x2019a7], 1': 1,
            '0x40069d:\tpop\trbp': 1,
            '0x40069e:\tret\t': 1,
            '0x40069f:\tnop\t': 1,
            '0x4006a1:\tnop\tword ptr cs:[rax + rax]': 1,
            '0x4006ac:\tnop\tdword ptr [rax]': 1,
            '0x4006b0:\tendbr64\t': 1,
            '0x4006b4:\tjmp\t0x400640': 1,
            '0x4006b6:\tpush\trbp': 1,
            '0x4006b7:\tmov\trbp, rsp': 1,
            '0x4006ba:\tsub\trsp, 0x30': 1,
            '0x4006be:\tmov\tqword ptr [rbp - 0x28], rdi': 1,
            '0x4006c2:\tcmp\tqword ptr [rbp - 0x28], 0': 1,
            '0x4006c7:\tjns\t0x4006de': 1,
            '0x4006c9:\tcall\t0x400590': 1,
            '0x4006de:\tmov\trcx, qword ptr [rbp - 0x28]': 1,
            '0x4006e2:\tmovabs\trdx, 0x20c49ba5e353f7cf': 1,
            '0x4006ec:\tmov\trax, rcx': 1,
            '0x4006ef:\timul\trdx': 1,
            '0x4006f2:\tsar\trdx, 7': 1,
            '0x4006f6:\tmov\trax, rcx': 1,
            '0x4006f9:\tsar\trax, 0x3f': 1,
            '0x4006fd:\tsub\trdx, rax': 1,
            '0x400700:\tmov\trax, rdx': 1,
            '0x400703:\tmov\tqword ptr [rbp - 0x20], rax': 1,
            '0x400707:\tmov\trcx, qword ptr [rbp - 0x28]': 1,
            '0x40070b:\tmovabs\trdx, 0x20c49ba5e353f7cf': 1,
            '0x400715:\tmov\trax, rcx': 1,
            '0x400718:\timul\trdx': 1,
            '0x40071b:\tsar\trdx, 7': 1,
            '0x40071f:\tmov\trax, rcx': 1,
            '0x400722:\tsar\trax, 0x3f': 1,
            '0x400726:\tsub\trdx, rax': 1,
            '0x400729:\tmov\trax, rdx': 1,
            '0x40072c:\timul\trax, rax, 0x3e8': 1,
            '0x400733:\tsub\trcx, rax': 1,
            '0x400736:\tmov\trax, rcx': 1,
            '0x400739:\timul\trax, rax, 0xf4240': 1,
            '0x400740:\tmov\tqword ptr [rbp - 0x18], rax': 1,
            '0x400744:\tlea\trdx, [rbp - 0x20]': 2,
            '0x400748:\tlea\trax, [rbp - 0x20]': 2,
            '0x40074c:\tmov\trsi, rdx': 2,
            '0x40074f:\tmov\trdi, rax': 2,
            '0x400752:\tcall\t0x4005a0': 2,
            '0x4006ce:\tmov\tdword ptr [rax], 0x16': 1,
            '0x4006d4:\tmov\teax, 0xffffffff': 1,
            '0x4006d9:\tjmp\t0x40076f': 1,
            '0x400757:\tmov\tdword ptr [rbp - 4], eax': 1,
            '0x40075a:\tcmp\tdword ptr [rbp - 4], 0': 1,
            '0x40075e:\tje\t0x40076c': 1,
            '0x40076f:\tleave\t': 2,
            '0x400770:\tret\t': 2,
            '0x40076c:\tmov\teax, dword ptr [rbp - 4]': 1,
            '0x400760:\tcall\t0x400590': 1,
            '0x400765:\tmov\teax, dword ptr [rax]': 1,
            '0x400767:\tcmp\teax, 4': 1,
            '0x40076a:\tje\t0x400744': 1,
            '0x400771:\tpush\trbp': 1,
            '0x400772:\tmov\trbp, rsp': 1,
            '0x400775:\tsub\trsp, 0x20': 1,
            '0x400779:\tmov\tqword ptr [rbp - 0x18], rdi': 1,
            '0x40077d:\tmov\tedi, 0xa': 1,
            '0x400782:\tcall\t0x400580': 1,
            '0x400787:\tmov\tdword ptr [rbp - 4], 0': 1,
            '0x40078e:\tjmp\t0x40079e': 1,
            '0x40079e:\tcmp\tdword ptr [rbp - 4], 0x1f': 2,
            '0x4007a2:\tjle\t0x400790': 2,
            '0x400790:\tmov\tedi, 0x2d': 1,
            '0x400795:\tcall\t0x400580': 1,
            '0x4007a4:\tmov\tedi, 0xa': 1,
            '0x4007a9:\tcall\t0x400580': 1,
            '0x40079a:\tadd\tdword ptr [rbp - 4], 1': 1,
            '0x4007ae:\tmov\tdword ptr [rbp - 8], 0': 1,
            '0x4007b5:\tjmp\t0x400829': 1,
            '0x400829:\tcmp\tdword ptr [rbp - 8], 0xe': 2,
            '0x40082d:\tjle\t0x4007b7': 2,
            '0x4007b7:\tmov\tedi, 0x7c': 1,
            '0x4007bc:\tcall\t0x400580': 1,
            '0x40082f:\tmov\tdword ptr [rbp - 0x10], 0': 1,
            '0x400836:\tjmp\t0x400846': 1,
            '0x4007c1:\tmov\tdword ptr [rbp - 0xc], 0': 1,
            '0x4007c8:\tjmp\t0x40080b': 1,
            '0x400846:\tcmp\tdword ptr [rbp - 0x10], 0x1f': 2,
            '0x40084a:\tjle\t0x400838': 2,
            '0x40080b:\tcmp\tdword ptr [rbp - 0xc], 0x1d': 2,
            '0x40080f:\tjle\t0x4007ca': 2,
            '0x400838:\tmov\tedi, 0x2d': 1,
            '0x40083d:\tcall\t0x400580': 1,
            '0x40084c:\tmov\tedi, 0xa': 1,
            '0x400851:\tcall\t0x400580': 1,
            '0x4007ca:\tmov\teax, dword ptr [rbp - 8]': 1,
            '0x4007cd:\tmovsxd\trdx, eax': 1,
            '0x4007d0:\tmov\trax, rdx': 1,
            '0x4007d3:\tshl\trax, 4': 1,
            '0x4007d7:\tsub\trax, rdx': 1,
            '0x4007da:\tadd\trax, rax': 1,
            '0x4007dd:\tmov\trdx, rax': 1,
            '0x4007e0:\tmov\trax, qword ptr [rbp - 0x18]': 1,
            '0x4007e4:\tadd\trdx, rax': 1,
            '0x4007e7:\tmov\teax, dword ptr [rbp - 0xc]': 1,
            '0x4007ea:\tcdqe\t': 1,
            '0x4007ec:\tmovzx\teax, byte ptr [rdx + rax]': 1,
            '0x4007f0:\ttest\tal, al': 1,
            '0x4007f2:\tje\t0x4007fb': 1,
            '0x400811:\tmov\tedi, 0x7c': 1,
            '0x400816:\tcall\t0x400580': 1,
            '0x400842:\tadd\tdword ptr [rbp - 0x10], 1': 1,
            '0x400856:\tnop\t': 1,
            '0x400857:\tleave\t': 1,
            '0x400858:\tret\t': 1,
            '0x4007fb:\tmov\teax, 0x20': 1,
            '0x400800:\tmov\tedi, eax': 2,
            '0x400802:\tcall\t0x400580': 2,
            '0x4007f4:\tmov\teax, 0x58': 1,
            '0x4007f9:\tjmp\t0x400800': 1,
            '0x40081b:\tmov\tedi, 0xa': 1,
            '0x400820:\tcall\t0x400580': 1,
            '0x400807:\tadd\tdword ptr [rbp - 0xc], 1': 1,
            '0x400825:\tadd\tdword ptr [rbp - 8], 1': 1,
            '0x400859:\tpush\trbp': 1,
            '0x40085a:\tmov\trbp, rsp': 1,
            '0x40085d:\tmov\tqword ptr [rbp - 0x18], rdi': 1,
            '0x400861:\tmov\tdword ptr [rbp - 0x1c], esi': 1,
            '0x400864:\tmov\tdword ptr [rbp - 0x20], edx': 1,
            '0x400867:\tmov\tdword ptr [rbp - 4], 0': 1,
            '0x40086e:\tcmp\tdword ptr [rbp - 0x1c], 0': 1,
            '0x400872:\tjle\t0x4008ac': 1,
            '0x4008ac:\tcmp\tdword ptr [rbp - 0x1c], 0': 2,
            '0x4008b0:\tjle\t0x4008e1': 2,
            '0x400874:\tcmp\tdword ptr [rbp - 0x20], 0': 1,
            '0x400878:\tjle\t0x4008ac': 1,
            '0x4008e1:\tcmp\tdword ptr [rbp - 0x1c], 0': 2,
            '0x4008e5:\tjle\t0x40091f': 2,
            '0x4008b2:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
            '0x4008b5:\tmovsxd\trdx, eax': 1,
            '0x4008b8:\tmov\trax, rdx': 1,
            '0x4008bb:\tshl\trax, 4': 1,
            '0x4008bf:\tsub\trax, rdx': 1,
            '0x4008c2:\tadd\trax, rax': 1,
            '0x4008c5:\tlea\trdx, [rax - 0x1e]': 1,
            '0x4008c9:\tmov\trax, qword ptr [rbp - 0x18]': 1,
            '0x4008cd:\tadd\trdx, rax': 1,
            '0x4008d0:\tmov\teax, dword ptr [rbp - 0x20]': 1,
            '0x4008d3:\tcdqe\t': 1,
            '0x4008d5:\tmovzx\teax, byte ptr [rdx + rax]': 1,
            '0x4008d9:\ttest\tal, al': 1,
            '0x4008db:\tje\t0x4008e1': 1,
            '0x40087a:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
            '0x40087d:\tmovsxd\trdx, eax': 1,
            '0x400880:\tmov\trax, rdx': 1,
            '0x400883:\tshl\trax, 4': 1,
            '0x400887:\tsub\trax, rdx': 1,
            '0x40088a:\tadd\trax, rax': 1,
            '0x40088d:\tlea\trdx, [rax - 0x1e]': 1,
            '0x400891:\tmov\trax, qword ptr [rbp - 0x18]': 1,
            '0x400895:\tadd\trdx, rax': 1,
            '0x400898:\tmov\teax, dword ptr [rbp - 0x20]': 1,
            '0x40089b:\tsub\teax, 1': 1,
            '0x40089e:\tcdqe\t': 1,
            '0x4008a0:\tmovzx\teax, byte ptr [rdx + rax]': 1,
            '0x4008a4:\ttest\tal, al': 1,
            '0x4008a6:\tje\t0x4008ac': 1,
            '0x40091f:\tcmp\tdword ptr [rbp - 0x20], 0': 2,
            '0x400923:\tjle\t0x400956': 2,
            '0x4008e7:\tcmp\tdword ptr [rbp - 0x20], 0x1c': 1,
            '0x4008eb:\tjg\t0x40091f': 1,
            '0x4008dd:\tadd\tdword ptr [rbp - 4], 1': 1,
            '0x4008a8:\tadd\tdword ptr [rbp - 4], 1': 1,
            '0x400956:\tcmp\tdword ptr [rbp - 0x20], 0x1c': 2,
            '0x40095a:\tjg\t0x40098d': 2,
            '0x400925:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
            '0x400928:\tmovsxd\trdx, eax': 1,
            '0x40092b:\tmov\trax, rdx': 1,
            '0x40092e:\tshl\trax, 4': 1,
            '0x400932:\tsub\trax, rdx': 1,
            '0x400935:\tadd\trax, rax': 1,
            '0x400938:\tmov\trdx, rax': 1,
            '0x40093b:\tmov\trax, qword ptr [rbp - 0x18]': 1,
            '0x40093f:\tadd\trdx, rax': 1,
            '0x400942:\tmov\teax, dword ptr [rbp - 0x20]': 1,
            '0x400945:\tsub\teax, 1': 1,
            '0x400948:\tcdqe\t': 1,
            '0x40094a:\tmovzx\teax, byte ptr [rdx + rax]': 1,
            '0x40094e:\ttest\tal, al': 1,
            '0x400950:\tje\t0x400956': 1,
            '0x4008ed:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
            '0x4008f0:\tmovsxd\trdx, eax': 1,
            '0x4008f3:\tmov\trax, rdx': 1,
            '0x4008f6:\tshl\trax, 4': 1,
            '0x4008fa:\tsub\trax, rdx': 1,
            '0x4008fd:\tadd\trax, rax': 1,
            '0x400900:\tlea\trdx, [rax - 0x1e]': 1,
            '0x400904:\tmov\trax, qword ptr [rbp - 0x18]': 1,
            '0x400908:\tadd\trdx, rax': 1,
            '0x40090b:\tmov\teax, dword ptr [rbp - 0x20]': 1,
            '0x40090e:\tadd\teax, 1': 1,
            '0x400911:\tcdqe\t': 1,
            '0x400913:\tmovzx\teax, byte ptr [rdx + rax]': 1,
            '0x400917:\ttest\tal, al': 1,
            '0x400919:\tje\t0x40091f': 1,
            '0x40095c:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
            '0x40095f:\tmovsxd\trdx, eax': 1,
            '0x400962:\tmov\trax, rdx': 1,
            '0x400965:\tshl\trax, 4': 1,
            '0x400969:\tsub\trax, rdx': 1,
            '0x40096c:\tadd\trax, rax': 1,
            '0x40096f:\tmov\trdx, rax': 1,
            '0x400972:\tmov\trax, qword ptr [rbp - 0x18]': 1,
            '0x400976:\tadd\trdx, rax': 1,
            '0x400979:\tmov\teax, dword ptr [rbp - 0x20]': 1,
            '0x40097c:\tadd\teax, 1': 1,
            '0x40097f:\tcdqe\t': 1,
            '0x400981:\tmovzx\teax, byte ptr [rdx + rax]': 1,
            '0x400985:\ttest\tal, al': 1,
            '0x400987:\tje\t0x40098d': 1,
            '0x40098d:\tcmp\tdword ptr [rbp - 0x1c], 0xd': 2,
            '0x400991:\tjg\t0x4009cd': 2,
            '0x400952:\tadd\tdword ptr [rbp - 4], 1': 1,
            '0x40091b:\tadd\tdword ptr [rbp - 4], 1': 1,
            '0x400989:\tadd\tdword ptr [rbp - 4], 1': 1,
            '0x400993:\tcmp\tdword ptr [rbp - 0x20], 0': 1,
            '0x400997:\tjle\t0x4009cd': 1,
            '0x4009cd:\tcmp\tdword ptr [rbp - 0x1c], 0xd': 2,
            '0x4009d1:\tjg\t0x400a04': 2,
            '0x400999:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
            '0x40099c:\tcdqe\t': 1,
            '0x40099e:\tlea\trdx, [rax + 1]': 1,
            '0x4009a2:\tmov\trax, rdx': 1,
            '0x4009a5:\tshl\trax, 4': 1,
            '0x4009a9:\tsub\trax, rdx': 1,
            '0x4009ac:\tadd\trax, rax': 1,
            '0x4009af:\tmov\trdx, rax': 1,
            '0x4009b2:\tmov\trax, qword ptr [rbp - 0x18]': 1,
            '0x4009b6:\tadd\trdx, rax': 1,
            '0x4009b9:\tmov\teax, dword ptr [rbp - 0x20]': 1,
            '0x4009bc:\tsub\teax, 1': 1,
            '0x4009bf:\tcdqe\t': 1,
            '0x4009c1:\tmovzx\teax, byte ptr [rdx + rax]': 1,
            '0x4009c5:\ttest\tal, al': 1,
            '0x4009c7:\tje\t0x4009cd': 1,
            '0x4009d3:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
            '0x4009d6:\tcdqe\t': 1,
            '0x4009d8:\tlea\trdx, [rax + 1]': 1,
            '0x4009dc:\tmov\trax, rdx': 1,
            '0x4009df:\tshl\trax, 4': 1,
            '0x4009e3:\tsub\trax, rdx': 1,
            '0x4009e6:\tadd\trax, rax': 1,
            '0x4009e9:\tmov\trdx, rax': 1,
            '0x4009ec:\tmov\trax, qword ptr [rbp - 0x18]': 1,
            '0x4009f0:\tadd\trdx, rax': 1,
            '0x4009f3:\tmov\teax, dword ptr [rbp - 0x20]': 1,
            '0x4009f6:\tcdqe\t': 1,
            '0x4009f8:\tmovzx\teax, byte ptr [rdx + rax]': 1,
            '0x4009fc:\ttest\tal, al': 1,
            '0x4009fe:\tje\t0x400a04': 1,
            '0x400a04:\tcmp\tdword ptr [rbp - 0x1c], 0xd': 2,
            '0x400a08:\tjg\t0x400a44': 2,
            '0x4009c9:\tadd\tdword ptr [rbp - 4], 1': 1,
            '0x400a00:\tadd\tdword ptr [rbp - 4], 1': 1,
            '0x400a0a:\tcmp\tdword ptr [rbp - 0x20], 0x1c': 1,
            '0x400a0e:\tjg\t0x400a44': 1,
            '0x400a44:\tmov\teax, dword ptr [rbp - 4]': 2,
            '0x400a47:\tpop\trbp': 2,
            '0x400a48:\tret\t': 2,
            '0x400a10:\tmov\teax, dword ptr [rbp - 0x1c]': 1,
            '0x400a13:\tcdqe\t': 1,
            '0x400a15:\tlea\trdx, [rax + 1]': 1,
            '0x400a19:\tmov\trax, rdx': 1,
            '0x400a1c:\tshl\trax, 4': 1,
            '0x400a20:\tsub\trax, rdx': 1,
            '0x400a23:\tadd\trax, rax': 1,
            '0x400a26:\tmov\trdx, rax': 1,
            '0x400a29:\tmov\trax, qword ptr [rbp - 0x18]': 1,
            '0x400a2d:\tadd\trdx, rax': 1,
            '0x400a30:\tmov\teax, dword ptr [rbp - 0x20]': 1,
            '0x400a33:\tadd\teax, 1': 1,
            '0x400a36:\tcdqe\t': 1,
            '0x400a38:\tmovzx\teax, byte ptr [rdx + rax]': 1,
            '0x400a3c:\ttest\tal, al': 1,
            '0x400a3e:\tje\t0x400a44': 1,
            '0x400a40:\tadd\tdword ptr [rbp - 4], 1': 1,
            '0x400a49:\tpush\trbp': 1,
            '0x400a4a:\tmov\trbp, rsp': 1,
            '0x400a4d:\tsub\trsp, 0x3b0': 1,
            '0x400a54:\tmov\tedi, 0x3039': 1,
            '0x400a59:\tcall\t0x4005b0': 1,
            '0x400a5e:\tmov\tdword ptr [rbp - 4], 0': 1,
            '0x400a65:\tjmp\t0x400ace': 1,
            '0x400ace:\tcmp\tdword ptr [rbp - 4], 0xe': 2,
            '0x400ad2:\tjle\t0x400a67': 2,
            '0x400a67:\tmov\tdword ptr [rbp - 8], 0': 1,
            '0x400a6e:\tjmp\t0x400ac4': 1,
            '0x400ad4:\tlea\trax, [rbp - 0x1e0]': 1,
            '0x400adb:\tmov\trdi, rax': 1,
            '0x400ade:\tcall\t0x400771': 1,
            '0x400ac4:\tcmp\tdword ptr [rbp - 8], 0x1d': 2,
            '0x400ac8:\tjle\t0x400a70': 2,
            '0x400ae3:\tmov\tdword ptr [rbp - 0xc], 0': 1,
            '0x400aea:\tjmp\t0x400c37': 1,
            '0x400a70:\tcall\t0x4005c0': 1,
            '0x400aca:\tadd\tdword ptr [rbp - 4], 1': 1,
            '0x400c37:\tcmp\tdword ptr [rbp - 0xc], 0xe': 2,
            '0x400c3b:\tjle\t0x400aef': 2,
            '0x400a75:\tcvtsi2ss\txmm0, eax': 1,
            '0x400a79:\tmovss\txmm1, dword ptr [rip + 0x2e7]': 1,
            '0x400a81:\tdivss\txmm0, xmm1': 1,
            '0x400a85:\tmovaps\txmm1, xmm0': 1,
            '0x400a88:\tmovss\txmm0, dword ptr [rip + 0x2dc]': 1,
            '0x400a90:\tcomiss\txmm0, xmm1': 1,
            '0x400a93:\tseta\tal': 1,
            '0x400a96:\tmov\tesi, eax': 1,
            '0x400a98:\tmov\teax, dword ptr [rbp - 8]': 1,
            '0x400a9b:\tmovsxd\trcx, eax': 1,
            '0x400a9e:\tmov\teax, dword ptr [rbp - 4]': 1,
            '0x400aa1:\tmovsxd\trdx, eax': 1,
            '0x400aa4:\tmov\trax, rdx': 1,
            '0x400aa7:\tshl\trax, 4': 1,
            '0x400aab:\tsub\trax, rdx': 1,
            '0x400aae:\tadd\trax, rax': 1,
            '0x400ab1:\tadd\trax, rbp': 1,
            '0x400ab4:\tadd\trax, rcx': 1,
            '0x400ab7:\tsub\trax, 0x1e0': 1,
            '0x400abd:\tmov\tbyte ptr [rax], sil': 1,
            '0x400ac0:\tadd\tdword ptr [rbp - 8], 1': 1,
            '0x400aef:\tmov\tdword ptr [rbp - 0x10], 0': 1,
            '0x400af6:\tjmp\t0x400c29': 1,
            '0x400c41:\tmov\tdword ptr [rbp - 0x14], 0': 1,
            '0x400c48:\tjmp\t0x400cb0': 1,
            '0x400c29:\tcmp\tdword ptr [rbp - 0x10], 0x1d': 3,
            '0x400c2d:\tjle\t0x400afb': 3,
            '0x400cb0:\tcmp\tdword ptr [rbp - 0x14], 0xe': 2,
            '0x400cb4:\tjle\t0x400c4a': 2,
            '0x400afb:\tmov\tedx, dword ptr [rbp - 0x10]': 1,
            '0x400afe:\tmov\tecx, dword ptr [rbp - 0xc]': 1,
            '0x400b01:\tlea\trax, [rbp - 0x1e0]': 1,
            '0x400b08:\tmov\tesi, ecx': 1,
            '0x400b0a:\tmov\trdi, rax': 1,
            '0x400b0d:\tcall\t0x400859': 1,
            '0x400c33:\tadd\tdword ptr [rbp - 0xc], 1': 1,
            '0x400c4a:\tmov\tdword ptr [rbp - 0x18], 0': 1,
            '0x400c51:\tjmp\t0x400ca6': 1,
            '0x400cb6:\tmov\tedi, 0x1f4': 1,
            '0x400cbb:\tcall\t0x4006b6': 1,
            '0x400b12:\tmov\tdword ptr [rbp - 0x1c], eax': 1,
            '0x400b15:\tmov\teax, dword ptr [rbp - 0x10]': 1,
            '0x400b18:\tmovsxd\trcx, eax': 1,
            '0x400b1b:\tmov\teax, dword ptr [rbp - 0xc]': 1,
            '0x400b1e:\tmovsxd\trdx, eax': 1,
            '0x400b21:\tmov\trax, rdx': 1,
            '0x400b24:\tshl\trax, 4': 1,
            '0x400b28:\tsub\trax, rdx': 1,
            '0x400b2b:\tadd\trax, rax': 1,
            '0x400b2e:\tadd\trax, rbp': 1,
            '0x400b31:\tadd\trax, rcx': 1,
            '0x400b34:\tsub\trax, 0x1e0': 1,
            '0x400b3a:\tmovzx\teax, byte ptr [rax]': 1,
            '0x400b3d:\ttest\tal, al': 1,
            '0x400b3f:\tje\t0x400b7a': 1,
            '0x400ca6:\tcmp\tdword ptr [rbp - 0x18], 0x1d': 2,
            '0x400caa:\tjle\t0x400c53': 2,
            '0x400cc0:\tjmp\t0x400ad4': 1,
            '0x400b7a:\tmov\teax, dword ptr [rbp - 0x10]': 1,
            '0x400b7d:\tmovsxd\trcx, eax': 1,
            '0x400b80:\tmov\teax, dword ptr [rbp - 0xc]': 1,
            '0x400b83:\tmovsxd\trdx, eax': 1,
            '0x400b86:\tmov\trax, rdx': 1,
            '0x400b89:\tshl\trax, 4': 1,
            '0x400b8d:\tsub\trax, rdx': 1,
            '0x400b90:\tadd\trax, rax': 1,
            '0x400b93:\tadd\trax, rbp': 1,
            '0x400b96:\tadd\trax, rcx': 1,
            '0x400b99:\tsub\trax, 0x1e0': 1,
            '0x400b9f:\tmovzx\teax, byte ptr [rax]': 1,
            '0x400ba2:\ttest\tal, al': 1,
            '0x400ba4:\tjne\t0x400bd6': 1,
            '0x400b41:\tcmp\tdword ptr [rbp - 0x1c], 1': 1,
            '0x400b45:\tjle\t0x400b4d': 1,
            '0x400c53:\tmov\teax, dword ptr [rbp - 0x18]': 1,
            '0x400c56:\tmovsxd\trcx, eax': 1,
            '0x400c59:\tmov\teax, dword ptr [rbp - 0x14]': 1,
            '0x400c5c:\tmovsxd\trdx, eax': 1,
            '0x400c5f:\tmov\trax, rdx': 1,
            '0x400c62:\tshl\trax, 4': 1,
            '0x400c66:\tsub\trax, rdx': 1,
            '0x400c69:\tadd\trax, rax': 1,
            '0x400c6c:\tadd\trax, rbp': 1,
            '0x400c6f:\tadd\trax, rcx': 1,
            '0x400c72:\tsub\trax, 0x3b0': 1,
            '0x400c78:\tmovzx\tecx, byte ptr [rax]': 1,
            '0x400c7b:\tmov\teax, dword ptr [rbp - 0x18]': 1,
            '0x400c7e:\tmovsxd\trsi, eax': 1,
            '0x400c81:\tmov\teax, dword ptr [rbp - 0x14]': 1,
            '0x400c84:\tmovsxd\trdx, eax': 1,
            '0x400c87:\tmov\trax, rdx': 1,
            '0x400c8a:\tshl\trax, 4': 1,
            '0x400c8e:\tsub\trax, rdx': 1,
            '0x400c91:\tadd\trax, rax': 1,
            '0x400c94:\tadd\trax, rbp': 1,
            '0x400c97:\tadd\trax, rsi': 1,
            '0x400c9a:\tsub\trax, 0x1e0': 1,
            '0x400ca0:\tmov\tbyte ptr [rax], cl': 1,
            '0x400ca2:\tadd\tdword ptr [rbp - 0x18], 1': 1,
            '0x400cac:\tadd\tdword ptr [rbp - 0x14], 1': 1,
            '0x400ba6:\tcmp\tdword ptr [rbp - 0x1c], 3': 1,
            '0x400baa:\tjne\t0x400bd6': 1,
            '0x400bd6:\tmov\teax, dword ptr [rbp - 0x10]': 1,
            '0x400bd9:\tmovsxd\trcx, eax': 1,
            '0x400bdc:\tmov\teax, dword ptr [rbp - 0xc]': 1,
            '0x400bdf:\tmovsxd\trdx, eax': 1,
            '0x400be2:\tmov\trax, rdx': 1,
            '0x400be5:\tshl\trax, 4': 1,
            '0x400be9:\tsub\trax, rdx': 1,
            '0x400bec:\tadd\trax, rax': 1,
            '0x400bef:\tadd\trax, rbp': 1,
            '0x400bf2:\tadd\trax, rcx': 1,
            '0x400bf5:\tsub\trax, 0x1e0': 1,
            '0x400bfb:\tmovzx\tecx, byte ptr [rax]': 1,
            '0x400bfe:\tmov\teax, dword ptr [rbp - 0x10]': 1,
            '0x400c01:\tmovsxd\trsi, eax': 1,
            '0x400c04:\tmov\teax, dword ptr [rbp - 0xc]': 1,
            '0x400c07:\tmovsxd\trdx, eax': 1,
            '0x400c0a:\tmov\trax, rdx': 1,
            '0x400c0d:\tshl\trax, 4': 1,
            '0x400c11:\tsub\trax, rdx': 1,
            '0x400c14:\tadd\trax, rax': 1,
            '0x400c17:\tadd\trax, rbp': 1,
            '0x400c1a:\tadd\trax, rsi': 1,
            '0x400c1d:\tsub\trax, 0x3b0': 1,
            '0x400c23:\tmov\tbyte ptr [rax], cl': 1,
            '0x400c25:\tadd\tdword ptr [rbp - 0x10], 1': 2,
            '0x400b4d:\tmov\teax, dword ptr [rbp - 0x10]': 1,
            '0x400b50:\tmovsxd\trcx, eax': 1,
            '0x400b53:\tmov\teax, dword ptr [rbp - 0xc]': 1,
            '0x400b56:\tmovsxd\trdx, eax': 1,
            '0x400b59:\tmov\trax, rdx': 1,
            '0x400b5c:\tshl\trax, 4': 1,
            '0x400b60:\tsub\trax, rdx': 1,
            '0x400b63:\tadd\trax, rax': 1,
            '0x400b66:\tadd\trax, rbp': 1,
            '0x400b69:\tadd\trax, rcx': 1,
            '0x400b6c:\tsub\trax, 0x3b0': 1,
            '0x400b72:\tmov\tbyte ptr [rax], 0': 1,
            '0x400b75:\tjmp\t0x400c25': 1,
            '0x400b47:\tcmp\tdword ptr [rbp - 0x1c], 3': 1,
            '0x400b4b:\tjle\t0x400b7a': 1,
            '0x400bac:\tmov\teax, dword ptr [rbp - 0x10]': 1,
            '0x400baf:\tmovsxd\trcx, eax': 1,
            '0x400bb2:\tmov\teax, dword ptr [rbp - 0xc]': 1,
            '0x400bb5:\tmovsxd\trdx, eax': 1,
            '0x400bb8:\tmov\trax, rdx': 1,
            '0x400bbb:\tshl\trax, 4': 1,
            '0x400bbf:\tsub\trax, rdx': 1,
            '0x400bc2:\tadd\trax, rax': 1,
            '0x400bc5:\tadd\trax, rbp': 1,
            '0x400bc8:\tadd\trax, rcx': 1,
            '0x400bcb:\tsub\trax, 0x3b0': 1,
            '0x400bd1:\tmov\tbyte ptr [rax], 1': 1,
            '0x400bd4:\tjmp\t0x400c25': 1,
            '0x400cc5:\tnop\tword ptr cs:[rax + rax]': 1,
            '0x400ccf:\tnop\t': 1,
            '0x400cd0:\tendbr64\t': 1,
            '0x400cd4:\tpush\tr15': 1,
            '0x400cd6:\tmov\tr15, rdx': 1,
            '0x400cd9:\tpush\tr14': 1,
            '0x400cdb:\tmov\tr14, rsi': 1,
            '0x400cde:\tpush\tr13': 1,
            '0x400ce0:\tmov\tr13d, edi': 1,
            '0x400ce3:\tpush\tr12': 1,
            '0x400ce5:\tlea\tr12, [rip + 0x201114]': 1,
            '0x400cec:\tpush\trbp': 1,
            '0x400ced:\tlea\trbp, [rip + 0x201114]': 1,
            '0x400cf4:\tpush\trbx': 1,
            '0x400cf5:\tsub\trbp, r12': 1,
            '0x400cf8:\tsub\trsp, 8': 1,
            '0x400cfc:\tcall\t0x400550': 1,
            '0x400d01:\tsar\trbp, 3': 1,
            '0x400d05:\tje\t0x400d26': 1,
            '0x400d26:\tadd\trsp, 8': 1,
            '0x400d2a:\tpop\trbx': 1,
            '0x400d2b:\tpop\trbp': 1,
            '0x400d2c:\tpop\tr12': 1,
            '0x400d2e:\tpop\tr13': 1,
            '0x400d30:\tpop\tr14': 1,
            '0x400d32:\tpop\tr15': 1,
            '0x400d34:\tret\t': 1,
            '0x400d07:\txor\tebx, ebx': 1,
            '0x400d09:\tnop\tdword ptr [rax]': 1,
            '0x400d10:\tmov\trdx, r15': 2,
            '0x400d13:\tmov\trsi, r14': 2,
            '0x400d16:\tmov\tedi, r13d': 2,
            '0x400d19:\tcall\tqword ptr [r12 + rbx*8]': 2,
            '0x400d1d:\tadd\trbx, 1': 1,
            '0x400d21:\tcmp\trbp, rbx': 1,
            '0x400d24:\tjne\t0x400d10': 1,
            '0x400d35:\tnop\tword ptr cs:[rax + rax]': 1,
            '0x400d40:\tendbr64\t': 1,
            '0x400d44:\tret\t': 1,
            '0x400d48:\tendbr64\t': 1,
            '0x400d4c:\tsub\trsp, 8': 1,
            '0x400d50:\tadd\trsp, 8': 1,
            '0x400d54:\tret\t': 1,
        },
    }

    # Adding functions to the cfg
    if build_level in ['cfg']:
        __auto_cfg.add_function(*__auto_functions.values())
    else:
        __auto_cfg.functions = list(__auto_functions.values())
        __auto_cfg.blocks = list(__auto_blocks.values())

    return {
        'blocks': __auto_blocks,
        'file': os.path.basename(__file__),
        'inputs': _load_angr(),
        'cfg': __auto_cfg,
        'functions': __auto_functions,
        'expected': expected,
    }


def _load_angr():
    """Loads the Conways GOL example using angr"""
    angr = get_module('angr', raise_err=True)
    project = angr.Project(os.path.join(os.path.dirname(__file__), 'gol.compiled'), auto_load_libs=False)
    cfg = project.analyses.CFGFast()
    return [project, cfg]