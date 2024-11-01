"""A manually made tiny CFG with a bunch of weird connections and whatnot"""
import os
from bincfg import CFGFunction, CFGBasicBlock, CFG, CFGEdge, EdgeType
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
        4096: func_type(parent_cfg=__auto_cfg, address=4096, name='func_name', is_extern_function=False, metadata={'func-level': [12412, 'metadata']}),
        4160: func_type(parent_cfg=__auto_cfg, address=4160, name='__UNNAMED_FUNC_4160', is_extern_function=False, metadata={}),
        4655: func_type(parent_cfg=__auto_cfg, address=4655, name='main', is_extern_function=False, metadata={'more_fun': 'c metadata'}),
    }

    # Building basic blocks. Dictionary maps integer address to CFGBasicBlock() object
    __auto_blocks = {
        4096: CFGBasicBlock(parent_function=__auto_functions[4096], address=4096, asm_memory_addresses=[4096, 4100, 4104, 4111, 4114], metadata={'random': 'metadata'}, asm_lines=[
            'nop',
            'sub    rsp, 0x08',
            'mov    rax, qword ds:[rip + 0x0000000000002fd9<12249,absolute=0x0000000000003fe8>]',
            'test   rax, rax',
            'je     0x0000000000001016<4118>',
        ]),
        4116: CFGBasicBlock(parent_function=__auto_functions[4096], address=4116, asm_memory_addresses=[4116], metadata={}, asm_lines=[
            'call   rax',
        ]),
        4118: CFGBasicBlock(parent_function=__auto_functions[4096], address=4118, asm_memory_addresses=[4118, 4122], metadata={}, asm_lines=[
            'add    rsp, 0x08',
            'ret',
        ]),
        4160: CFGBasicBlock(parent_function=__auto_functions[4160], address=4160, asm_memory_addresses=[4160, 4164], metadata={}, asm_lines=[
            'nop',
            'jmp    qword ds:[rip + 0x0000000000002fad<12205,absolute=0x0000000000003ff8>]',
        ]),
        4655: CFGBasicBlock(parent_function=__auto_functions[4655], address=4655, asm_memory_addresses=[4655, 4659, 4660, 4663, 4667, 4670, 4674, 4678, 4682, 4686, 4690, 4695, 4698], metadata={}, asm_lines=[
            'nop',
            'push   rbp',
            'mov    rbp, rsp',
            'sub    rsp, 0x20',
            'mov    dword ds:[rbp + 0xec<-20>], edi',
            'mov    qword ds:[rbp + 0xe0<-32>], rsi',
            'mov    rax, qword ds:[rbp + 0xe0<-32>]',
            'mov    rax, qword ds:[rax + 0x08]',
            'mov    qword ds:[rbp + 0xf8<-8>], rax',
            'mov    rax, qword ds:[rbp + 0xf8<-8>]',
            'mov    esi, 0x0000003a',
            'mov    rdi, rax',
            'call   0x0000000000001149<4425,(func)search_char>',
        ]),
        4703: CFGBasicBlock(parent_function=__auto_functions[4655], address=4703, asm_memory_addresses=[4703, 4706, 4709, 4711, 4718, 4723], metadata={'another': 'test', 'set': 'of dict', 'values': 10}, asm_lines=[
            'mov    dword ds:[rbp + 0xf0<-16>], eax',
            'mov    eax, dword ds:[rbp + 0xf0<-16>]',
            'mov    esi, eax',
            'lea    rdi, [rip + 0x0000000000000d96<3478,absolute=0x0000000000002004>]',
            'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'call   0x0000000000001050<4176>',
        ]),
        4728: CFGBasicBlock(parent_function=__auto_functions[4655], address=4728, asm_memory_addresses=[4728, 4732, 4735, 4742, 4747], metadata={}, asm_lines=[
            'mov    rax, qword ds:[rbp + 0xf8<-8>]',
            'mov    rsi, rax',
            'lea    rdi, [rip + 0x0000000000000d8d<3469,absolute=0x0000000000002013>]',
            'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'call   0x0000000000001050<4176>',
        ]),
        4752: CFGBasicBlock(parent_function=__auto_functions[4655], address=4752, asm_memory_addresses=[4752, 4756, 4761, 4764], metadata={}, asm_lines=[
            'mov    rax, qword ds:[rbp + 0xf8<-8>]',
            'mov    esi, 0x00007ab7<31415>',
            'mov    rdi, rax',
            'call   0x00000000000011a2<4514,(func)mutate>',
        ]),
        4769: CFGBasicBlock(parent_function=__auto_functions[4655], address=4769, asm_memory_addresses=[4769, 4772, 4776, 4779, 4786, 4791], metadata={'test_val': 10}, asm_lines=[
            'mov    dword ds:[rbp + 0xf4<-12>], eax',
            'mov    rax, qword ds:[rbp + 0xf8<-8>]',
            'mov    rsi, rax',
            'lea    rdi, [rip + 0x0000000000000d6d<3437,absolute=0x000000000000201f>]',
            'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'call   0x0000000000001050<4176>',
        ]),
        4796: CFGBasicBlock(parent_function=__auto_functions[4655], address=4796, asm_memory_addresses=[4796, 4801, 4802], metadata={}, asm_lines=[
            'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>',
            'leave',
            'ret',
        ]),
    }

    # Building all edges
    __auto_blocks[4096].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4096], to_block=__auto_blocks[4118], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4096], to_block=__auto_blocks[4116], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4096], to_block=__auto_blocks[4655], edge_type='normal'),
        CFGEdge(from_block=__auto_blocks[4096], to_block=__auto_blocks[4655], edge_type='function_call'),
        CFGEdge(from_block=__auto_blocks[4096], to_block=__auto_blocks[4728], edge_type='normal'),
        CFGEdge(from_block=__auto_blocks[4096], to_block=__auto_blocks[4769], edge_type='function_call'),
        CFGEdge(from_block=__auto_blocks[4096], to_block=__auto_blocks[4752], edge_type='function_call'),
        CFGEdge(from_block=__auto_blocks[4096], to_block=__auto_blocks[4796], edge_type='normal'),
    ])

    __auto_blocks[4116].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4116], to_block=__auto_blocks[4118], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4118].edges_out = set([
        
    ])

    __auto_blocks[4160].edges_out = set([
        
    ])

    __auto_blocks[4655].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4655], to_block=__auto_blocks[4703], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4703].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4160], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4728], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4752], edge_type='normal'),
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4752], edge_type='function_call'),
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4096], edge_type='normal'),
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4096], edge_type='function_call'),
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4769], edge_type='function_call'),
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4118], edge_type='normal'),
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4116], edge_type='normal'),
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4703], edge_type='normal'),
        CFGEdge(from_block=__auto_blocks[4703], to_block=__auto_blocks[4703], edge_type='function_call'),
    ])

    __auto_blocks[4728].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4728], to_block=__auto_blocks[4160], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4728], to_block=__auto_blocks[4752], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4752].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4752], to_block=__auto_blocks[4655], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[4752], to_block=__auto_blocks[4769], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[4769].edges_out = set([
        CFGEdge(from_block=__auto_blocks[4769], to_block=__auto_blocks[4796], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[4769], to_block=__auto_blocks[4160], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[4796].edges_out = set([
        
    ])

    # Set the edges_in on the blocks
    for b in __auto_blocks.values():
        for e in b.edges_out:
            e.to_block.edges_in.add(CFGEdge(b, e.to_block, e.edge_type))

    # Adding basic blocks to their associated functions
    __auto_functions[4096].blocks = [
        __auto_blocks[4096],
        __auto_blocks[4116],
        __auto_blocks[4118],
    ]

    __auto_functions[4160].blocks = [
        __auto_blocks[4160],
    ]

    __auto_functions[4655].blocks = [
        __auto_blocks[4655],
        __auto_blocks[4703],
        __auto_blocks[4728],
        __auto_blocks[4752],
        __auto_blocks[4769],
        __auto_blocks[4796],
    ]

    expected = {
        'sorted_func_order': [4096, 4160, 4655],
        'sorted_block_order': [4096, 4116, 4118, 4160, 4655, 4703, 4728, 4752, 4769, 4796],
        'architecture': 'x86',
        'num_blocks': {4096: 3, 4160: 1, 4655: 6},
        'num_asm_lines_per_block': {4096: 5, 4116: 1, 4118: 2, 4160: 2, 4655: 13, 4703: 6, 4728: 5, 4752: 4, 4769: 6, 4796: 3},
        'num_asm_lines_per_function': {4096: 8, 4160: 2, 4655: 37},
        'num_functions': 3,
        'is_root_function': {4096: False, 4160: False, 4655: False},
        'is_recursive': {4096: False, 4160: False, 4655: True},
        'is_extern_function': {4096: False, 4160: False, 4655: False},
        'is_intern_function': {4096: True, 4160: True, 4655: True},
        'function_entry_block': {4096: 4096, 4160: 4160, 4655: 4655},
        'called_by': {4096: {4703}, 4160: {4728, 4769, 4703}, 4655: {4096, 4752, 4703}},
        'function_hashes': {4096: 273089078936644660, 4160: 760806268535495279, 4655: 27734138268289825},
        'block_hashes': {4096: 6868004256956990, 4116: 1511865352607541905, 4118: 965672024675359758, 4160: 1929242761529262007, 4655: 2147071593812274926, 4703: 785225323957816538, 4728: 289649005325774078, 4752: 308277429646246485, 4769: 114050936081365769, 4796: 944032198737404855},
        'cfg_hash': 696619345802459356,
        'memcfg_hashes': {'base_norm-op': 1201854069018640920, 'base_norm-inst': 1853273498353907935, 'innereye-op': 1249022082061231475, 'innereye-inst': 1327838857736745439, 'safe-op': 1788614906753244308, 'safe-inst': 840813117365618003, 'deepbindiff-op': 988240775412090641, 'deepbindiff-inst': 766307620237373259, 'deepsemantic-op': 1762155316900906719, 'deepsemantic-inst': 348702077813181361, 'compressed_stats-op': 1707661846003147430, 'compressed_stats-inst': 640351797697781201, 'hpcdata-op': 766455179519236919, 'hpcdata-inst': 179566051100344238},
        'metadata': {'some': 'cfg-level', 134: ('metadata', True, None), (1, 2, 3): 'apples'},
        'block_metadatas': {4096: {'random': 'metadata'}, 4116: {}, 4118: {}, 4160: {}, 4655: {}, 4703: {'another': 'test', 'set': 'of dict', 'values': 10}, 4728: {}, 4752: {}, 4769: {'test_val': 10}, 4796: {}},
        'function_metadatas': {4096: {'func-level': [12412, 'metadata']}, 4160: {}, 4655: {'more_fun': 'c metadata'}},
        'asm_counts_per_block': {
            4096: {'nop': 1, 'sub    rsp, 0x08': 1, 'mov    rax, qword ds:[rip + 0x0000000000002fd9<12249,absolute=0x0000000000003fe8>]': 1, 'test   rax, rax': 1, 'je     0x0000000000001016<4118>': 1},
            4116: {'call   rax': 1},
            4118: {'add    rsp, 0x08': 1, 'ret': 1},
            4160: {'nop': 1, 'jmp    qword ds:[rip + 0x0000000000002fad<12205,absolute=0x0000000000003ff8>]': 1},
            4655: {'nop': 1, 'push   rbp': 1, 'mov    rbp, rsp': 1, 'sub    rsp, 0x20': 1, 'mov    dword ds:[rbp + 0xec<-20>], edi': 1, 'mov    qword ds:[rbp + 0xe0<-32>], rsi': 1, 'mov    rax, qword ds:[rbp + 0xe0<-32>]': 1, 'mov    rax, qword ds:[rax + 0x08]': 1, 'mov    qword ds:[rbp + 0xf8<-8>], rax': 1, 'mov    rax, qword ds:[rbp + 0xf8<-8>]': 1, 'mov    esi, 0x0000003a': 1, 'mov    rdi, rax': 1, 'call   0x0000000000001149<4425,(func)search_char>': 1},
            4703: {'mov    dword ds:[rbp + 0xf0<-16>], eax': 1, 'mov    eax, dword ds:[rbp + 0xf0<-16>]': 1, 'mov    esi, eax': 1, 'lea    rdi, [rip + 0x0000000000000d96<3478,absolute=0x0000000000002004>]': 1, 'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'call   0x0000000000001050<4176>': 1},
            4728: {'mov    rax, qword ds:[rbp + 0xf8<-8>]': 1, 'mov    rsi, rax': 1, 'lea    rdi, [rip + 0x0000000000000d8d<3469,absolute=0x0000000000002013>]': 1, 'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'call   0x0000000000001050<4176>': 1},
            4752: {'mov    rax, qword ds:[rbp + 0xf8<-8>]': 1, 'mov    esi, 0x00007ab7<31415>': 1, 'mov    rdi, rax': 1, 'call   0x00000000000011a2<4514,(func)mutate>': 1},
            4769: {'mov    dword ds:[rbp + 0xf4<-12>], eax': 1, 'mov    rax, qword ds:[rbp + 0xf8<-8>]': 1, 'mov    rsi, rax': 1, 'lea    rdi, [rip + 0x0000000000000d6d<3437,absolute=0x000000000000201f>]': 1, 'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'call   0x0000000000001050<4176>': 1},
            4796: {'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 1, 'leave': 1, 'ret': 1},
        },
        'asm_counts_per_function': {
            4096: {
                'nop': 1,
                'sub    rsp, 0x08': 1,
                'mov    rax, qword ds:[rip + 0x0000000000002fd9<12249,absolute=0x0000000000003fe8>]': 1,
                'test   rax, rax': 1,
                'je     0x0000000000001016<4118>': 1,
                'call   rax': 1,
                'add    rsp, 0x08': 1,
                'ret': 1,
            },
            4160: {
                'nop': 1,
                'jmp    qword ds:[rip + 0x0000000000002fad<12205,absolute=0x0000000000003ff8>]': 1,
            },
            4655: {
                'nop': 1,
                'push   rbp': 1,
                'mov    rbp, rsp': 1,
                'sub    rsp, 0x20': 1,
                'mov    dword ds:[rbp + 0xec<-20>], edi': 1,
                'mov    qword ds:[rbp + 0xe0<-32>], rsi': 1,
                'mov    rax, qword ds:[rbp + 0xe0<-32>]': 1,
                'mov    rax, qword ds:[rax + 0x08]': 1,
                'mov    qword ds:[rbp + 0xf8<-8>], rax': 1,
                'mov    rax, qword ds:[rbp + 0xf8<-8>]': 4,
                'mov    esi, 0x0000003a': 1,
                'mov    rdi, rax': 2,
                'call   0x0000000000001149<4425,(func)search_char>': 1,
                'mov    dword ds:[rbp + 0xf0<-16>], eax': 1,
                'mov    eax, dword ds:[rbp + 0xf0<-16>]': 1,
                'mov    esi, eax': 1,
                'lea    rdi, [rip + 0x0000000000000d96<3478,absolute=0x0000000000002004>]': 1,
                'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 4,
                'call   0x0000000000001050<4176>': 3,
                'mov    rsi, rax': 2,
                'lea    rdi, [rip + 0x0000000000000d8d<3469,absolute=0x0000000000002013>]': 1,
                'mov    esi, 0x00007ab7<31415>': 1,
                'call   0x00000000000011a2<4514,(func)mutate>': 1,
                'mov    dword ds:[rbp + 0xf4<-12>], eax': 1,
                'lea    rdi, [rip + 0x0000000000000d6d<3437,absolute=0x000000000000201f>]': 1,
                'leave': 1,
                'ret': 1,
            },
        },
        'asm_counts': {
            'nop': 3,
            'sub    rsp, 0x08': 1,
            'mov    rax, qword ds:[rip + 0x0000000000002fd9<12249,absolute=0x0000000000003fe8>]': 1,
            'test   rax, rax': 1,
            'je     0x0000000000001016<4118>': 1,
            'call   rax': 1,
            'add    rsp, 0x08': 1,
            'ret': 2,
            'jmp    qword ds:[rip + 0x0000000000002fad<12205,absolute=0x0000000000003ff8>]': 1,
            'push   rbp': 1,
            'mov    rbp, rsp': 1,
            'sub    rsp, 0x20': 1,
            'mov    dword ds:[rbp + 0xec<-20>], edi': 1,
            'mov    qword ds:[rbp + 0xe0<-32>], rsi': 1,
            'mov    rax, qword ds:[rbp + 0xe0<-32>]': 1,
            'mov    rax, qword ds:[rax + 0x08]': 1,
            'mov    qword ds:[rbp + 0xf8<-8>], rax': 1,
            'mov    rax, qword ds:[rbp + 0xf8<-8>]': 4,
            'mov    esi, 0x0000003a': 1,
            'mov    rdi, rax': 2,
            'call   0x0000000000001149<4425,(func)search_char>': 1,
            'mov    dword ds:[rbp + 0xf0<-16>], eax': 1,
            'mov    eax, dword ds:[rbp + 0xf0<-16>]': 1,
            'mov    esi, eax': 1,
            'lea    rdi, [rip + 0x0000000000000d96<3478,absolute=0x0000000000002004>]': 1,
            'mov    eax, 0x00000000<(func)__cxa_finalize@@GLIBC_2.2.5>': 4,
            'call   0x0000000000001050<4176>': 3,
            'mov    rsi, rax': 2,
            'lea    rdi, [rip + 0x0000000000000d8d<3469,absolute=0x0000000000002013>]': 1,
            'mov    esi, 0x00007ab7<31415>': 1,
            'call   0x00000000000011a2<4514,(func)mutate>': 1,
            'mov    dword ds:[rbp + 0xf4<-12>], eax': 1,
            'lea    rdi, [rip + 0x0000000000000d6d<3437,absolute=0x000000000000201f>]': 1,
            'leave': 1,
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
        'inputs': [],
        'cfg': __auto_cfg,
        'functions': __auto_functions,
        'expected': expected,
    }

