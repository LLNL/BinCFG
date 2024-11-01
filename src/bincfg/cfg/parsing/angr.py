"""Parsing angr outputs"""
from ..cfg_basic_block import CFGBasicBlock
from ..cfg_function import CFGFunction
from ..cfg_edge import EdgeType
from ...utils import get_module


def parse_angr(angr_obj, cfg):
    """Parses an angr object into a CFG
    
    Args:
        cfg (CFG): an empty/loading CFG() object
        data (Union[angr.project.Project, angr.analyses.cfg.cfg_fast.CFGFast]): angr project or pre-processed angr CFG
            (CFGFast or CFGEmulated). If it is a project, we analyze using CFGFast() with its default parameters
    """
    angr = get_module('angr', raise_err=True)

    cfg.metadata['file_type'] = 'angr'

    # Turn off all those annoying warnings
    import logging
    logging.getLogger('angr').setLevel('CRITICAL')
    logging.getLogger('angr.analyses').setLevel('CRITICAL')
    
    # If a project is passed, do default CFGFast() analysis
    if isinstance(angr_obj, angr.project.Project):
        angr_obj = angr_obj.analyses.CFGFast()
    
    # We have to normalize it first to clean up the CFG
    angr_obj.normalize()

    # Compute the edges out (since these don't seem to be a part of the basic blocks for some reason?)
    edges_out_map = {}
    for from_node, to_node, md in angr_obj.graph.edges(data=True):
        if md['jumpkind'] not in ['Ijk_Call', 'Ijk_FakeRet', 'Ijk_Ret', 'Ijk_Boring']:
            raise ValueError("Unknown jumpkind: %s" % repr(md['jumpkind']))
        et = EdgeType.FUNCTION_CALL if md['jumpkind'] in ['Ijk_Call'] else EdgeType.NORMAL
        edges_out_map.setdefault(from_node.addr, []).append((None, to_node.addr, et))

    # Build all of the function objects
    cfg.add_function(*[
        CFGFunction(address=func_addr, name=func.name, is_extern_function=func.is_plt or func.is_syscall, metadata={}, blocks=[
            CFGBasicBlock(
                address=block.addr, asm_memory_addresses=block.instruction_addrs, metadata={},
                asm_lines=[str(insn) for insn in block.capstone.insns],
                edges_out=edges_out_map[block.addr] if block.addr in edges_out_map else set(),
            ) for block in func.blocks
        ]) for func_addr, func in angr_obj.kb.functions.items()
    ], override=True)

    # TODO: Fix times when Angr, for some ungodly reason, outputs duplicate blocks. It seems to happen when a block
    #   jumps to a block that resides in a different function. Then, entirely duplicate functions (at least, from
    #   those jumps onward) are generated. Why would they not just have a jump to that block and leave it at that?
    #   Who knows...
    #
    # Once that is fixed, remove the 'override=True' kwargs from the .add_function() call above
    #
    # NOTE: this doesn't happen with the example GOL binary, but I did see it happen in some of the codeforces ones