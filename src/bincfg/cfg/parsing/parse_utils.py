"""Utilities for parsing CFG inputs"""
import re
from ..cfg_basic_block import CFGBasicBlock


# Regular expressions that denote an external function
EXTERN_FUNC_NAME_REGEXS = [re.compile(r'.*@plt'), re.compile(r'.*@.*[.]dll')]


# List of strings that are names for 'indeterminate' nodes
INDETERMINATE_NODE_NAMES = ['indeterminate', 'nonexisting']


def is_extern_func_name(name):
    """Returns True if name is an external function name, False otherwise
    
    Args:
        name (Union[str, None]): the name
    
    Returns:
        bool: True if name is an external function name, False otherwise
    """
    return name is not None and any(s.fullmatch(name) is not None for s in EXTERN_FUNC_NAME_REGEXS)


def create_basic_block(curr_blocks, address, **kwargs):
    """Checks if there is a basic block with the given address in `curr_blocks`, and if not, creates it. Returns the block

    If the block does exist, then any kwargs in ``kwargs`` will be updated in the CFGBasicBlock, unless that block already
    has a parent_func in which case None will be returned and no blocks will be updated

    Args:
        curr_blocks (Dict[int, CFGBasicBlock]): curr_blocks: a dictionary mapping basic block addresses to 
            ``CFGBasicBlock`` objects. We need this to create new basic blocks on the fly in order to make 
            ``CFGEdge``'s work properly
        address (int): the integer memory address of the new basic block
        kwargs (Any): extra kwargs to pass to ``CFGBasicBlock`` object creation, or to update an already existing 
            CFGBasicBlock

    Raises:
        ValueError: _description_

    Returns:
        CFGBasicBlock: _description_
    """
    if address not in curr_blocks:
        curr_blocks[address] = CFGBasicBlock(address=address, **kwargs)
    elif len(kwargs) == 0:
        return curr_blocks[address]
    elif curr_blocks[address].parent_function is not None:
        return None
    else:
        for k, v in kwargs.items():
            if k in ['parent_function', 'edges_in', 'edges_out', 'asm_lines', 'metadata', 'asm_memory_addresses']:
                setattr(curr_blocks[address], k, v)
            else:
                raise ValueError("Unknown basic block kwarg: %s" % repr(k))
    return curr_blocks[address]


class CFGParseError(Exception):
    """Exception that occurs during CFG input data parsing"""
    pass