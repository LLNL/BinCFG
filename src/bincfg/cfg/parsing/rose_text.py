"""Parsing ROSE text output files"""

from ..cfg_function import CFGFunction
from ..cfg_edge import EdgeType, CFGEdge
from ...utils import get_address
from .parse_utils import is_extern_func_name, create_basic_block


def parse_rose_txt(cfg, lines):
    """Reads input as a .txt file

    Args:
        cfg (CFG): an empty/loading CFG() object
        lines (str, Iterable[str], TextIO): the data to parse. Can be a string (which will be split on newlines to get each
            individual line), a list of string (each element will be considered one line), or an open file to call
            `.readlines()` on

    Raises:
        CFGParseError: when file does not fit expected format
    """
    if isinstance(lines, str):
        lines = lines.split('\n')
    elif hasattr(lines, 'readlines') and callable(lines, 'readlines'):
        lines = lines.readlines()

    try:
        # Clean up the lines a bit
        lines = [l.strip() for l in lines if l.strip()]
    except:
        raise TypeError("Could not parse rose txt input of type: '%s'" % type(lines).__name__)

    cfg.metadata['file_type'] = 'txt'
    
    # Make the dictionary of the current blocks
    curr_blocks = {}
    funcs = []
    
    # Go through lines finding each function
    curr_func_lines = [lines[0]]
    for line in lines[1:]:
        
        # Make the next function with the current list of lines
        if line.startswith('function 0x'):
            funcs.append(_parse_txt_function(cfg, curr_func_lines, curr_blocks))
            curr_func_lines = [line]
        else:
            curr_func_lines.append(line)
    
    funcs.append(_parse_txt_function(cfg, curr_func_lines, curr_blocks))
    cfg.add_function(*funcs)


def _parse_txt_function(cfg, func_lines, curr_blocks):
    """Parses the function lines from a rose txt file into a ``CFGFunction``, and returns the function

    Args:
        cfg (CFG): the ``CFG`` to which this function would belong
        func_lines (List[str]): list of string lines from file to parse for this function
        curr_blocks (Dict[int, CFGBasicBlock]): a dictionary mapping basic block addresses to ``CFGBasicBlock`` objects. 
            We need this to create new basic blocks on the fly in order to make ``CFGEdge``'s work properly

    Returns:
        CFGFunction: the cfg function
    """
    # Create the CFGFunction() object with its parent_cfg, name (while removing quotes from rose text), and is_extern_func
    _, address, *func_name_lines = func_lines[0].split(" ")
    name = ''.join(func_name_lines)[1:-1] if func_name_lines else None
    func = CFGFunction(parent_cfg=cfg, address=get_address(address), name=name, is_extern_function=is_extern_func_name(name))

    # Build up every basic block
    curr_block_lines = [func_lines[1]]
    for line in func_lines[2:]:
        # Make the next block
        if line.startswith("B"):
            # Check if this is the first basic block and has the same starting address as the function
            _parse_txt_block(func, curr_block_lines, curr_blocks)
            curr_block_lines = [line]
        else:
            curr_block_lines.append(line)
    
    # Add in final block
    _parse_txt_block(func, curr_block_lines, curr_blocks)

    return func


def _parse_txt_block(func, block_lines, curr_blocks):
    """Parses the incoming block lines from a rose text file, and appends it to func's blocks

    Args:
        func (CFGFunction): the function to which this basic block belongs
        block_lines (List[str]): list of string block lines to build this block from
        curr_blocks (Dict[int, CFGBasicBlock]): a dictionary mapping basic block addresses to ``CFGBasicBlock`` objects. 
            We need this to create new basic blocks on the fly in order to make ``CFGEdge``'s work properly

    Raises:
        ValueError: on an unknown edge line
    """

    # Parse the block name and address, and check if it is a function entry point
    *_, block_address = block_lines[0].rpartition(" ")
    address = func.address if block_address[0] == 'p' else block_address[:-1] if block_address[-1] == ':' else block_address

    block = create_basic_block(curr_blocks, address=get_address(address), parent_function=func)
    
    # If block is None, then this block already exists in another function, no need to recreate it
    if block is None:
        return

    for line in block_lines[1:]:

        # If this line is to tell us that this is a function return block "block is a function return/call"
        if line[0] == 'b':
            continue
        
        # This is an assembly line. Add the memory address and string line as a tuple
        # IMPORTANT: do this before the " edge " detection in case of string literals in rose <> info
        elif line.startswith("0x"):
            address, _, asm_line = line.partition(": ")
            block.asm_lines.append(asm_line.strip())
            block.asm_memory_addresses.append(get_address(address))
        
        # Currently just ignoring the 'also_owned_by' for now
        elif line[0] == 'a':
            #owned_by = line[23:].partition(" ")[0]
            #block.also_owned_by.add(int(owned_by, 16))
            pass
        
        # Otherwise this must be an edge line
        else:
            # Check for "function entry point", then lines using function names in quotes, then just normal address
            edge_addr = block.parent_function.address if line[-1] == 't' else \
                line.rpartition(' "')[0].rpartition(' ')[-1] if line[-1] == '"' else line.rpartition(' ')[-1]
            
            # edge_addr might already be an int from it's parent address
            if not isinstance(edge_addr, int):
                # Check for indeterminate/nonexistant edges. We ignore these, but check to see if this is a function return
                if edge_addr[0] != '0':
                    continue

                # Convert edge_addr to int
                edge_addr = int(edge_addr, 16)
            
            # Check for lines like "function call edge from/to", and "function return edge to" 
            if line[0] == 'f':
                if line[9] == 'c':
                    if line[19] == 't':
                        block.edges_out.add(CFGEdge(block, create_basic_block(curr_blocks, edge_addr), 
                            edge_type=EdgeType.FUNCTION_CALL))
            
            # Check for "call return edge to" or "normal edge to"
            elif (line[0] == 'c' and line[17] == 't') or (line[0] == 'n' and line[12] == 't'):
                block.edges_out.add(CFGEdge(block, create_basic_block(curr_blocks, edge_addr), edge_type=EdgeType.NORMAL))
            
            # Check to make sure this line is an 'edge from' line. Otherwise this is an unknown line, raise an error
            elif 'edge from' not in line:
                raise ValueError("Unknown edge line: %s" % repr(line))
    
    func.blocks.append(block)

