"""Parsing ROSE graphviz files"""
import re
import html
from ..cfg_function import CFGFunction
from ..cfg_edge import CFGEdge, EdgeType
from ...utils import get_address
from .parse_utils import is_extern_func_name, create_basic_block, CFGParseError, INDETERMINATE_NODE_NAMES


DIGRAPH_START_STRINGS = ['digraph', 'graph', 'node', 'edge']

# Regex matches
FUNC_STR_MATCH = re.compile(r'label=".*\\""')
FUNC_STR_MATCH_NO_NAME = re.compile(r'label="function 0x[0-9a-fA-F]*"')
ASM_LINE_MATCH = re.compile(r'label=<.*/>>')
FUNCTIONLESS_GV_BLOCK = re.compile(r'V_0x[0-9a-fA-F]* \[ .*')


# Map a rose name to its in/out edge type ints
ROSE_EDGE_TYPES = {
    'call': EdgeType.FUNCTION_CALL,
    'cret': EdgeType.NORMAL,
    'cret\\nassumed': EdgeType.NORMAL,
    'cret\\\\nassumed': EdgeType.NORMAL,
    '': EdgeType.NORMAL,
    'other': EdgeType.NORMAL
}


def parse_rose_gv(cfg, lines):
    """Reads input as a graphviz file

    Args:
        cfg (CFG): an empty/loading CFG() object
        lines (str, Iterable[str], TextIO): the data to parse. Can be a string (which will be split on newlines to get each
            individual line), a list of string (each element will be considered one line), or an open file to call
            `.readlines()` on

    Raises:
        CFGParseError: when the file cannot be parsed correctly
    """
    if isinstance(lines, str):
        lines = lines.split('\n')
    elif hasattr(lines, 'readlines') and callable(lines, 'readlines'):
        lines = lines.readlines()

    try:
        # Clean up the lines a bit
        lines = [l.strip() for l in lines if l.strip()]
    except:
        raise TypeError("Could not parse rose graphviz input of type: '%s'" % type(lines).__name__)

    cfg.metadata['file_type'] = 'gv'

    subgraphs = []
    edges = {}
    curr_blocks = {}
    
    # Keeping track of states
    in_subgraph = False
    eof = False

    for line in lines:
        # Check for empty string, for beginning digraph strings to ignore, and indeterminate/nonexisting nodes
        if line == '' or any(line.startswith(s) for s in DIGRAPH_START_STRINGS + INDETERMINATE_NODE_NAMES):
            continue
        
        # Check for subgraph cluster
        elif line[0] == 's':
            in_subgraph = True

            # Get the function string
            func_str_matches = FUNC_STR_MATCH.findall(line)

            # Check for functions with no name
            if len(func_str_matches) == 0:
                func_str_matches = FUNC_STR_MATCH_NO_NAME.findall(line)
            
            # Otherwise continue normally
            if len(func_str_matches) != 1:
                raise CFGParseError("Could not parse function string from: %s\n Found matches: %s" % (repr(line), func_str_matches))

            # Add a new subgraph to the list (getting the [7:-1] works in both named and unnamed cases)
            # The func_str should be something like 'function [MEMORY_ADDRESS] "[FUNCTION_NAME]"' or 'function [MEMORY_ADDRESS]'
            func_str = func_str_matches[0][7:-1]
            _, func_address, *func_name = func_str.split(' ')
            func_name = ' '.join(func_name)[2:-2] if func_name else None

            subgraphs.append((func_name, int(func_address, 0), []))
        
        # Check for end of subgraph cluster/eof
        elif line[0] == '}':
            # Check to make sure there is only one eof '}' line
            if not in_subgraph:
                if not eof:
                    eof = True
                else:
                    raise CFGParseError("Found multiple lines starting with '}' that did not end subgraphs")

            in_subgraph = False
        
        # Check for nodes/node edges
        elif line[0] == 'V':
            
            # Handle subgraph node, or if it is a block with no parent function
            if in_subgraph or FUNCTIONLESS_GV_BLOCK.fullmatch(line) is not None:

                # Get the node address
                address, _, rest = line.partition(" [ ")
                address = int(address[2:], 0)
                
                # Get the asm line string
                asm_line_match = ASM_LINE_MATCH.findall(rest)

                # Need to leave the first and last <>
                asm_line = asm_line_match[0][6:] if len(asm_line_match) > 0 else ''

                # The tuple for this current node
                node_tup = (address, asm_line)

                # Add this node to our current subgraph if we are in one
                if in_subgraph:
                    subgraphs[-1][2].append(node_tup)
                
                # Otherwise, we are parsing a functionless basic block, create a dummy function to wrap it
                else:
                    subgraphs.append(("__DUMMY_FUNCTION_AT_0x%x__" % address, address, [node_tup]))

            # Handle edge
            else:
                # Get the source and destination names
                source, rest = [a.strip() for a in line.split('->')]
                dest, rest = [a.strip() for a in rest.split(' [ ')]

                # Get the label name by splitting on quotes and getting first index, checking for empty string as well
                label = "" if 'label=""' in rest else rest.split('"')[1]

                # Don't deal with indeterminate edges, unless they are a function return, then send that info
                if dest in INDETERMINATE_NODE_NAMES:
                    continue

                source, dest = int(source[2:], 0), int(dest[2:], 0)

                # Add the edge into the dictionary for the outgoing edge
                edges.setdefault(source, []).append((ROSE_EDGE_TYPES[label], dest))

        # Otherwise, raise error
        else:
            raise CFGParseError("Unknown line: '%s'" % line)
    
    funcs = [_parse_gv_function(cfg, name, address, nodes, edges, curr_blocks) for name, address, nodes in subgraphs]
    cfg.add_function(*funcs)


def _parse_gv_function(cfg, name, address, nodes, edges, curr_blocks):
    """Parses the func_info as a graphviz dot file, returns the function

    Args:
        cfg (CFG): the ``CFG`` to which this function belongs
        name (Union[str, None]): the function name, or None if it doesn't have one
        address (int): the integer address of this function
        nodes (Iterable[Tuple[int, str]]): an iterable of nodes in this subgraph. Each 'node' should be a tuple of 
            (node_address: int, node_asm_lines: str), with the 'node_asm_lines' being the unprocessed string from the 
            graphviz file
        edges (Dict[int, List[Tuple[EdgeType, int]]]): a dictionary of all edges in the cfg. Each key should be a 'from'
            basic block integer address, and values are tuples of outgoing edge information for the block with that 
            address. Each edge information is a tuple of (edge_type: EdgeType, to_address: int)
        curr_blocks (Dict[int, CFGBasicBlock]): a dictionary mapping basic block addresses to ``CFGBasicBlock`` objects. 
            We need this to create new basic blocks on the fly in order to make ``CFGEdge``'s work properly

    Returns:
        CFGFunction: the cfg function
    """
    func = CFGFunction(parent_cfg=cfg, address=get_address(address), name=name, is_extern_function=is_extern_func_name(name))
    for address, asm_lines in nodes:
        _parse_gv_block(func, address, asm_lines, edges.get(address, []), curr_blocks)
    
    return func


def _parse_gv_block(func, address, asm_lines, node_edges, curr_blocks):
    """Parses the incoming block info assuming it is from a graphviz dot file, and appends it to func's blocks

    Args:
        func (CFGFunction): the ``CFGFunction`` this block belongs to
        address (int): integer memory address of the node
        asm_lines (str): the UNPARSED asm lines from the raw gv dot file
        node_edges (Iterable[Tuple[EdgeType, int]]): an iterable of information for all outgoing edges for this block. 
            Each element should be a tuple of (edge_type: EdgeType, to_address: int)
        curr_blocks (Dict[int, CFGBasicBlock]): a dictionary mapping basic block addresses to ``CFGBasicBlock`` objects. 
            We need this to create new basic blocks on the fly in order to make ``CFGEdge``'s work properly
    """
    # Get the CFGBasicBlock with this address
    asm_stuff = {k: v for k, v in zip(['asm_lines', 'asm_memory_addresses'], get_asm_from_node_label(asm_lines))}
    block = create_basic_block(curr_blocks, address, parent_function=func, **asm_stuff)

    # Parse out the edges
    for edge_type, address in node_edges:
        block.edges_out.add(CFGEdge(block, create_basic_block(curr_blocks, address), edge_type))
    
    func.blocks.append(block)


GV_SPLIT = re.compile(r'<br [^>]*/>')
def get_asm_from_node_label(label):
    """Converts a node's label into a list of assembly lines at that basic block.

    Args:
        label (str): the unparsed string label

    Returns:
        Tuple[List[str], List[int]]: tuple of 2 lists: (asm_lines, asm_memory_addresses)
    """
    if label == '' or label is None:
        return []

    # Remove the first and last <>, replace all "??" with empty string, and html-unescape the ampersand encoded things
    ret = [('0x' + html.unescape(l.replace("??", ""))) for l in GV_SPLIT.split(label[1:-1]) if l != ""]

    # Split on spaces and get the first one to get the memory address, the rest are joined to be the instruction
    lines = [line.strip() for r in ret for addr, _, line in [r.replace('\t', '').partition(' ')]]
    addrs = [int(addr, 0) for r in ret for addr, _, line in [r.replace('\t', '').partition(' ')]]
    return lines, addrs

