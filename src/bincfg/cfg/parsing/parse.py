"""Parse CFG input data

To make a new data parsing method:

    - Add a check for that data type in the `parse_cfg_data` function, if needed. If your new method relys on other
      libraries being installed, make sure your checks don't require them to be installed. If your method reads text
      input from a file, then add a new way of checking for specific strings and whatnot in the `_get_parse_func_from_lines`
      method below
    - Make a new file here in the `bincfg.cfg.parsing` folder containing the function used to parse that data. It should
      modify a newly constructed empty CFG() object
    - Import that new method in this file and call this new method from your check in the `parse_cfg_data` function
"""
import sys
import bincfg
import copy
from ..cfg_function import CFGFunction
from ..cfg_basic_block import CFGBasicBlock
from ...utils import get_module
from .rose_text import parse_rose_txt
from .rose_gv import parse_rose_gv
from .smda import parse_smda
from .angr import parse_angr


if get_module('angr', raise_err=False):
    angr = get_module('angr')
    _ANGR_DTYPES = [angr.project.Project, angr.analyses.cfg.cfg_fast.CFGFast, 
                    angr.analyses.cfg.cfg_emulated.CFGEmulated, angr.analyses.cfg.cfg_base.CFGBase]
else:
    _ANGR_DTYPES = []


def parse_cfg_data(cfg, data):
    """Parses the incoming cfg data. Infers type of data

    Args:
        cfg (CFG): the cfg to parse into
        data (Union[str, Sequence[str], TextIO, pd.DataFrame]): the data to parse, can be:

            - string: either string with newline characters that will be split on all newlines and treated as either a
              text or graphviz rose input, or a string with no newline characters that will be treated as a filename.
              Filenames will be opened as ghidra parquet files if they end with either '.pq' or '.parquet', and
              text/graphviz rose input otherwise
            - Sequence of string: will be treated as already-read-in text/graphviz rose input
            - open file object: will be read in using `.readlines`, then treated as text/graphviz rose input
            - pandas dataframe: will be parsed as ghidra parquet file

    Raises:
        ValueError: bad ``str`` filename, or an unknown file start string
        TypeError: bad ``data`` input type
        CFGParseError: if there is an error during CFG parsing (but data type was inferred correctly)
    """
    if isinstance(data, str):

        # Check for the empty string, and initialize empty
        if data == '':
            return
            
        # Check for single string to split on newlines
        if '\n' in data:
            data = [l.strip() for l in data.split('\n') if l.strip()]

        # Otherwise, assume it is a file
        else:
            cfg.metadata['filepath'] = data

            # Assume it is a text file
            try:
                with open(data, 'r') as f:
                    data = [l.strip() for l in f.readlines() if l.strip()]
            except:
                raise ValueError("Data was assumed to be a filename, but that file could not be opened/read!: %s" % repr(data))

    # Check for an open file
    elif hasattr(data, 'readlines') and callable(data.readlines):
        data = [l.strip() for l in data.readlines() if l.strip()]

    # Check for a copy constructor
    elif isinstance(data, bincfg.CFG):
        return _copy_constructor(cfg, data)
    
    # Check for a smda report object from a disassembled file
    elif get_module('smda', raise_err=False) and isinstance(data, (sys.modules['smda'].common.SmdaReport.SmdaReport)):
        return parse_smda(data, cfg)
    
    # Check for an angr project, or pre-computed angr CFG
    elif get_module('angr', raise_err=False) and any(isinstance(data, cls) for cls in _ANGR_DTYPES):
        return parse_angr(data, cfg)

    # Check for a networkx to read in. Do this after angr since they also use networkx
    elif get_module('networkx', raise_err=False) and isinstance(data, (sys.modules['networkx'].DiGraph)):
        return bincfg.CFG.from_networkx(data, cfg=cfg)

    # Otherwise, assume it is a sequence of string lines
    else:
        try:
            data = [l.strip() for l in data if l.strip()]
        except:
            raise TypeError("Could not parse CFG data from data of type: '%s'" % type(data).__name__)
    
    # If data is a list right now, assume we need to get the function from a list of lines
    if isinstance(data, list):
        func = _get_parse_func_from_lines(data)
    
    func(cfg, data)


def _get_parse_func_from_lines(lines):
    """Returns the function that should be used to parse this list of lines.

    Assumes all empty lines have already been stripped/removed

    Args:
        lines (Sequence[str]): the list of lines

    Returns:
        Callable[[CFG, Any], None]: the function to use to parse
    """

    # If lines is empty, return a function that does nothing
    if len(lines) == 0:
        return lambda *args, **kwargs: None
    
    # Otherwise check for different start lines
    else:
        if lines[0].startswith('digraph'):
            return parse_rose_gv
        elif lines[0].startswith('function'):
            return parse_rose_txt
        else:
            raise ValueError("Unknown file start string, could not infer file type!:\n%s\n..." % repr(lines[0][:100]))


def _copy_constructor(cfg, input_data):
    """Copies CFG data from `input_data` to `cfg`"""
    cfg.add_function(*[
        CFGFunction(address=func.address, name=func.name, is_extern_function=func._is_extern_function, metadata=copy.deepcopy(func.metadata), blocks=[
            CFGBasicBlock(
                address=block.address,
                edges_out=[(e.from_block.address, e.to_block.address, e.edge_type) for e in block.edges_out],
                asm_lines=copy.deepcopy(block.asm_lines),
                asm_memory_addresses=copy.deepcopy(block.asm_memory_addresses),
                metadata=copy.deepcopy(block.metadata),
            ) for block in func.blocks
        ]) for func in input_data.functions_dict.values()
    ])

    # Also need to copy the metadata
    cfg.metadata = copy.deepcopy(input_data.metadata)

    # Also copy the normalizer
    cfg.normalizer = copy.deepcopy(input_data.normalizer)

