"""Parse SMDA processed outputs"""
import bincfg
import re
from ..cfg_basic_block import CFGBasicBlock
from ..cfg_function import CFGFunction
from ...utils.type_utils import IN_PYTHON_TYPING_VERSION


if IN_PYTHON_TYPING_VERSION:
    import smda


def parse_smda(data: 'smda.common.SmdaReport.SmdaReport', cfg: 'bincfg.CFG'):
    """Parses SMDA data into a CFG

    NOTE: SMDA does not create function call edges, those must be made later
    
    Args:
        data (smda.common.SmdaReport.SmdaReport): SMDA processed data from smda.Disassembler.Disassembler().disassembleFile()
        cfg (CFG): an empty/loading CFG() object
    """
    cfg.metadata['file_type'] = 'smda'

    cfg.add_function(*[
        CFGFunction(address=func.offset, name=func.function_name, blocks=[
                CFGBasicBlock(
                    address=block.offset,
                    edges_in=[(a, None, 'normal') for a in block.getPredecessors()], 
                    edges_out=[(None, a, 'normal') for a in block.getSuccessors()],
                    asm_lines=[str(i) for i in block.getInstructions()]
                ) for block in func.getBlocks()
        ]) for func in data.getFunctions()
    ])

    # Remove all the ": (      f30f1efa) - " inside the assembly lines
    for block in cfg.blocks:
        block.asm_lines = [re.sub(r': *\( *[0-9a-fA-F]+ *\) *- *', ': ', line) for line in block.asm_lines]
