import numpy as np
import pickle
import warnings
from enum import Enum
from ..normalization import normalize_cfg_data, Architectures
from .cfg import CFG
from ..utils import get_smallest_np_dtype, scatter_nd_numpy, hash_obj, eq_obj, get_module, AtomicTokenDict
from ..labeling import NODE_LABELS


# Global constants for edge connection value types
NORMAL_EDGE_CONN_VALUE = 1
FUNCTION_CALL_EDGE_CONN_VALUE = 2


class MemCFG:
    """A CFG that is more memory/speed efficient.
    
    Keeps only the bare minimum information needed from a CFG. Stores edge connections in a CSR-like format.

    Parameters
    ----------
    cfg: `CFG`
        a CFG object. Can be a normalized or un-normalized. If un-normalized, then it will be normalized using the 
        `normalizer` parameter.
    normalizer: `Optional[Union[str, Normalizer]]`
        the normalizer to use to normalize the incoming CFG (or None if it is already normalized). If the incoming CFG 
        object has already been normalized, and `normalizer` is not None, then this will attempt to normalize the CFG 
        again with this normalizer
    inplace: `bool`
        if True and cfg needs to be normalized, it will be normalized inplace
    using_tokens: `Union[Dict[str, int], AtomicTokenDict]`
        if not None, then a dictionary mapping token strings to integer values. Any tokens in cfg but not in using_tokens
        will be added. Can also be an AtomicTokenDict for atomic updates to tokens
    force_renormalize: `bool`
        by default, this method will only normalize cfg's whose .normalizer != to the passed normalizer. However if 
        `force_renormalize=True`, then all cfg's will be renormalized even if they have been previously normalized with 
        the same normalizer.
    """

    normalizer = None
    """The normalizer used to normalize input before converting to ``MemCFG``
    
    Can be shared with a ``MemCFGDataset`` object if this ``MemCFG`` is a part of one
    """

    tokens = None
    """Dictionary mapping token strings to integer values used in this ``MemCFG``
    
    Can be shared with a ``MemCFGDataset`` object if this ``MemCFG`` is a part of one.

    Can also be an AtomicTokenDict object for atomic token updates
    """

    function_name_to_idx = None
    """Dictionary mapping string function names to their integer ids used in this ``MemCFG``"""

    asm_lines = None
    """Assembly line information
    
    A contiguous 1-d numpy array of shape (num_asm_lines,) of integer assembly line tokens. Dtype is the smallest 
    unsigned dtype needed to store the largest token value in this ``MemCFG``

    To get the assembly lines for some block index `block_idx`, you must get the assembly line indices from ``block_asm_idx``,
    and use those to slice the assembly lines:

    >>> block_idx = 7
    >>> memcfg.asm_lines[memcfg.block_asm_idx[block_idx]:memcfg.block_asm_idx[block_idx + 1]]

    Also see :func:`~bincfg.MemCFG.get_block_asm_lines`
    """

    block_asm_idx = None
    """Indices in ``asm_lines`` that correspond to the assembly lines for each basic block in this ``MemCFG``
    
    A 1-d numpy array of shape (num_blocks + 1,). Dtype is the smallest unsigned dtype needed to store the value 
    `num_asm_lines`. Assembly tokens for a block at index `i` would have a start index of `block_asm_idx[i]` and an end
    index of `block_asm_idx[i + 1]` in ``asm_lines``.
    """

    block_func_idx = None
    """Integer ids for the function that each basic block belongs to
    
    A 1-d numpy array of shape (num_blocks,) where each element is a function id for the block at that index. The id
    can be found in ``function_name_to_idx``. Dtype is the smallest unsigned dtype needed to store the value `num_functions`

    Also see :func:`~bincfg.MemCFG.get_block_function_idx` and :func:`~bincfg.MemCFG.get_block_function_name`
    """

    block_flags = None
    """Integer of bit flags for each basic block
    
    A 1-d numpy array of shape (num_blocks,) where each element is an integer of bit flags. See ``BlockInfoBitMask``
    for more info. Dtype is the smallest unsigned dtype with enough bits to store all flags in ``BlockInfoBitMask``

    Also see :func:`~bincfg.MemCFG.get_block_flags`
    """

    block_labels = None
    """Dictionary mapping block indices to integer block label bit flags
    
    Only blocks that have known labels will be in this dictionary. The bit flags integer will have the bit set for each
    index in the `bincfg.labelling.NODE_LABELS` list. EG: the 0-th element would correspond to the 1's bit, the 1-th
    element would correspond to the 2's bit, etc.

    Also see :func:`~bincfg.MemCFG.get_block_labels`
    """

    metadata = None
    """Dictionary of metadata associated with this MemCFG"""

    graph_c = None
    """Array containing all of the outgoing edges for each block in order
    
    1-D numpy array of shape (num_edges,). Dtype will be the smallest unsigned dtype required to store the value
    `num_blocks + 1`. Each element is a block index to which that edge connects. Edges will be in the order they appear in 
    each block's ``edges_out`` attribute, for each block in order of their block_idx. 

    Also see :func:`~bincfg.MemCFG.get_edges_out`

    NOTE: this also contains information on which types of edges they are inherently. If the block is NOT a function call
    (stored as bit flag in the block_info array), then all edges for that block are normal edges. If it IS a function
    call, then there are 3 cases:

        1. it has one outgoing edge: that edge is always a function call
        2. it has two outgoing edges, one function call, one normal: the first edge is the function call edge, the second
           is a normal edge
        3. it has >2 outgoing edges, or 2 function call edges: the edges will be listed first by function call edges, 
           then by normal edges, with a separator inbetween. The separator will have the max unsigned int value for 
           graph_c's dtype. This is why we use the dtype that can store `num_blocks + 1`, since we need this extra value
           just in case. Whatever exactly it means for a basic block to have >2 outgoing edges while being a function 
           call is left up to the user. Possibly due to call operators with non-explicit operands (eg: register memory 
           locations)?
    """

    graph_r = None
    """Array containing information on the number of outgoing edges for each block
    
    1-D numpy array of shape (num_edges + 1,). Dtype will be the smallest unsigned dtype required to store the value
    `num_edges`. This array is a cumulative sum of the number of edges for each basic block. One could get all of the 
    outgoing edges for a block using:

    >>> start_idx = memcfg.graph_r[block_idx]
    >>> end_idx = memcfg.graph_r[block_idx + 1]
    >>> edges = memcfg.graph_c[start_idx:end_idx]

    Also see :func:`~bincfg.MemCFG.get_edges_out`
    """

    class BlockInfoBitMask(Enum):
        """An Enum for block info bit masks
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        Each value is a tuple of the bit mask for that boolean, and a function to call with the block that returns a
        boolean True if that bit should be set, False otherwise. If True, then that bit will be '1' in that block's 
        block_flags int.
        """

        IS_FUNCTION_CALL = 1 << 0, lambda block: block.is_function_call
        """Bit set if this block is a function call. See :py:func:`~bincfg.cfg_basic_block.is_function_call`"""
        IS_FUNCTION_ENTRY = 1 << 1, lambda block: block.is_function_entry
        """Bit set if this block is a function entry. See :py:func:`~bincfg.cfg_basic_block.is_function_entry`"""
        IS_FUNCTION_RETURN = 1 << 2, lambda block: block.is_function_return
        """Bit set if this block is a function return. See :py:func:`~bincfg.cfg_basic_block.is_function_return`"""
        IS_IN_EXTERN_FUNCTION = 1 << 3, lambda block: block.parent_function.is_extern_function
        """Bit set if this block is within an external function. See :py:func:`~bincfg.cfg_function.is_extern_function`"""
        IS_FUNCTION_JUMP = 1 << 4, lambda block: block.is_function_jump
        """Bit set if this block is a function jump. IE: this block has a jump instruction that resolves to a basic block
        in a separate function. See :py:func:`~bincfg.cfg_basic_block.is_function_jump`"""
        IS_MULTI_FUNCTION_CALL = 1 << 5, lambda block: 0
        """Bit set if this block is a multi-function call. IE: this block has either two or more function call edges out,
        or one function call and two or more normal edges out. See :py:func:`~bincfg.cfg_basic_block.is_multi_function_call`
        
        Currently not setting the block here in _block_flags_int(), but instead in MemCFG initialization in order to save
        time (we don't have to compute get_sorted_edges() multiple times)
        """

    @staticmethod
    def _block_flags_int(block):
        """Gets all of the block information and stores it as a integer of bit-set flags

        Args:
            block (CFGBasicBlock): the CFGBasicBlock to get the info int from

        Returns:
            int: the integer of bit-set flags
        """
        block_flags = 0
        for bm in MemCFG.BlockInfoBitMask:
            block_flags |= bm.value[0] * bm.value[1](block)  # Times will convert boolean to 1 or 0
        return block_flags

    def __init__(self, cfg, normalizer=None, inplace=False, using_tokens=None, force_renormalize=False):
        # We can initialize empty if cfg is None
        if cfg is None:
            return

        # Make sure input is a cfg
        if not isinstance(cfg, CFG):
            raise TypeError("Can only build a MemCFG out of a CFG object, not '%s'" % type(cfg).__name__)

        # Normalize the CFG if needed
        if normalizer is not None or cfg.normalizer is None:
            # Make sure there is some normalizer to use
            if normalizer is None:
                raise ValueError("Must pass a normalizer if `cfg` is unnormalized!")
            cfg = normalize_cfg_data(cfg, normalizer=normalizer, inplace=inplace, force_renormalize=force_renormalize)
        
        # Keep cfg's normalization if possible
        self.normalizer = cfg.normalizer

        # Figure out what the tokens should be, updating the token dict if we find new ones
        self.tokens = {} if using_tokens is None else using_tokens

        if isinstance(self.tokens, AtomicTokenDict):
            self.tokens.addtokens(*{t for block in cfg.blocks for l in block.asm_lines for t in l[1]})
        else:
            for block in cfg.blocks:
                for l in block.asm_lines:
                    for t in l[1]:
                        self.tokens.setdefault(t, len(self.tokens))
                    
        max_token = len(self.tokens)

        # Make mappings from function names to indices. Make sure there aren't duplicates (most likely only going to
        #   occur in functions with no name)
        self.function_name_to_idx = {}
        function_addr_to_idx = {}
        for i, f in enumerate(cfg.functions):
            if f.nice_name in self.function_name_to_idx:
                func_name = f.nice_name
                fn_idx = 0
                while func_name in self.function_name_to_idx:
                    func_name = f.nice_name + '_%d' % fn_idx
                    fn_idx += 1
            else:
                func_name = f.nice_name
            self.function_name_to_idx[func_name] = i
            function_addr_to_idx[f.address] = i

        # Make the data arrays
        self.asm_lines = np.empty([cfg.num_asm_lines], dtype=get_smallest_np_dtype(max_token))
        self.block_asm_idx = np.empty([cfg.num_blocks + 1], dtype=get_smallest_np_dtype(cfg.num_asm_lines))
        self.block_func_idx = np.empty([cfg.num_blocks], dtype=get_smallest_np_dtype(len(cfg.functions_dict)))
        self.block_flags = np.empty([cfg.num_blocks], dtype=get_smallest_np_dtype(len(MemCFG.BlockInfoBitMask)))
        graph_c = []
        self.graph_r = np.empty([cfg.num_blocks + 1])
        self.block_labels = {}  # Mapping of block indices to block labels

        # Set the initial graph_r start and final block_asm_idx
        self.graph_r[0] = 0
        self.block_asm_idx[-1] = len(self.asm_lines)

        # Copy the metadata from the cfg
        self.metadata = cfg.metadata.copy()

        # Create temporary mapping of CFGBasicBlock.address to integer index of block (sorted by address for determinism)
        block_addr_to_idx = {block.address: i for i, block in enumerate(cfg.blocks)}

        # Convert the data at each block to better memory one
        asm_line_idx = 0
        for block_idx, block in enumerate(cfg.blocks):

            # Get the assembly lines, and store the length of the assembly lines
            asm_line_end = asm_line_idx + block.num_asm_lines
            self.asm_lines[asm_line_idx: asm_line_end] = [self.tokens[t] for l in block.asm_lines for t in l[1]]
            self.block_asm_idx[block_idx] = asm_line_idx
            asm_line_idx = asm_line_end

            # Get the block's function name
            self.block_func_idx[block_idx] = function_addr_to_idx[block.parent_function.address]

            # Get all of the edges associated with this block in order: normal edges, function call edges
            edge_lists = block.get_sorted_edges(direction='out', as_sets=False)

            # Add in the function call edges
            graph_c += [block_addr_to_idx[edge.to_block.address] for edge in edge_lists[1]]

            # Check for either >1 function call edges, or >2 edges while being a function call.
            # We add a -1, which will be rolled over to the unsigned int max value for graph_c array
            added_minus_1 = 0
            if len(edge_lists[1]) >= 2 or (len(edge_lists[1]) == 1 and len(edge_lists[0]) > 1):
                graph_c.append(-1)
                added_minus_1 = 1
            
            # Add in the normal edges
            graph_c += [block_addr_to_idx[edge.to_block.address] for edge in edge_lists[0]]

            # Update the new graph_r, taking into account whether or not we inserted an extra -1 to split function call/normal edges
            self.graph_r[block_idx + 1] = sum(len(l) for l in edge_lists) + added_minus_1
            
            # Get the block information flags for this block.
            # Set the flag for is_multi_function_call here so we don't have to call get_sorted_edges more than once
            self.block_flags[block_idx] = MemCFG._block_flags_int(block) | \
                (MemCFG.BlockInfoBitMask.IS_MULTI_FUNCTION_CALL.value[0] * added_minus_1)

            # Get the block label info
            if len(block.labels) > 0:
                block_val = 0
                for label_num in block.labels:
                    block_val |= 2 ** label_num
                self.block_labels[block_idx] = block_val
        
        # Conver graph_c and graph_r to numpy arrays
        self.graph_c = np.array(graph_c, dtype=get_smallest_np_dtype(len(cfg.blocks_dict) + 1))
        self.graph_r = np.cumsum(self.graph_r, dtype=get_smallest_np_dtype(len(self.graph_c)))
        self.max_graph_c_int = np.iinfo(self.graph_c.dtype).max
    
    def get_block(self, block_idx, as_dict=False):
        """Returns all the info associated with the given block index

        Args:
            block_idx (int): integer block index
            as_dict (bool): if True, will return a dictionary of the values with the same names in the "Returns" section

        Returns:
            Tuple[np.ndarray, np.ndarray, int, bool, bool, bool, bool, bool, List[int]]: the block info - 
                (asm_lines, edges_out, edge_types, function_idx, is_function_call, is_function_entry, is_function_return, 
                is_extern_function, is_function_jump, is_block_multi_function_call, labels_list)
        """
        ret = (self.get_block_asm_lines(block_idx),) + self.get_block_edges_out(block_idx, ret_edge_types=True) \
            + (self.get_block_function_idx(block_idx),) + self.get_block_flags(block_idx) + (self.get_block_labels(block_idx),)
        
        if as_dict:
            ret = {k: v for k, v in zip(['asm_lines', 'edges_out', 'edge_types', 'function_idx', 'is_function_call', 
                    'is_function_entry', 'is_function_return', 'is_extern_function', 'is_function_jump', 
                    'is_block_multi_function_call', 'labels_list'], ret)}
        
        return ret
    
    def get_block_info(self, block_idx, as_dict=False):
        """Alias for :func:`~bincfg.MemCFG.get_block`. Returns all the info associated with the given block index

        Args:
            block_idx (int): integer block index
            as_dict (bool): if True, will return a dictionary of the values with the same names in the "Returns" section

        Returns:
            Tuple[np.ndarray, np.ndarray, int, bool, bool, bool, bool, bool, List[int]]: the block info - 
                (asm_lines, edges_out, edge_types, function_idx, is_function_call, is_function_entry, is_function_return, 
                is_extern_function, is_function_jump, is_block_multi_function_call, labels_list)
        """
        return self.get_block(block_idx, as_dict=as_dict)

    def get_block_asm_lines(self, block_idx):
        """Get the asm lines associated with this block index

        Args:
            block_idx (int): 

        Returns:
            np.ndarray: a numpy array of assembly tokens
        """
        return self.asm_lines[self.block_asm_idx[block_idx]:self.block_asm_idx[block_idx+1]]
    
    def get_block_edges_out(self, block_idx, ret_edge_types=False):
        """Get numpy array of block indices for all edges out associated with the given block index

        Args:
            block_idx (int): integer block index
            ret_edge_types (bool): if True, will also return a numpy array (1-d, dtype np.uint8) containing the edge
                type values for each edge with values:

                    - 1: normal edge
                    - 2: function call edge

        Returns:
            np.ndarray: a numpy array of block indices
        """
        # Get all of the edges
        #ret = self.graph_c[slice(*self.graph_r[block_idx:block_idx+2])]
        ret = self.graph_c[self.graph_r[block_idx]:self.graph_r[block_idx+1]]

        # Check if we are returning the edge types as well
        if ret_edge_types:

            # Check if this block is a function call
            if self.is_block_function_call(block_idx):

                # Check to see if this block is a multi-function call, in which case we have to split up the array
                if self.is_block_multi_function_call(block_idx):
                    
                    # Get the split index (index in ret that is equal to unsigned dtype max). It should have split_idx
                    #   function call edges, and len(ret) - split_idx - 1 normal edges (to account for the fact that
                    #   the split_idx itself is also stored in the array)
                    split_idx = np.argwhere(ret == self.max_graph_c_int)[0][0]
                    return ret[ret != self.max_graph_c_int], \
                        np.array([FUNCTION_CALL_EDGE_CONN_VALUE] * split_idx + [NORMAL_EDGE_CONN_VALUE] * (len(ret) - split_idx - 1))
                
                # Otherwise it is not a multi-function call. We can simply return ret, and edge types are either 
                #   [function_call] or [function_call, normal]
                else:
                    return ret, np.array(([FUNCTION_CALL_EDGE_CONN_VALUE] if len(ret) == 1 else \
                        [FUNCTION_CALL_EDGE_CONN_VALUE, NORMAL_EDGE_CONN_VALUE]), dtype=np.uint8)
            
            # It's not a function call, so we can just return ret and all edge types must be normal edges
            else:
                return ret, np.full(ret.shape, NORMAL_EDGE_CONN_VALUE, dtype=np.uint8)

        # Otherwise we are not returning the edge types, just return all values in ret that are not the splitting value
        # We only need to remove the splitting value if there are more than two edges, AND it is a function call. Otherwise
        #   we are doing a useless numpy comparison and memory-grabbing every call which takes a non-neglible amount of time
        return ret[ret != self.max_graph_c_int] #if len(ret) > 2 and self.is_block_function_call(block_idx) else ret

    def get_block_function_idx(self, block_idx):
        """Get the function index for the given block index

        Args:
            block_idx (int): integer block index

        Returns:
            int: the integer function index for the given block index
        """
        return self.block_func_idx[block_idx]
    
    def get_block_function_name(self, block_idx):
        """Get the function name for the given block index
        
        Functions without names will start with '__unnamed_func__'

        Args:
            block_idx (int): integer block index

        Returns:
            str: the function name for the given block index
        """
        func_idx = self.get_block_function_idx(block_idx)

        # Make an inverse mapping now that we know we are calling this function
        if not hasattr(self, 'function_idx_to_name'):
            self.function_idx_to_name = {v: k for k, v in self.function_name_to_idx.items()}

        return self.function_idx_to_name[func_idx]
    
    def get_block_flags(self, block_idx):
        """Get all block flags for the given block index

        Args:
            block_idx (int): integer block index

        Returns:
            Tuple[bool, bool, bool, bool, bool]: (is_block_function_call, is_block_function_start, is_block_function_return, 
                is_block_extern_function, is_block_function_jump, is_block_multi_function_call)
        """
        return self.is_block_function_call(block_idx), self.is_block_function_entry(block_idx), \
            self.is_block_function_return(block_idx), self.is_block_extern_function(block_idx), \
            self.is_block_function_jump(block_idx), self.is_block_multi_function_call(block_idx)
    
    def get_block_labels(self, block_idx):
        """Returns a list of integer block labels for the given block_idx. 
        
        Labels will be integers of the indices in NODE_LABELS. IE: if this CFG has labels ['ecryption', 'file_io', 
        'network_io', 'error_handler', 'string_parser'], and a block at block_idx has block_labels of [0, 3], then that 
        block would be both an 'ecryption' block and a 'error_handler' block. If a block has no labels ([]), then it 
        should be assumed that we don't know what labels it should have, as opposed to it having no labels.

        Args:
            block_idx (int): integer block index

        Returns:
            List[int]: list of integer labels for the given block index
        """
        if block_idx in self.block_labels:
            return list(i for i in range(len(NODE_LABELS)) if self.block_labels[block_idx] & 2 ** i != 0)
        return []
    
    def get_function_block_inds(self, func_idx):
        """Returns all of the block indices that are within the given function"""
        if func_idx not in self.function_name_to_idx.values():
            raise ValueError("Unknown function index: %d" % func_idx)
        return [i for i in range(self.num_blocks) if self.block_func_idx[i] == func_idx]
    
    def is_block_function_call(self, block_idx):
        """True if this block is a function call, False otherwise"""
        return (self.block_flags[block_idx] & MemCFG.BlockInfoBitMask.IS_FUNCTION_CALL.value[0]) > 0
    
    def is_block_function_return(self, block_idx):
        """True if this block is a function return, False otherwise"""
        return (self.block_flags[block_idx] & MemCFG.BlockInfoBitMask.IS_FUNCTION_RETURN.value[0]) > 0
    
    def is_block_function_entry(self, block_idx):
        """True if this block is a function entry, False otherwise"""
        return (self.block_flags[block_idx] & MemCFG.BlockInfoBitMask.IS_FUNCTION_ENTRY.value[0]) > 0
    
    def is_block_extern_function(self, block_idx):
        """True if this block is in an external function, False otherwise"""
        return (self.block_flags[block_idx] & MemCFG.BlockInfoBitMask.IS_IN_EXTERN_FUNCTION.value[0]) > 0
    
    def is_block_function_jump(self, block_idx):
        """True if this block is a function jump, False otherwise"""
        return (self.block_flags[block_idx] & MemCFG.BlockInfoBitMask.IS_FUNCTION_JUMP.value[0]) > 0
    
    def is_block_multi_function_call(self, block_idx):
        """True if this block is a multi-function call, False otherwise"""
        return (self.block_flags[block_idx] & MemCFG.BlockInfoBitMask.IS_MULTI_FUNCTION_CALL.value[0]) > 0
    
    def is_block_labeled(self, block_idx, label=None):
        """Checks if the given block_idx is labeled

        If label is None, returns True if the block at the given index has a label, False if it has no labels 
        (self.block_labels[block_idx] == 0)
        Otherwise returns True if the block at the given index has the given label, False if not

        Args:
            block_idx (int): integer block index
            label (Union[str, int, None], optional): if not None, then the label to check the block_idx for. Can be 
                either a string (whos lowercase name must be in NODE_LABELS), or an integer for the index in NODE_LABELS 
                to check for. Otherwise if label is None, then this will check to see if the given block_idx is labeled 
                at all. Defaults to None.

        Raises:
            ValueError: for a bad/unknown `label` value
            TypeError: for a bad `label` type

        Returns:
            bool: True if the block has the label, False otherwise
        """
        if label is None:
            return block_idx in self.block_labels and self.block_labels[block_idx] != 0
        
        # Check for bad label values
        if isinstance(label, str):
            if label.lower() not in NODE_LABELS:
                raise ValueError("Unknown label: %s" % label)
            label = NODE_LABELS.index(label.lower())
        elif isinstance(label, int):
            if label < 0 or label >= len(NODE_LABELS):
                raise ValueError("Cannot get label index %d of NODE_LABELS with length %d" % (label, len(NODE_LABELS)))
        else:
            raise TypeError("Unknown label type: %s. Should be None, string, or int" % type(label))
        
        # Check block for specific label
        return block_idx in self.block_labels and self.block_labels[block_idx] & 2 ** label != 0
    
    @property
    def num_blocks(self):
        """The number of blocks in this ``MemCFG``"""
        return len(self.block_func_idx)
    
    @property
    def num_edges(self):
        """The number of edges in this ``MemCFG``"""
        return len(self.graph_c)
    
    @property
    def num_asm_lines(self):
        """The number of assembly lines in this ``MemCFG``"""
        return len(self.asm_lines)
    
    @property
    def num_functions(self):
        """The number of functions in this ``MemCFG``"""
        return len(self.function_name_to_idx)
    
    def update_metadata(self, other):
        """Updates this MemCFG's metadata dictionary with the given dictionary, and returns self"""
        self.metadata.update(other)
        return self
    
    def set_metadata(self, metadata):
        """Sets the metadata and returns self"""
        self.metadata = metadata
        return self
    
    def set_tokens(self, tokens):
        """Sets this MemCFG's tokens to the given tokens, and returns self"""
        self.tokens = tokens
        return self
    
    def normalize(self, normalizer=None, using_tokens=None, inplace=True, force_renormalize=False):
        """Normalizes this memcfg in-place.

        Args:
            normalizer (Union[str, Normalizer], optional): the normalizer to use. Can be a ``Normalizer`` object, or a 
                string, or None to use the default BaseNormalizer(). Defaults to None.
            using_tokens (TokenDictType): tokens to use when normalizing
            inplace (bool, optional): whether or not to normalize inplace. Defaults to True.
            force_renormalize (bool, optional): by default, this method will only normalize this cfg if the passed 
                `normalizer` is != `self.normalizer`. However if `force_renormalize=True`, then this will be renormalized
                even if it has been previously normalized with the same normalizer. Defaults to False.

        Returns:
            MemCFG: this ``MemCFG`` normalized
        """
        return normalize_cfg_data(self, normalizer=normalizer, using_tokens=using_tokens, inplace=inplace, 
                                  force_renormalize=force_renormalize)
    
    @property
    def architecture(self):
        """Returns the architecture being used. Currently a WIP
        
        Checks for an 'arch' or 'architecture' key in the metadata and returns it if it is known. Can currently return:
        'java', 'x86'
        """
        for k in ['arch', 'architecture']:
            if k in self.metadata:
                arch = self.metadata[k]
                break
        else:
            raise ValueError("Could not find 'arch' or 'architecture' key in metadata")
        
        if arch in ['x86']:
            return Architectures.X86
        elif arch in ['java', 'java-bytecode']:
            return Architectures.JAVA
        else:
            raise ValueError("Unknown architecture value: %s" % repr(arch))
    
    def get_edge_values(self):
        """Returns the edge type values
        
        Returns a 1-d numpy array of length self.num_edges and dtype np.int32 containing an integer type for each
        edge depending on if it is a normal/function call/call return edge:

        Edges are directed and have values:

            - 0: No edge
            - 1: Normal edge
            - 2: Function call edge
        
        NOTE: this returns as type np.int32 since pytorch can be finicky about what dtypes it wants

        Returns:
            np.ndarray: the edge type values
        """
        edge_values = np.empty(self.graph_c.shape, dtype=np.int32)
        evi = 0
        for bi in range(self.num_blocks):
            l = self.graph_r[bi + 1] - self.graph_r[bi]
            if self.is_block_function_call(bi):
                if l >= 2:
                    edge_values[evi:evi+2] = (FUNCTION_CALL_EDGE_CONN_VALUE, NORMAL_EDGE_CONN_VALUE)
                    evi += 2
                else:
                    assert l == 1
                    edge_values[evi:evi + 1] = FUNCTION_CALL_EDGE_CONN_VALUE
                    evi += 1
            else:
                edge_values[evi:evi + l] = NORMAL_EDGE_CONN_VALUE
                evi += l

        return edge_values
    
    def get_coo_indices(self):
        """Returns the COO indices for this MemCFG

        Returns a 2-d numpy array of shape (num_edges, 2) of dtype np.int32. Each row is an edge, column 0 is the 'row' 
        indexer, and column 1 is the 'column' indexer. EG:
        
        .. highlight:: python
        .. code-block:: python
        
            original = np.array([
                [0, 1],
                [1, 1]
            ])

            coo_indices = np.array([
                [0, 1],
                [1, 0],
                [1, 1]
            ])
        
        NOTE: this returns as type np.int32 since pytorch can be finicky about what dtypes it wants
        NOTE: pytorch sparse_coo_tensor's indicies are the transpose of the array this method returns

        Returns:
            np.ndarray: the coo indices
        """
        inds = np.empty([self.num_edges, 2], dtype=np.int32)
        for bi in range(self.num_blocks):
            start, end = self.graph_r[bi], self.graph_r[bi + 1]
            inds[start:end, 0] = bi
            inds[start:end, 1] = self.graph_c[start:end]
        return inds
    
    def to_adjacency_matrix(self, type='np', sparse=False):
        """Returns an adjacency matrix representation of this memcfg's graph connections

        Connections will be directed and have values:

            - 0: No edge
            - 1: Normal edge
            - 2: Function call edge

        See :func:`bincfg.memcfg.to_adjacency_matrix` for more details

        Args:
            type (str, optional): the type of matrix to return. Defaults to 'np'. Can be:

                - 'np'/'numpy' for a numpy ndarray (dtype: np.int32)
                - 'torch'/'pytorch' for a pytorch tensor (type: LongTensor)
            
            sparse (bool, optional): whether or not the return value should be a sparse matrix. Defaults to False. Has 
                different behaviors based on type:

                - numpy array: returns a 2-tuple of sparse COO representation (indices, values). 
                    NOTE: if you want sparse CSR format, you already have it with self.graph_c and self.graph_r
                - pytorch tensor: returns a pytorch sparse COO tensor. 
                    NOTE: not using sparse CSR format for now since it seems to have less documentation/supportedness. 

        Returns:
            Union[np.ndarray, torch.Tensor]: an adjacency matrix representation of this ``MemCFG``
        """
        type = type.lower()

        # Return adj mat as intended type
        if type in ['np', 'numpy']:
            if sparse:
                return (self.get_coo_indices().T, self.get_edge_values())
            else:
                ret = np.zeros((self.num_blocks, self.num_blocks))
                return scatter_nd_numpy(ret, self.get_coo_indices(), self.get_edge_values())

        elif type in ['torch', 'pytorch']:
            torch = get_module('torch', err_message="Cannot find module `torch` required to return pytorch tensors!")
            sparse_coo = torch.sparse_coo_tensor(indices=self.get_coo_indices().T, values=self.get_edge_values(), size=(self.num_blocks, self.num_blocks)).coalesce()
            return sparse_coo if sparse else sparse_coo.to_dense()

        else:
            raise ValueError("Unknown adjacency matrix type: '%s'" % type)
    
    def to_cfg(self):
        """Converts this MemCFG back into a CFG, though some information may have been lost
        
        Specifically, the following information will/may have been lost:

            - memory addresses have been erased. Fake ones will be used in the construction of the new CFG. There is
              no guarantee about the order/values of these memory addresses, only that there will be a unique one for
              each basic block, and each function will have the same address as its entry basic block.
            - assembly lines may have lost some information depending on the normalization method used when building
              this MemCFG
         
        All other information should be the same as the original CFG including:

            - basic blocks and functions
            - basic block and function information (excluding addresses) such as function names, 
        """
        raise NotImplementedError
    
    def save(self, path):
        """Saves this MemCFG to the given path"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def dumps(self):
        """Returns this object pickled with pickle.dumps()"""
        return pickle.dumps(self)
    
    def drop_tokens(self):
        """Sets the tokens in this normalizer to None. Make sure you only do this if tokens are saved elsewhere! Returns self"""
        self.tokens = None
        return self
    
    @classmethod
    def load(cls, path):
        """Loads a MemCFG from the given path"""
        with open(path, 'rb') as f:
            return pickle.load(f)
        
    def __str__(self):
        return "MemCFG with normalizer: %s and %d functions, %d blocks, %d assembly lines, and %d edges" % \
            (repr(str(self.normalizer)), self.num_functions, self.num_blocks, self.num_asm_lines, self.num_edges)
    
    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        """State for pickling"""
        ret = self.__dict__.copy()
        if 'function_idx_to_name' in ret:
            del ret['function_idx_to_name']
        return ret
    
    def __setstate__(self, state):
        """Set state for pickling"""
        for k, v in state.items():
            setattr(self, k, v)

        if not hasattr(self, 'max_graph_c_int'):
            self.max_graph_c_int = np.iinfo(self.graph_c.dtype).max
    
    def __eq__(self, other):
        return isinstance(other, MemCFG) and all(eq_obj(self, other, selector=s) for s in [
            'asm_lines', 'block_asm_idx', 'block_func_idx', 'block_flags', 'graph_c', 'graph_r',
            'block_labels', 'metadata', 'function_name_to_idx', 'normalizer', 'tokens',
        ])
    
    def __hash__(self):
        return hash_obj([
            self.asm_lines, self.block_asm_idx, self.block_func_idx, self.block_flags, self.graph_c, self.graph_r,
            self.block_labels, self.metadata, self.function_name_to_idx, self.tokens, self.normalizer,
        ], return_int=True)
    
    def hash(self, strict=True, num_wl_iters=3):
        """Returns an integer hash of this MemCFG
        
        Args:
            strict (str): whether or not to do a strict hashing. If true, then this does the strictest hashing, including 
                things like function names, etc. This is the default behavior, and the same as the __hash__ function. If
                False, then this does a less strict hashing using only things like graph/function topology and assembly
                lines (not addresses). This performs multiple passes of the weisfeiler lehman kernel to produce the final
                hash. The nodes are then aggregated into function hashes, and the WL kernel is applied to the function
                call graph multiple times as well. Those function hashes are then aggregated into the final binary hash.
                This way, we should be invariant to node rearrangings (within the same function) as well as function
                rearrangings. This, however, is not perfect. Any two CFG's with differing hashes are for sure different
                (there are no false negatives), however it is possible for two differing CFG's to hash to the same
                value depending on graph structure (although, it is unlikely for genuine binaries)
            num_wl_iters (int): the number of iterations of the weisfeiler lehman kernel to apply
        """
        _INT_DTYPE = np.uint64

        if strict:
            return self.__hash__()
        
        # Generate the initial node states by hashing their data. Also save function info that will be used later
        block_hashes = np.zeros([self.num_blocks], dtype=_INT_DTYPE)
        func_block_inds = [[] for _ in range(len(self.function_name_to_idx))]
        func_call_inds = []
        for i in range(self.num_blocks):
            block_info = self.get_block(i, as_dict=True)

            # Keep track of the function info
            func_block_inds[block_info['function_idx']].append(i)
            func_call_inds += [(block_info['function_idx'], self.block_func_idx[block_info['edges_out'][tidx]]) for tidx, t in enumerate(block_info['edge_types']) if t == FUNCTION_CALL_EDGE_CONN_VALUE]

            # Remove any edge and function information (that will come later in the WL passes)
            del block_info['edges_out'], block_info['edge_types'], block_info['function_idx']
            block_info['asm_lines'] = [self.tokens[i] for i in block_info['asm_lines']]

            block_hashes[i] = hash_obj(block_info, return_int=True) % np.iinfo(_INT_DTYPE).max
        
        # Get the node adjacency matrix. We can keep function call edges, we just make sure their edge types are included
        #   as a part of the hashing process
        indices, values = self.to_adjacency_matrix(type='np', sparse=True)

        # Perform some number of WL iterations on nodes
        for it in range(num_wl_iters):
            block_hashes = _sparse_wl_iteration(block_hashes, indices, values)

        # Aggregate blocks into function hashes then multiply by another prime and add in an extra value in case some
        #   functions have no blocks, and find the unique indices in func_call_inds
        func_hashes = np.array([block_hashes[fbis].sum() for fbis in func_block_inds]) * 37 + 41
        func_call_inds = np.unique(func_call_inds).T
        func_values = np.ones([func_call_inds.shape[1]])

        # Perform some number of WL iterations on functions
        for it in range(num_wl_iters):
            func_hashes = _sparse_wl_iteration(func_hashes, func_call_inds, func_values)
        
        # Now we can aggregate all of the function hashes into the final binary hash and return
        return func_hashes.sum()
    
    def __lt__(self, other):
        import warnings
        warnings.warn("Less than is not currently implemented for MemCFG's!")
        return False


_SPARSE_WL_PRIMES = np.array([17, 19, 31])  # The first one won't ever be used since values is 1 or 2, but oh well
def _sparse_wl_iteration(hashes, indices, values):
    """Perform a single WL iteration and return the new hashes
    
    Can be performed on both node and function hashings

    Args:
        hashes (np.ndarray): 1-d integer numpy array of the current hashes of shape (num_objects,)
        indices (np.ndarray): 2-d integer numpy array of indices in a sparse COO tensor of shape (2, num_non_zero). These
            correspond to the non zero coordinates in the adjacency matrix
        values (np.ndarray): 1-d integer numpy array of the connection types of shape (num_non_zero,). For basic blocks,
            these values can either be 1 or 2 (NORMAL_EDGE, FUNCTION_CALL_EDGE). For functions, these should all be 1's
    """ 
    # Attempt to load either torch or scipy to perform sparse matrix multiplies
    torch = get_module('torch', raise_err=False)
    scipy = get_module('scipy', raise_err=False)
    if torch is None and scipy is None:
        warnings.warn("Could not load either torch or scipy to perform sparse matrix multiplications, this will be slow...")
    
    # Convert the values to primes so we have unique hashes
    values = _SPARSE_WL_PRIMES[values]
    
    # If pytorch is available, do a simple sparse matrix multiply to get the new dense values
    if torch is not None:
        adj_mat = torch.sparse_coo_tensor(indices=indices, values=values, shape=(len(hashes), len(hashes)))
        hashes = torch.tensor(hashes, dtype=torch.uint64).view([-1, 1])
        new_hashes = torch.sparse.mm(adj_mat, hashes).view([-1]).numpy()
    
    # If scipy is available, we can use that
    elif scipy is not None:
        adj_mat = scipy.sparse.coo_matrix((values, (indices[0, :], indices[1, :]))).tocsr()
        new_hashes = (adj_mat @ hashes.reshape([-1, 1])).reshape([-1])

    # Otherwise, we're doing slow numpy version
    else:
        raise NotImplementedError("Slow numpy WL hashing")
    
    # Now, we can add on the old hashes, multiplied by some more primes. Overflows should automatically wrap thanks to numpy
    return hashes * 13 + new_hashes * 11
