import pytest
import numpy as np
from bincfg import MemCFG, normalize_cfg_data, CFG, EdgeType, CFGEdge, get_architecture
from bincfg.normalization.norm_utils import INSTRUCTION_START_TOKEN
from .manual_cfgs import get_all_manual_cfg_functions
from .test_normalize_cfg import ARCH_NORMS


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
@pytest.mark.parametrize('norm_method', ['prenorm', 'pass_norm', 'call_normfunc', 'normfunc_convert'])#['prenorm', 'pass_norm', 'call_normfunc', 'normfunc_convert'])
def test_manual_memcfg(cfg_func, norm_method, print_hashes):
    """Convert CFG() to MemCFG() as expected"""
    res = cfg_func(build_level='cfg')
    orig_cfg: CFG = res['cfg'].copy()

    memcfg_hashes = {}

    for norm_name, (norm_class, norm_kwargs) in ARCH_NORMS[orig_cfg.architecture].items():
        for tl in ['op', 'inst']:
            normalizer = norm_class(tokenization_level=tl, **norm_kwargs)

            if norm_method == 'prenorm':
                memcfg = MemCFG(orig_cfg.copy().normalize(normalizer), keep_memory_addresses=True, inplace=True)
            elif norm_method == 'pass_norm':
                memcfg = MemCFG(orig_cfg.copy(), normalizer=normalizer, keep_memory_addresses=True, inplace=False, force_renormalize=True)
            elif norm_method == 'call_normfunc':
                memcfg = MemCFG(normalize_cfg_data(orig_cfg.copy(), normalizer=normalizer, inplace=True), keep_memory_addresses=True)
            elif norm_method == 'normfunc_convert':
                memcfg: MemCFG = normalize_cfg_data(orig_cfg.copy(), normalizer=normalizer, inplace=False, convert_to_mem=True, conv_keep_mem_addrs=True)
            else:
                raise NotImplementedError(norm_method)
            
            memcfg_hashes[norm_name+'-'+tl] = hash(memcfg)
            if not print_hashes:
                assert hash(memcfg) == res['expected']['memcfg_hashes'][norm_name+'-'+tl]
            
            cfg: CFG = orig_cfg.copy().normalize(normalizer, inplace=True)

            # Make sure our memcfg matches expected values
            assert memcfg.metadata == cfg.metadata
            assert memcfg.num_asm_lines == cfg.num_asm_lines
            assert memcfg.num_blocks == cfg.num_blocks
            assert memcfg.num_functions == cfg.num_functions
            assert memcfg.num_edges == cfg.num_edges
            assert memcfg.architecture == cfg.architecture == get_architecture(res['expected']['architecture'])

            # Extra functions and things
            block_inds = {b.address: i for i, b in enumerate(cfg.blocks)}
            assert list(memcfg.get_edge_values()) == [e.edge_type.value for b in cfg.blocks for eset in b.get_sorted_edges(direction='out', edge_types=['fc', 'normal']) for e in eset]
            assert memcfg.get_coo_indices().tolist() == [[block_inds[b.address], block_inds[e.to_block.address]] for b in cfg.blocks for eset in b.get_sorted_edges(direction='out', edge_types=['fc', 'normal']) for e in eset]

            # Make sure functions match expected values
            memcfg.get_block_function_name(0)
            for i, func in enumerate(cfg.functions):
                assert memcfg.get_function_metadata(i) == func.metadata
                assert set(memcfg.get_function_block_inds(i)) == set([block_inds[b.address] for b in func.blocks])  # Function might have blocks in a different order
            
            # Check function names, and add '._memcfg_parent_func_name' attribute to blocks
            for temp_cfg in [cfg, orig_cfg]:
                name_mapping = {}
                for i, func in enumerate(temp_cfg.functions):
                    name_mapping.setdefault(func.name, []).append((i, func))
                for name, func_list in name_mapping.items():
                    if len(func_list) == 1:
                        assert memcfg.function_idx_to_name[func_list[0][0]] == name
                        assert memcfg.function_name_to_idx[name] == func_list[0][0]

                        for block in func_list[0][1].blocks:
                            block._memcfg_parent_func_name = func_list[0][1].name
                    else:
                        for v in range(len(func_list)):
                            fn = name if v == 0 else (name + ('_%d' % (v - 1)))
                            assert memcfg.function_idx_to_name[func_list[v][0]] == fn
                            assert memcfg.function_name_to_idx[fn] == func_list[v][0]

                            for block in func_list[v][1].blocks:
                                block._memcfg_parent_func_name = fn

            # Make sure blocks match expected values
            assert len(set(f.address for f in cfg.functions)) == len(cfg.functions)
            func_inds = {f.address: i for i, f in enumerate(cfg.functions)}
            block_func_inds = {i: func_inds[b.parent_function.address] for i, b in enumerate(cfg.blocks)}
            block_addresses = {b.address: i for i, b in enumerate(cfg.blocks)}
            assert memcfg.get_block_metadata(None) == [b.metadata for b in cfg.blocks]
            for i, block in enumerate(orig_cfg.blocks):
                assert [memcfg.inv_tokens[t] for t in list(memcfg.get_block_asm_lines(i))] == normalizer.normalize(*block.asm_lines, block=block, cfg=cfg) == cfg.get_block(block).asm_lines
                assert memcfg.get_block_memory_address(i) == block.address
                assert list(memcfg.get_block_asm_memory_addresses(i)) == block.asm_memory_addresses
                assert memcfg.get_block_function_idx(i) == block_func_inds[i]
                assert memcfg.get_block_function_name(i) == block._memcfg_parent_func_name

                norm_edges = [block_addresses[edge.to_block.address] for edge in sorted(list(block.edges_out)) if edge.edge_type == EdgeType.NORMAL]
                func_edges = [block_addresses[edge.to_block.address] for edge in sorted(list(block.edges_out)) if edge.edge_type == EdgeType.FUNCTION_CALL]
                assert list(memcfg.get_block_edges_out(i, ret_edge_types=False)) == (func_edges + norm_edges) == [block_addresses[e.to_block.address] for eset in block.get_sorted_edges(direction='out', edge_types=['fc', 'normal']) for e in eset]
                assert list(memcfg.get_block_edges_out(i, ret_edge_types=True)[1]) == [e.edge_type.value for eset in block.get_sorted_edges(direction='out', edge_types=['fc', 'normal']) for e in eset]

                is_block_function_call = len(block.get_sorted_edges(edge_types='function_call', direction='out')[0]) > 0
                is_block_function_entry = block.address == block.parent_function.address
                is_block_extern_function = block.parent_function.is_extern_function
                is_block_function_jump = any(e.to_block.parent_function.address != block.parent_function.address for e in block.edges_out if e.edge_type == EdgeType.NORMAL)
                is_block_multi_function_call = len(block.get_sorted_edges(edge_types='function_call', direction='out')[0]) > 1
                assert memcfg.get_block_flags(i) == (is_block_function_call, is_block_function_entry,
                                                        is_block_extern_function, is_block_function_jump, is_block_multi_function_call)
                assert memcfg.get_block_flags(i) == (memcfg.is_block_function_call(i), memcfg.is_block_function_entry(i),
                                                        memcfg.is_block_extern_function(i),
                                                        memcfg.is_block_function_jump(i), memcfg.is_block_multi_function_call(i))
                
                assert memcfg.get_block_metadata(i) == block.metadata

                assert {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in memcfg.get_block_info(i).items()} == {
                    'asm_lines': memcfg.get_block_asm_lines(i).tolist(),
                    'asm_memory_addresses': memcfg.get_block_asm_memory_addresses(i).tolist(),
                    'edges_out': memcfg.get_block_edges_out(i, ret_edge_types=False).tolist(),
                    'edge_types': memcfg.get_block_edges_out(i, ret_edge_types=True)[1].tolist(),
                    'function_index': memcfg.get_block_function_idx(i),
                    'is_function_call': memcfg.is_block_function_call(i),
                    'is_function_entry': memcfg.is_block_function_entry(i),
                    'is_extern_function': memcfg.is_block_extern_function(i),
                    'is_function_jump': memcfg.is_block_function_jump(i),
                    'is_multi_function_call': memcfg.is_block_multi_function_call(i),
                    'metadata': memcfg.get_block_metadata(i),
                }

        # We can convert back to CFG and it works. Rename CFG functions
        cfg_func_names = {}
        for func in cfg.functions:
            if func.name in cfg_func_names:
                cfg_func_names[func.name] += 1
                func.name = func.name + '_%d' % (cfg_func_names[func.name] - 1)
            else:
                cfg_func_names[func.name] = 0
        assert memcfg.to_cfg() == cfg

        # Make sure we can update metadata/tokens
        expected_metadata = memcfg.metadata.copy()
        expected_metadata['TEST_EXPECTED_VALUE'] = 10
        assert memcfg.update_metadata({'TEST_EXPECTED_VALUE': 10}).metadata == expected_metadata
        assert memcfg.set_tokens({'a': 1}).tokens == {'a': 1}
        assert memcfg.drop_tokens().tokens is None

    if print_hashes:
        print(__file__+'-manual_memcfg_hash', memcfg_hashes)
    

@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_memcfg_renormalize(cfg_func):
    """Make sure we can renormalize correctly"""
    orig_cfg = cfg_func(build_level='cfg')['cfg']

    # Make sure we can renormalize
    normalizers = ARCH_NORMS[orig_cfg.architecture].values()
    renorms = [(norm_class, norm_kwargs) for norm_class, norm_kwargs in normalizers if norm_class(**norm_kwargs).renormalizable]
    for tl1 in ['op', 'inst']:
        for tl2 in ['op', 'inst']:
            for renorm_class, renorm_kwargs in renorms:
                renorm = renorm_class(tokenization_level=tl1, **renorm_kwargs)
                for norm_class, norm_kwargs in normalizers:
                    norm = norm_class(tokenization_level=tl2, **norm_kwargs)
                    for keep_mem in [True, False]:
                        if keep_mem:
                            assert MemCFG(orig_cfg.copy(), normalizer=renorm, keep_memory_addresses=keep_mem).normalize(norm, inplace=True).to_cfg() == CFG(orig_cfg.copy(), normalizer=norm), "Fail with renorm=%s, norm=%s, keep_mem=%s" % (repr(renorm), repr(norm), keep_mem)
                        assert MemCFG(orig_cfg.copy(), normalizer=norm, keep_memory_addresses=keep_mem) == MemCFG(orig_cfg.copy(), normalizer=renorm, keep_memory_addresses=keep_mem).normalize(norm, inplace=True)


def test_empty_memcfg():
    """Tests passing an empty CFG() into MemCFG"""
    assert MemCFG(CFG(), normalizer='x86base').to_cfg() == CFG(normalizer='x86base')
    
