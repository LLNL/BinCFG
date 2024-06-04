import pytest
from bincfg import get_normalizer, Architectures
from .manual_cfgs import get_all_manual_cfg_functions


# All of the normalizers to test per architecture
ARCH_NORMS = {
    Architectures.X86: ['x86innereye', 'x86safe', 'x86deepsemantic', 'x86deepbindiff', 'x86hpc', 'x86base'],
    Architectures.JAVA: ['java'],
}


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_normalized(cfg_func):
    """Tests normalization of cfg's"""
    res = cfg_func(build_level='cfg')
    cfg = res['cfg']

    assert not hasattr(cfg, 'tokens')
    for norm_str in ARCH_NORMS[cfg.architecture]:
        normalizer = get_normalizer(norm_str)

        for inplace in [False, True]:
            using_tokens = {}
            norm_cfg = cfg.normalize(normalizer=normalizer, using_tokens=using_tokens, inplace=inplace)
            
            if inplace:
                assert cfg is norm_cfg
            else:
                assert cfg is not norm_cfg
                assert cfg != norm_cfg

            # Make sure the graph structure is all the same
            assert len(cfg.blocks) == len(norm_cfg.blocks)
            assert len(cfg.functions) == len(norm_cfg.functions)
            for f1, f2 in zip(cfg.functions, norm_cfg.functions):
                assert len(f1.blocks) == len(f2.blocks)

                for b1, b2 in zip(list(sorted(f1.blocks, key=lambda b: b.address)), list(sorted(f2.blocks, key=lambda b: b.address))):
                    assert b1.edges_in == b2.edges_in
                    assert b1.edges_out == b2.edges_out