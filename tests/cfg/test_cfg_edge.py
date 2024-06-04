import pytest
from bincfg import CFGEdge, CFGBasicBlock, EdgeType


def test_construction():
    """Building the CFGEdge fails with bad inputs, works with good ones"""
    block = CFGBasicBlock()
    with pytest.raises(TypeError):
        CFGEdge(None, None, None)
    with pytest.raises(TypeError):
        CFGEdge(block, int, EdgeType.FUNCTION_CALL)
    with pytest.raises(TypeError):
        CFGEdge('a', block, EdgeType.NORMAL)
    with pytest.raises(TypeError):
        CFGEdge('a', block, None)
    with pytest.raises(TypeError):
        CFGEdge('a', block, 'blahblah')
    with pytest.raises(ValueError):
        CFGEdge(block, block, 'blahblah')
    
    CFGEdge(block, block, 'normal')
    CFGEdge(block, block, 'function_call')
    CFGEdge(block, block, EdgeType.NORMAL)
    CFGEdge(block, block, EdgeType.FUNCTION_CALL)


def test_immutable():
    """Can't overwrite values in a CFGEdge, must make new ones"""
    block = CFGBasicBlock()
    edge = CFGEdge(block, block, 'normal')

    with pytest.raises(TypeError):
        edge.from_block = block
    with pytest.raises(TypeError):
        edge.to_block = block
    with pytest.raises(TypeError):
        edge.edge_type = EdgeType.NORMAL
    with pytest.raises(TypeError):
        edge.test_value = 'normal'
    with pytest.raises(TypeError):
        edge.__init__(block, block, 'normal')


def test_builtins():
    """Builtin dunder methods"""
    block = CFGBasicBlock()
    edge = CFGEdge(block, block, 'normal')

    assert isinstance(str(edge), str)
    assert isinstance(repr(edge), str)
    assert isinstance(hash(edge), int)


def test_properties():
    """Test various @property values"""
    blocks = {1: CFGBasicBlock(address='1'), 2: CFGBasicBlock(address='2'), 3: CFGBasicBlock(address='3')}

    for et in ['normal', EdgeType.NORMAL]:
        edge = CFGEdge(blocks[1], blocks[2], et)
        assert edge.is_normal_edge
        assert not edge.is_function_call_edge
    for et in ['function_call', EdgeType.FUNCTION_CALL]:
        edge = CFGEdge(blocks[1], blocks[2], et)
        assert edge.is_function_call_edge
        assert not edge.is_normal_edge
    
    blocks[2].edges_out = set([CFGEdge(blocks[2], blocks[3], 'normal')])
    blocks[3].edges_out = set([CFGEdge(blocks[3], blocks[1], 'normal')])

    blocks[1].edges_out = set([CFGEdge(blocks[1], blocks[2], 'normal'), CFGEdge(blocks[1], blocks[3], 'normal')])
    assert list(blocks[1].edges_out)[0].is_branch
    assert not list(blocks[2].edges_out)[0].is_branch
    assert not list(blocks[3].edges_out)[0].is_branch

    blocks[1].edges_out = set([CFGEdge(blocks[1], blocks[2], 'normal'), CFGEdge(blocks[1], blocks[3], 'function_call')])
    assert not list(blocks[1].edges_out)[0].is_branch
    blocks[1].edges_out = set([CFGEdge(blocks[1], blocks[2], 'normal'), CFGEdge(blocks[1], blocks[3], 'normal'), CFGEdge(blocks[1], blocks[1], 'normal')])
    assert not list(blocks[1].edges_out)[0].is_branch
    blocks[1].edges_out = set([CFGEdge(blocks[1], blocks[2], 'normal')])
    assert not list(blocks[1].edges_out)[0].is_branch

def test_eq_and_hash():
    """Equality and hashing are equal, uses basic block addresses to end recursion"""
    blocks = {1: CFGBasicBlock(address='1'), 2: CFGBasicBlock(address='2'), 3: CFGBasicBlock(address='3')}
    
    edges = [
        (CFGEdge(blocks[1], blocks[2], 'normal'), 'a'),
        (CFGEdge(blocks[1], blocks[2], 'normal'), 'a'),

        (CFGEdge(blocks[1], blocks[3], 'normal'), 'b'),
        (CFGEdge(blocks[2], blocks[2], 'normal'), 'c'),

        (CFGEdge(blocks[1], blocks[2], 'function_call'), 'd'),

        (CFGEdge(blocks[2], blocks[1], 'normal'), 'e'),

        (CFGEdge(blocks[1], blocks[3], EdgeType.FUNCTION_CALL), 'f'),
        (CFGEdge(blocks[1], blocks[3], EdgeType.FUNCTION_CALL), 'f'),
    ]

    for e1, v1 in edges:
        for e2, v2 in edges:
            if v1 == v2:
                assert e1 == e2, "e1: %s, v1: %s, e2: %s, v2: %s" % (e1, v1, e2, v2)
                assert e2 == e1, "e1: %s, v1: %s, e2: %s, v2: %s" % (e1, v1, e2, v2)
                assert hash(e1) == hash(e2), "hash e1: %s, v1: %s, e2: %s, v2: %s" % (e1, v1, e2, v2)
            else:
                assert e1 != e2, "e1: %s, v1: %s, e2: %s, v2: %s" % (e1, v1, e2, v2)
                assert e2 != e1, "e1: %s, v1: %s, e2: %s, v2: %s" % (e1, v1, e2, v2)
                assert hash(e1) != hash(e2), "hash e1: %s, v1: %s, e2: %s, v2: %s" % (e1, v1, e2, v2)

