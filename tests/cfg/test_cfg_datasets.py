import pickle
import os
import pyarrow.parquet as pq
from .manual_cfgs import get_all_manual_cfg_functions
from bincfg import CFGDataset, CFG, MemCFG, MemCFGDataset, Architectures


def test_build_cfg_dataset():
    """Can build basic CFGDatasets"""
    _test_build_dataset(CFGDataset)


def test_build_memcfg_dataset():
    """Can build basic CFGDatasets"""
    _test_build_dataset(MemCFGDataset)


def _test_build_dataset(dataset_class):
    """Can build basic CFGDatasets"""
    dataset = dataset_class()
    load_and_copy_dataset(dataset)
    
    cfg = CFG()
    dataset = dataset_class(cfg, normalizer=None, metadata=None)

    assert len(dataset) == 1
    assert dataset[0] == cfg
    assert dataset.num_blocks == 0
    assert dataset.num_asm_lines == 0
    assert dataset.num_edges == 0
    assert dataset.num_functions == 0
    assert dataset.num_cfgs == 1

    dataset.add_data(cfg)

    assert len(dataset) == 2
    assert dataset[0] == cfg
    assert dataset[1] == cfg
    for _c in dataset:
        assert _c == cfg
    for _c in dataset.cfgs:
        assert _c == cfg
    assert dataset.num_blocks == 0
    assert dataset.num_asm_lines == 0
    assert dataset.num_edges == 0
    assert dataset.num_functions == 0
    assert dataset.num_cfgs == 2
    assert dataset == dataset

    load_and_copy_dataset(dataset)


def test_medium_cfg_dataset():
    """Building a medium-sized dataset with our built-in cfgs"""
    cfgs = [func(build_level='cfg')['cfg'] for func in get_all_manual_cfg_functions()]
    dataset = CFGDataset(cfgs, normalizer=None, metadata={'1': 4, 'a': 'ten'})

    load_and_copy_dataset(dataset)

    new_dataset = CFGDataset(normalizer=None, metadata={'1': 4, 'a': 'ten'})
    for cfg in cfgs:
        new_dataset.add_data(cfg)
    
    assert new_dataset == dataset


def test_medium_memcfg_dataset():
    """Building a medium-sized memcfg dataset with our built-in cfgs"""
    cfgs = [func(build_level='cfg') for func in get_all_manual_cfg_functions()]
    cfgs = [c['cfg'] for c in cfgs if c['cfg'].architecture == Architectures.X86]
    dataset = MemCFGDataset(CFGDataset(cfgs, normalizer=None), normalizer='x86deepsemantic', inplace=False)

    load_and_copy_dataset(dataset)

    cfgs = CFGDataset()
    for i, cfg in enumerate(cfgs):
        if i % 3 == 0:
            cfgs.add_data(cfg)
        elif i % 3 == 1:
            cfgs.add_data(cfg.normalize('x86deepsemantic'))
        elif i % 3 == 2:
            cfgs.add_data(cfg.normalize('x86hpc'))
        else:
            raise NotImplementedError
    
    new_dataset = MemCFGDataset(cfgs, normalizer='x86deepsemantic', inplace=False)
    assert new_dataset == dataset

    new_dataset = MemCFGDataset(CFGDataset(cfgs), normalizer='x86deepsemantic', inplace=False)
    assert new_dataset == dataset

    repeat_cfgs = cfgs * 3
    tokens = {}
    new_dataset = MemCFGDataset(using_tokens=tokens, normalizer='x86deepsemantic')
    for i, cfg in enumerate(repeat_cfgs):
        n = 6
        if i % n == 0:
            new_dataset.add_data(cfg)
        elif i % n == 1:
            new_dataset.add_data(cfg.normalize('x86deepsemantic'))
        elif i % n == 2:
            new_dataset.add_data(cfg.normalize('x86hpc'))
        elif i % n == 3:
            new_dataset.add_data(MemCFG(cfg, normalizer='x86hpc'))
        elif i % n == 4:
            new_dataset.add_data(MemCFG(cfg.normalize('x86deepsemantic')))
        elif i % n == 5:
            new_dataset.add_data(MemCFG(cfg.normalize('x86hpc')))
        else:
            raise NotImplementedError
    
    assert new_dataset == MemCFGDataset(repeat_cfgs, normalizer='x86deepsemantic')


def load_and_copy_dataset(dataset):
    """Tests we can load an copy the given dataset"""
    new_dataset = pickle.loads(pickle.dumps(dataset))
    assert new_dataset == dataset

    filepath = os.path.join(os.path.dirname(__file__), './_temp_dataset')

    # Saving/loading pickle files
    dataset.save(filepath, format='pickle', freeze_tokens=True)
    new_dataset = type(dataset).load(filepath)
    assert new_dataset == dataset

    # Saving/loading parquet files
    dataset.save(filepath, format='parquet', freeze_tokens=True, add_parquet_metadata=False)
    new_dataset_pq = type(dataset).load(filepath)
    assert new_dataset_pq == dataset
    assert new_dataset_pq == new_dataset

    if len(dataset) > 0 and len(dataset[0].metadata) > 0:
        dataset.save(filepath, format='parquet', freeze_tokens=True, add_parquet_metadata=True)
        new_dataset_pq = type(dataset).load(filepath)
        assert new_dataset_pq == dataset
        assert new_dataset_pq == new_dataset

        schema = pq.read_schema(filepath)
        meta_names = set(k for c in dataset for k in c.metadata.keys())

        assert set(schema.names) == meta_names
    
    os.remove(filepath)
