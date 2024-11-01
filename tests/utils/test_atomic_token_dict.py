"""Testing for atomic token dictionary"""
import os
import pickle
import bincfg
import pytest
import time
import numpy as np
import multiprocessing
from bincfg import AtomicTokenDict, AcquireLockError


def _rm_paths(*paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def test_make_token_dict():
    """we can make a token dictionary and load/store tokens"""
    tokens_path = os.path.join(os.path.dirname(__file__), 'test_tokens.pkl')
    lock_path = tokens_path + '.lock'
    _rm_paths(tokens_path, lock_path)

    tokens = {'a': 0, 'b': 1}
    tokens = AtomicTokenDict(init_data=tokens, filepath=tokens_path, lockpath=lock_path, timeout=30)
    tokens.update({'c': 2, 'd': 3})
    tokens.addtokens('e', 'f', 'g')

    expected = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6}
    assert tokens.data == expected

    del tokens
    
    with open(tokens_path, 'rb') as f:
        tokens = pickle.load(f)
    
    assert tokens == expected
    assert not os.path.exists(lock_path)
    _rm_paths(tokens_path, lock_path)


def test_overwrite_lock_file():
    """We can overwrite a broken lock file"""
    tokens_path = os.path.join(os.path.dirname(__file__), 'test_tokens.pkl')
    lock_path = tokens_path + '.lock'
    _rm_paths(tokens_path, lock_path)

    # Correct lockfile
    token_data = {'a': 0, 'b': 1}
    with open(tokens_path, 'wb') as f:
        pickle.dump(token_data, f)
    with open(lock_path, 'w') as f:
        f.write("abcdefg-11111")
    
    tokens = AtomicTokenDict(init_data=None, filepath=tokens_path, lockpath=lock_path, timeout=30)

    assert tokens.data == token_data
    del tokens
    assert not os.path.exists(lock_path)

    # Malformed lockfile
    with open(lock_path, 'w') as f:
        f.write("")
    
    tokens = AtomicTokenDict(init_data=None, filepath=tokens_path, lockpath=lock_path, timeout=30)

    assert tokens.data == token_data
    del tokens
    assert not os.path.exists(lock_path)
    _rm_paths(tokens_path, lock_path)


def test_many_lockfiles():
    """Multiple lockfiles still succeeding"""
    _multi_lockfiles(bincfg.utils.atomic_token_dict._MAX_ATOMIC_STALE_FILE_DEPTH // 2)


def test_too_many_lockfiles():
    """Too many lockfiles to overwrite"""
    with pytest.raises(AcquireLockError):
        _multi_lockfiles(bincfg.utils.atomic_token_dict._MAX_ATOMIC_STALE_FILE_DEPTH)


def test_multiple_processes():
    """Tests running multiple processes accessing the same file at the same time"""
    n_procs = 8
    path = os.path.join(os.path.dirname(__file__), 'test_tokens.pkl')
    runtime = 10.0
    
    with multiprocessing.Pool(processes=n_procs) as pool:
        start_time = time.time()
        results = pool.starmap(_multiple_processes_run_func, [(i, runtime, path, start_time) for i in range(n_procs)], chunksize=1)
    
    with open(path, 'rb') as f:
        pf_tokens = pickle.load(f)
    
    correct = set()
    for (_, id_str, idx) in results:
        for i in range(idx):
            correct.add(str(id_str) + '-%d' % i)
    
    assert correct == set(pf_tokens.keys())
    for (r, i, _) in results:
        assert r == pf_tokens


def _multiple_processes_run_func(id_str, runtime, path, start_time):
    rng = np.random.default_rng(hash(id_str))

    tokens = AtomicTokenDict(init_data=None, filepath=path)
    idx = 0
    while time.time() - start_time < runtime:
        new_tokens = [str(id_str) + '-%d' % idx for _ in range(100)]
        idx += len(new_tokens)

        # Add the tokens in a bunch of different ways
        choice = rng.choice(3)
        if choice == 0:
            tokens.addtokens(*new_tokens)
        elif choice == 1:
            for t in new_tokens:
                tokens.setdefault(t)
        elif choice == 2:
            tokens.addtokens(*new_tokens)
            for t in new_tokens:
                tokens.setdefault(t)
            tokens.addtokens(*new_tokens)
        else:
            raise ValueError()
    
    time.sleep(4)
    tokens.refresh()
    return (tokens.get_dict(), id_str, idx)


def _multi_lockfiles(n_locks):
    tokens_path = os.path.join(os.path.dirname(__file__), 'test_tokens.pkl')
    lock_paths = [tokens_path + '.lock']
    for i in range(n_locks):
        lock_paths.append(os.path.join(os.path.dirname(lock_paths[-1]), '%d-%s' % (i, os.path.basename(lock_paths[-1]))))
    _rm_paths(tokens_path, *lock_paths)

    for path in lock_paths:
        with open(path, 'w') as f:
            f.write('abcd-111')
    
    token_dict = {'z': 0, 'x': 1, 'y': 2}
    with open(tokens_path, 'wb') as f:
        pickle.dump(token_dict, f)
    
    try:
        tokens = AtomicTokenDict(init_data=None, filepath=tokens_path, lockpath=lock_paths[0], timeout=30)
        assert tokens.data == token_dict
        for path in lock_paths:
            assert not os.path.exists(path)
    finally:
        _rm_paths(tokens_path, *lock_paths)
