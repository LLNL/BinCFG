"""Atomically update tokens"""
import os
import pickle
import warnings
import time
import numpy as np
import socket
from threading import Thread
from .atomicwrites import atomic_write
from .misc_utils import hash_obj

import bincfg  # Needed for circular import


# Whether or not we warn about atomic data not being able to be loaded when unpickling
_WARN_ATOMIC_DATA = True

def _set_warn_atomic_data(val):
    global _WARN_ATOMIC_DATA
    _WARN_ATOMIC_DATA = val

_ATOMIC_READ_RAISE_ERR = object()

# Time in seconds to wait before writing another byte to the atomic lockfile to show updates
_ATOMIC_LOCK_FILE_UPDATE_TIME = 0.5

# The maximum depth to trying to delete stale files
_MAX_ATOMIC_STALE_FILE_DEPTH = 5


class AtomicData:
    """A class that allows for atomic reading/updating of the given data to a pickle file
    
    Parameters
    ----------
        init_data: `Any`
            Data to initialize the atomic file with. If the atomic file already exists, then that data will be loaded
        filepath: `Optional[str]`
            An optional filepath to store the dictionary, otherwise will be stored at './atomic_dict.pkl'
        lockpath: `Optional[str]`
            An optional filepath for the lock file to use to atomically update the dictionary, otherwise will be
                stored at './.[filepath].lock' where [filepath] is the given `filepath` parameter
        timeout: `Optional[float]`
            An optional float specifying the amount of time in seconds to attempt to acquire a lock before timing out
        delete_file: `bool`
            If True, then the file and lockfile will be deleted on initialization to start from scratch
    """

    def __init__(self, init_data, filepath=None, lockpath=None, timeout=None, delete_file=False):
        self._filepath = './atomic_data.pkl' if filepath is None else filepath
        self._temp_filepath = os.path.join(os.path.dirname(self._filepath), '__tEmPFilE_' + os.path.basename(self._filepath))
        self._lock_path = os.path.join(os.path.dirname(self._filepath), '.%s.lock' % os.path.basename(self._filepath)) if lockpath is None else lockpath
        self._lock = None

        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be > 0: %f" % timeout)
        self._timeout = 2**100 if timeout is None else timeout

        # Delete the files if starting from scratch
        if delete_file:
            self.delete_file(force=True)

        # Get the initial data
        self.atomic_read(default=init_data)
    
    def atomic_read(self, default=_ATOMIC_READ_RAISE_ERR):
        """Atomically reads the data from file, updating self.data
        
        Args:
            default (Optional[Any]): If this is passed and the file does not already exist, then this data will be saved
                to file and set to self.data
        """
        with _AcquireLock(self._timeout, self._lock_path):

            # If the path doesn't exist, check if we need to raise an error, or update the file
            if not os.path.exists(self._filepath):
                if default is not _ATOMIC_READ_RAISE_ERR:
                    self.data = default
                    self._locked_write()
                else:
                    raise FileNotFoundError('Could not find inital atomic file to read from, and `default` data was not passed: %s' % self._filepath)
            
            # Otherwise it does exist, update self
            else:
                self.data = self._locked_read()
        
        return self.data
    
    def _locked_read(self):
        """Reads the data from file, assuming a lock has already been acquired"""
        with open(self._filepath, 'rb') as f:
            return pickle.load(f)
    
    def atomic_write(self):
        """Atomically writes the data at self.data to the pickle file"""
        with _AcquireLock(self._timeout, self._lock_path):
            self._locked_write()
    
    def _locked_write(self):
        """Writes the data at self.data to file, assuming a lock has already been acquired
        
        Will write to a temporary file first, then rename to minimize chance of crashing/killing during write and
        overwriting data
        """
        with open(self._temp_filepath, 'wb') as f:
            pickle.dump(self.data, f)
        os.rename(self._temp_filepath, self._filepath)
    
    def atomic_update(self, update_func, *update_args, **update_kwargs):
        """Atomically updates the data
        
        Will first acquire a lock on the data, read it in, then call `update_func(file_data, update_data)` where `file_data`
        is the data from the current atomic file, then write the data back to file and finally release the lock.

        NOTE: this will prevent any and all updates to the atomic file until update_func has completed

        NOTE: any errors within the update_func will be handled properly and will likely not mess up the atomic file

        Args:
            update_func (Callable): function that takes in: the data currently saved in file, the current data, then the 
                passed args and kwargs, and returns the updated data to write back to file
            update_args (Any): args to pass to update_func, after the current data saved in file
            update_kwargs (Any): kwargs to pass to update_func
        
        Returns:
            Any: the updated data
        """
        with _AcquireLock(self._timeout, self._lock_path) as lock:
            self.data = update_func(self._locked_read(), self.data, *update_args, **update_kwargs)
            self._locked_write()
            return self.data
    
    def acquire_lock(self):
        """Acquires the lock needed to update data
        
        NOTE: this will prevent any and all updates to the atomic file until self.release_lock() is called. Make sure
        you call it quickly or other processes may hang!

        NOTE: if the lock has already been acquired, nothing will happen

        NOTE: it can be dangerous to attempt to acquire locks yourself, as any errors raised must be handled nicely and
        self.release_lock() must be called otherwise other processes may hang
        """
        if self._lock is None:
            self._lock = _AcquireLock(self._timeout, self._lock_path).__enter__()
    
    def release_lock(self):
        """Releases the lock. Assumes it has already been acquired, otherwise an error will be raised"""
        if self._lock is None:
            raise ValueError("release_lock() was called, but the lock has not been acquired!")
        self._lock.__exit__()
        self._lock = None
    
    def delete_file(self, force=False):
        """Atomically deletes the file being used"""
        if force:
            if os.path.exists(self._filepath):
                os.remove(self._filepath)
            if os.path.exists(self._lock_path):
                os.remove(self._lock_path)
        else:
            with _AcquireLock(self._timeout, self._lock_path):
                if os.path.exists(self._filepath):
                    os.remove(self._filepath)
    
    def __len__(self):
        """Gives length of current self.data"""
        return len(self.data)
    
    def __getstate__(self):
        """Doesn't send the actual data itself, that will be loaded"""
        ret = self.__dict__.copy()
        #del ret['data']
        return ret

    def __setstate__(self, state):
        """Set the state as normal, but read in the data when done"""
        for k, v in state.items():
            setattr(self, k, v)
        try:
            self._timeout, old = 3, self._timeout
            self.atomic_read()
            self._timeout = old
        except Exception as e:
            if _WARN_ATOMIC_DATA:
                warnings.warn("Could not load atomic data from file: %s, due to %s: %s. This data could be outdated!" % (self._filepath, type(e).__name__, e))


class _AcquireLock:
    """Context manager to acquire a file lock, and remove it when done"""
    def __init__(self, timeout, lock_path, _stale_file_depth=0):
        self._timeout, self._lock_path = timeout, lock_path
        self._stale_file_time, self._stale_file_hash, self._stale_file_size = None, None, None
        self._stale_file_depth = _stale_file_depth
        self._open_file = None
        self._writing_thread = None
    
    def __enter__(self):
        rng = np.random.default_rng(seed=os.getpid() * int(time.time() * 1_000_000))
        sleep_time = 1e-6
        start_time = time.time()

        while time.time() - start_time < self._timeout:
            try:
                with atomic_write(self._lock_path, overwrite=False) as f:
                    # Store hash of current process for other atomic threads to use to identify new processes that
                    #   have acquired the lock
                    id_hash = hash_obj([socket.gethostname(), os.getpid(), time.time()])
                    f.write('%s-1' % id_hash)
                    f.flush()
                
                self._open_file = open(self._lock_path, 'a')
                self._writing_thread = _AcquireLockUpdateThread(self._open_file)
                self._writing_thread.start()
                return self
            except FileExistsError:
                pass
            
            # Couldn't acquire file lock, sleep for a bit. Allow for random time between 0.9 -> 1.1x the current time
            time.sleep((0.9 + rng.random() * 0.2) * sleep_time)
            sleep_time = min(0.1, sleep_time * 1.05)

            # If our sleep_time is too large, or we are through over half our timeout, check if there is a stale file
            if (sleep_time > 0.08 or time.time() - start_time > 0.5 * self._timeout) and self._stale_file_time is None:
                self._stale_file_time, self._stale_file_hash, self._stale_file_size = self._get_stale_stats()
            
            # If we have been keeping track of our stale file, check if it's time to call it a stale one. Assume bad if 
            #   size is same after 2x our update time
            if self._stale_file_time is not None and (time.time() - self._stale_file_time > 2.0 * _ATOMIC_LOCK_FILE_UPDATE_TIME):
                _, new_hash, new_size = self._get_stale_stats()

                # The file did update, reset
                if self._stale_file_hash != new_hash or self._stale_file_size != new_size:
                    self._stale_file_time, self._stale_file_hash, self._stale_file_size = None, None, None
                    continue

                # Otherwise file hasn't yet updated. Assume it's dead. Attempt to get a lock on the lockfile to overwrite it.
                # If we are already in our max depth of stale files, don't do this
                if self._stale_file_depth >= _MAX_ATOMIC_STALE_FILE_DEPTH:
                    continue
                
                # Get the new lock
                new_lockfile = os.path.join(os.path.dirname(self._lock_path), '%d-%s' % (self._stale_file_depth, os.path.basename(self._lock_path)))
                with _AcquireLock(self._timeout, new_lockfile, self._stale_file_depth + 1):

                    # Now that we have the lock, check to make sure the file still exists as expected. If so, delete it
                    try:
                        _, stale_hash, stale_size = self._get_stale_stats()
                        if self._stale_file_hash == stale_hash and self._stale_file_size == stale_size and os.path.exists(self._lock_path):
                            os.remove(self._lock_path)

                    except Exception as e:
                        pass

        raise AcquireLockError(self._timeout, self._lock_path)

    def _get_stale_stats(self):
        """Get the current time, and attempt to read lockfile to get its size. If fails, set to None"""
        try:
            t = time.time()
            with open(self._lock_path, 'r') as f:
                line = f.read()
                h = line.partition('-')[0]
            return t, h, len(line)
        except Exception as e:
            return None, None
            
    def __exit__(self, exc_type, exc_value, exc_tb):
        if self._open_file is not None:
            self._open_file.close()
        if os.path.exists(self._lock_path):
            os.remove(self._lock_path)


class AcquireLockError(Exception):
    def __init__(self, _timeout, lock_path):
        super().__init__("Could not acquire file lock from file after %f seconds using lock path: %s" % (_timeout, lock_path))


class _AcquireLockUpdateThread(Thread):
    """Class to continually update lockfile to show it's still in use"""
    def __init__(self, openfile):
        super().__init__()
        self.openfile = openfile
    
    def run(self):
        while True:
            time.sleep(_ATOMIC_LOCK_FILE_UPDATE_TIME)
            try:
                self.openfile.write("1")
                self.openfile.flush()
            except Exception as e:
                break


class AtomicTokenDict:
    """Acts like a normal token dictionary, but allows for atomic operations
    
    Parameters
    ----------
        init_data: `Optional[Dict[str, int]]`
            Data to initialize the atomic token dict with. If the atomic file already exists, then that data will be loaded
        filepath: `Optional[str]`
            An optional filepath to store the dictionary, otherwise will be stored at './atomic_dict.pkl'
        lockpath: `Optional[str]`
            An optional filepath for the lock file to use to atomically update the dictionary, otherwise will be
                stored at './.[filepath].lock' where [filepath] is the given `filepath` parameter
        timeout: `Optional[float]`
            An optional float specifying the amount of time in seconds to attempt to acquire a lock before timing out
        delete_file: `bool`
            If True, then the file and lockfile will be deleted on initialization to start from scratch
    """

    def __init__(self, init_data=None, filepath=None, lockpath=None, timeout=None, delete_file=False):
        self._data = AtomicData(init_data={}, filepath=filepath, lockpath=lockpath, timeout=timeout, delete_file=delete_file)
        
        # Check to make sure init_data is a valid type, and there are no duplicate tokens
        if init_data is not None:
            if not isinstance(init_data, (dict, AtomicTokenDict)):
                raise TypeError("Can only initialize AtomicTokenDict with data of type 'dict' or 'AtomicTokenDict', not %s" 
                                % repr(type(init_data).__name__))
            
            found = {}
            for k, v in init_data.items():
                if v in found:
                    raise ValueError("Found tokens with duplicate values in init_data: %s" % [(found[v], v), (k, v)])
                if k in self and self[k] != v:
                    raise ValueError("Found token %s in init_data and in loaded atomic data with different values: %d != %d" % (repr(k), v, self[k]))
                found[v] = k
            
            self.update(init_data)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        if key in self.data:
            if self.data[key] != value:
                raise ValueError("Cannot set token key to a new value! key: %s, value: %s" % (repr(key), value))
            return
        
        self._atomic_update({key: value})
    
    def __contains__(self, key):
        return key in self.data
    
    def __len__(self):
        return len(self.data)
    
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return repr(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    def update(self, tokens):
        """Updates this dictionary with the given tokens
        
        Args:
            tokens (Union[Dict[str, int], AtomicTokenDict]): dictionary mapping token strings to their integer values. 
                Any tokens in the dictionary that are not in this dictionary will be added, and any tokens that already 
                exist and have the same value will be ignored. If there are any tokens that already exist, but have a 
                different value, then an error will be raised
        """
        update_tokens = {}
        for k, v in tokens.items():
            if k in self.data:
                if self.data[k] != v:
                    raise ValueError("Cannot set token key to a new value! key: %s, value: %s" % (repr(k), v))
                continue
            update_tokens[k] = v
        
        if len(update_tokens) > 0:
            self._atomic_update(update_tokens)
    
    def items(self):
        return self.data.items()
    
    def values(self):
        return self.data.values()
    
    def keys(self):
        return self.data.keys()
    
    def get(self, key, default=None):
        return self.data.get(key, default=default)

    def setdefault(self, key, default=None):
        """If the key exists, return the value. Otherwise set the key to the given default (or len(self) if default=None)"""
        if key in self.data:
            return self.data[key]
        
        default = len(self) if default is None else default
        self._atomic_update({key: default})
        return default
    
    def addtokens(self, *tokens):
        """Adds the given tokens to this dictionary, ignoring any that already exist
        
        Args:
            tokens (str): arbitrary number of string tokens to add to this token dict
        """
        update_tokens = {}
        for t in tokens:
            if not isinstance(t, str):
                raise TypeError("Each token must be a string, not %s" % repr(type(t).__name__))
            if t not in self:
                update_tokens[t] = len(self) + len(update_tokens)
        
        if len(update_tokens) > 0:
            self._atomic_update(update_tokens)

    def refresh(self):
        """Loads any new changes from the atomic file"""
        self._atomic_update()
    
    def _atomic_update(self, token_dict=None):
        """Atomically update the tokens from the given token_dict. Does no checks beforehand to see if there are any
        conflicts, duplicates, etc.
        
        Args:
            token_dict (Optional[Dict[str, int]]): token dictionary to update with, or None to just read in any updated
                tokens from file
        """
        self._data.atomic_update(bincfg.update_atomic_tokens, token_dict if token_dict is not None else {})
    
    def delete_file(self):
        "Deletes the atomic token dictinoary file"
        self._data.delete_file()
    
    def get_dict(self):
        """Returns the python dictionary that is holding all of the current AtomicData"""
        return self.data

    @property
    def data(self):
        """Returns the token dictionary"""
        return self._data.data
    
    @property
    def inverse(self):
        """Return a new dict containing an inverse mapping of this current dictionary"""
        return {v: k for k, v in self.items()}
    
    @property
    def filepath(self):
        """Return the filepath being used to store the atomic data"""
        return self._data._filepath
    
    @property
    def lock_path(self):
        """Return the lock path being used to store the atomic data"""
        return self._data._lock_path
    
    def __hash__(self):
        return hash_obj(self.data)
