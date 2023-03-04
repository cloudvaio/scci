# Copyright 2020 - 2022, EMCORE Sp. z o.o.
# SPDX-License-Identifier: MIT

import abc
import base64
import copy
import json.encoder
import json.decoder
import json.scanner
import mmap
import numpy as np
import os
import pickle

from typing import Any, Dict, List, Iterable


class TracesFileProxy(abc.ABC):

    @classmethod
    def create(cls,
               fname,
               total_count,
               max_length,
               dtype=np.float64,
               version=2):
        if version == 1:
            return TracesFileProxy_v1(fname, total_count, max_length, dtype)
        elif version == 2:
            return TracesFileProxy_v2(fname, total_count, max_length, dtype)
        else:
            raise ValueError(f"Unsupported TracesFileProxy version: {version}")

    @classmethod
    def load(cls, fname, mode='r'):
        """
        Load class instance from files. This operation opens a memmap to the trace file.
        
        :param fname: common prefix for the trace and metadata file name
                      (if the files are in a different directory than the execution context,
                      'fname' should be an absolute or relative path excluding file extensions)
        :returns: class instance
        """
        if os.path.exists(os.path.expanduser(fname)):
            return TracesFileProxy_v2.load(fname, mode)
        elif os.path.exists(os.path.expanduser(fname) +
                            '.meta') and os.path.exists(
                                os.path.expanduser(fname) + '.npy'):
            return TracesFileProxy_v1.load(fname, mode)
        else:
            raise ValueError(
                "Couldn't determine file version, neither v1 nor v2 file seem to exist."
            )

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def flush(self) -> None:
        raise NotImplementedError

    fname: str
    max_length: int
    total_count: int
    traces: np.ndarray


class TracesFileProxy_v2(TracesFileProxy):
    """
    The class provides functions for storing and accessing traces on disk.
    Trace files are memmapped, allowing for gathering and processing large amount of data with limited RAM.
    The second revision uses a mixed file format containing both pure traces and metadata in a single file,
    in a way that allows easily mem-mapping the beginning of the file as pure traces 2d array.
    Metadata includes per-trace properties, like 'key', 'textin' (input) and 'textout' (output), but also
    global metadata, such as equipment used and scope parameters.

    Note that due to memmapping, the performance depends on disk access speed
    and thus it may be noticeably slow on HDDs in some setups.

    :ivar fname: filename of the underlying file
    :ivar traces_fname: deprecated, same as fname
    :ivar meta_fname: deprecated, same as fname
    :ivar total_count: number of traces
    :ivar max_length: maximal length of a single trace in points
    :ivar traces: memmapped trace container of dimension [:total_count,:max_length]
    :ivar dtype: data type of the underlying trace container, float64 by default
    :ivar keys: generator (list) of keys used in each trace
    :ivar textins: generator (list) of inputs used in each trace
    :ivar textouts: generator (list) of outputs used in each trace
    """

    __offset_marker_length__ = 8  # 64-bits for "traces" length
    __dtype_str_map__ = {
        'f32': np.float32,
        'f64': np.float64,
    }
    __dtype_str_rmap__ = {v: k for k, v in __dtype_str_map__.items()}
    __mode_map__ = {
        'r': 'rb',
        'r+': 'rb+',
        'w+': 'wb+',
    }
    __access_map__ = {
        'r': mmap.ACCESS_READ,
        'r+': mmap.ACCESS_WRITE,
        'w+': mmap.ACCESS_WRITE
    }
    __default_trace_meta__ = {
        'key': None,
        'txi': None,
        'txo': None,
        'captured': False
    }

    class TFPJSONEncoder(json.encoder.JSONEncoder):

        def default(self, o: Any) -> Any:
            if isinstance(o, (bytes, bytearray, memoryview)):
                b = base64.b64encode(o).decode('ascii')
                return '!!b#' + b
            return super().default(o)

    class TFPJSONDecoder(json.decoder.JSONDecoder):

        @staticmethod
        def _parse_string(*args, **kwargs):
            orig, end = json.decoder.scanstring(*args, **kwargs)
            if orig.startswith('!!b#'):
                orig = base64.b64decode(orig[4:])
            return orig, end

        def __init__(self):
            super().__init__()

            self.parse_string = self._parse_string
            # Sadly, c_make_scanner doesn't handle custom parse_string,
            # so we need to do with python impl. here
            self.scan_once = json.scanner.py_make_scanner(self)

    _encoder = TFPJSONEncoder()
    _decoder = TFPJSONDecoder()

    def __init__(self, fname, mode):
        self.fname = os.path.expanduser(fname)
        self.traces: np.memmap = None
        self.global_meta: Dict[str, Any] = None
        self.trace_meta: List[Dict[str, Any]] = None
        self._mmode = mode
        self._fmode = self.__mode_map__[mode]
        self._amode = self.__access_map__[mode]
        self._file = open(self.fname, self._fmode)
    
    def _init_memmap(self):
        mlen = self.total_count * self.max_length * np.dtype(self.dtype).itemsize
        self._file.seek(0, 2)
        flen = self._file.tell()
        self._file.seek(0, 0)
        if flen < mlen:
            self._file.truncate(mlen)
        self._mmap = mmap.mmap(self._file.fileno(),
                               length=mlen,
                               access=self._amode,
                               offset=0)
        self.traces = np.ndarray(shape=(self.total_count, self.max_length),
                                 dtype=self.dtype,
                                 order='C',
                                 buffer=self._mmap,
                                 offset=0)

    @property
    def traces_fname(self):
        return self.fname

    @property
    def meta_fname(self):
        return self.fname

    @property
    def dtype(self):
        return self.__dtype_str_map__[self.global_meta['dtype']]

    @property
    def max_length(self) -> int:
        return self.global_meta['max_length']

    @property
    def total_count(self) -> int:
        return self.global_meta['total_count']

    def close(self):
        if self._file.mode in ('r+', 'w+'):
            self.flush()
        self.traces = None
        self._mmap.close()
        self._file.close()

    @classmethod
    def create(cls,
               fname,
               total_count,
               max_length,
               dtype=np.float64,
               default_trace_meta=None) -> 'TracesFileProxy_v2':
        """
        Create class instance. This operation opens a memmap to the trace file.
        
        :param fname: common prefix for the trace and metadata file name
        :param total_count: number of traces
        :param max_length: maximal length of a single trace in points
        :returns: class instance
        """
        # Default values:
        if default_trace_meta is None:
            default_trace_meta = cls.__default_trace_meta__

        instance = cls(fname, 'w+')
        instance.global_meta = {
            'total_count': total_count,
            'max_length': max_length,
            'dtype': cls.__dtype_str_rmap__[dtype],
            'default_trace_meta': default_trace_meta
        }
        instance.trace_meta = [
            copy.deepcopy(default_trace_meta) for _ in range(total_count)
        ]
        instance._init_memmap()
        instance.flush()
        return instance

    @property
    def _hdr_offs(self):
        return self.traces.size * self.traces.itemsize

    def flush(self):
        """
        Store (potentially modified) metadata in file.
        """
        assert self._file is not None and not self._file.closed
        hdr_offs = self._hdr_offs
        self._mmap.flush()
        self._file.seek(0, 2)
        old_len = self._file.tell()
        tail = self._encoder.encode(self.global_meta).encode('utf8') \
             + self._encoder.encode(self.trace_meta).encode('utf8') \
             + hdr_offs.to_bytes(self.__offset_marker_length__, 'little')
        # If the old tail was shorter or equal than the new one, we can just seek & write
        # Otherwise we must truncate the file first using mmap.resize();
        # Since the resize will have the same size, this should not change the allocation.
        self._file.seek(hdr_offs, 0)
        old_tail_len = old_len - hdr_offs
        if len(tail) < old_tail_len:
            # This hopefully doesn't reallocate since the old mmap size == new mmap size
            self._mmap.resize(hdr_offs)
            self._mmap.flush()
        self._file.write(tail)
        new_len = self._file.tell()
        self._file.flush()

    def _load_meta(self):
        """
        Load or reload metadata from disk.
        """
        self._file.seek(-self.__offset_marker_length__, 2)
        hdr_offs = int.from_bytes(
            self._file.read(self.__offset_marker_length__), 'little')
        self._file.seek(hdr_offs, 0)
        tail = self._file.read()[:-self.__offset_marker_length__].decode()
        self.global_meta, o = self._decoder.raw_decode(tail, 0)
        self.trace_meta, _ = self._decoder.raw_decode(tail, o)

    def save_batch(self, first_idx, traces, metadata):
        """
        Dump traces and metadata to the file at a specified offset.
        
        :param first_idx: offset in the traces file at which the first trace is to be stored
        :param traces: traces to be stored
        :param metadata: traces metadata: iterable of '(key, input, output)' tuples
        """
        for i, (trace, meta) in enumerate(zip(traces, metadata)):
            tracelen = len(trace)
            if tracelen > self.max_length:
                raise ValueError(
                    f"Captured trace longer ({tracelen}) than expected ({self.max_length})."
                )
            self.traces[first_idx + i, :tracelen] = trace[:]
            if tracelen < self.max_length:
                self.traces[first_idx + i,tracelen:] \
                    = np.zeros(self.max_length - tracelen)
            trace_meta = self.trace_meta[first_idx + i]
            trace_meta['key'] = meta[0]
            trace_meta['txi'] = meta[1]
            trace_meta['txo'] = meta[2]
            trace_meta['captured'] = True
        self.flush()

    @classmethod
    def convert_from_v1(cls, v1: 'TracesFileProxy_v1') -> 'TracesFileProxy_v2':
        file: TracesFileProxy_v2 = cls.create(v1.fname, v1.total_count,
                                              v1.max_length, v1.dtype)
        file.traces[:, :] = v1.traces[:, :]
        for i, (key, txi,
                txo) in enumerate(zip(v1.keys, v1.textins, v1.textouts)):
            file.trace_meta[i]['key'] = key
            file.trace_meta[i]['txi'] = txi
            file.trace_meta[i]['txo'] = txo
            file.trace_meta[i]['captured'] = True
        file.global_meta.update({
            'scope': 'unknown',
            'orig_meta': v1.meta_fname,
            'orig_traces': v1.traces_fname
        })
        file.flush()
        return file

    @classmethod
    def merge(cls, fname_out: str, fnames_in: Iterable[str]) -> None:
        """
        Merge input files into a single TFP file.
        The input files must all have the same data type and max_length.
        """
        fins = tuple(cls.load(fname, 'r') for fname in fnames_in)
        max_length = fins[0].max_length
        dtype = fins[0].dtype
        assert all(fin.max_length == max_length for fin in fins[1:])
        assert all(fin.dtype == dtype for fin in fins[1:])
        common_global_metadata = {
            'merged_from': {}
        }
        common_traces_metadata = []
        chunk_size = 1024*1024  # 1MB
        from tqdm.auto import tqdm
        with open(fname_out, 'wb') as fout:
            for fin in tqdm(fins, desc="Merging file"):
                common_global_metadata['merged_from'][fin.fname] = fin.global_meta
                common_global_metadata.update(fin.global_meta)
                common_traces_metadata.extend(fin.trace_meta)
                # Copy trace buffers as-is
                with tqdm(total=fin._hdr_offs, desc="Copy progress",
                          unit="B", unit_scale=True, unit_divisor=1024) as tbar:
                    for chunk_begin in range(0, fin._hdr_offs, chunk_size):
                        chunk_end = min(chunk_begin + chunk_size, fin._hdr_offs)
                        written = fout.write(fin._mmap[chunk_begin:chunk_end])
                        tbar.update(written)
            _hdr_offs = fout.tell()
            common_global_metadata['total_count'] = len(common_traces_metadata)
            tail = cls._encoder.encode(common_global_metadata).encode('utf8') \
                 + cls._encoder.encode(common_traces_metadata).encode('utf8') \
                 + _hdr_offs.to_bytes(cls.__offset_marker_length__, 'little')
            fout.write(tail)

    @classmethod
    def load(cls, fname, mode='r') -> 'TracesFileProxy_v2':
        """
        Load class instance from files. This operation opens a memmap to the trace file.
        
        :param fname: common prefix for the trace and metadata file name
                      (if the files are in a different directory than the execution context,
                      'fname' should be an absolute or relative path excluding file extensions)
        :returns: class instance
        """
        instance = cls(fname, mode)
        instance._load_meta()
        instance._init_memmap()
        return instance

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    class PropertyProxy:

        def __init__(self, meta_dict, key):
            self._meta_dict = meta_dict
            self._key = key

        def __getitem__(self, idx):
            return self._meta_dict[idx][self._key]

        def __setitem__(self, idx, value):
            self._meta_dict[idx][self._key] = value

        def __len__(self):
            return self._meta_dict.__len__()

        def __iter__(self):
            return (x[self._key] for x in self._meta_dict)

    @property
    def keys(self):
        return self.PropertyProxy(self.trace_meta, 'key')

    @property
    def textins(self):
        return self.PropertyProxy(self.trace_meta, 'txi')

    @property
    def textouts(self):
        return self.PropertyProxy(self.trace_meta, 'txo')


class TracesFileProxy_v1:
    """
    The class provides functions for storing and accessing traces on disk.
    Trace files are memmapped, allowing for gathering and processing large amount of data with limited RAM.
    Trace files use '.npy' extension.
    Keys, inputs and outputs are pickled in a separate metadata file with the same name but '.meta' extension.
    Note that due to memmapping, the performance depends on disk access speed
    and thus it may be noticeably slow on HDDs in some setups.

    This is the first version of the TracesFileProxy protocol and should be considered deprecated in favor of v2.
    
    :ivar fname: common prefix for the trace and metadata file name
    :ivar traces_fname: absolute path to the trace file
    :ivar meta_fname: absolute path to the metadata file
    :ivar total_count: number of traces
    :ivar max_length: maximal length of a single trace in points
    :ivar traces: memmapped trace container of dimension [:total_count,:max_length]
    :ivar dtype: data type of the underlying trace container, float64 by default
    :ivar keys: list of keys used in each trace
    :ivar textins: list of inputs used in each trace
    :ivar textouts: list of outputs used in each trace
    """

    def __init__(self, fname):
        self.fname = os.path.expanduser(fname)
        self.traces_fname = self.fname + '.npy'
        self.meta_fname = self.fname + '.meta'
        self.total_count = None
        self.max_length = None
        self.traces = None
        self.dtype = None
        self.keys = []
        self.textins = []
        self.textouts = []

    def close(self):
        self.flush()
        self.traces._mmap.close()

    @classmethod
    def create(cls, fname, total_count, max_length, dtype=np.float64) -> 'TracesFileProxy_v1':
        """
        Create class instance. This operation opens a memmap to the trace file.
        
        :param fname: common prefix for the trace and metadata file name
        :param total_count: number of traces
        :param max_length: maximal length of a single trace in points
        :returns: class instance
        """
        instance = cls(fname)
        instance.total_count = total_count
        instance.max_length = max_length
        instance.dtype = dtype
        instance.traces = np.memmap(instance.traces_fname,
                                    dtype=dtype,
                                    mode='w+',
                                    shape=(total_count, max_length),
                                    order='C')
        return instance

    def flush(self):
        """
        Store (potentially modified) metadata in file.
        """
        with open(self.meta_fname, 'wb') as output:
            pickle.dump((self.total_count, self.max_length, self.keys,
                         self.textins, self.textouts, self.dtype), output)

    def _load_meta(self):
        """
        Load or reload metadata from disk.
        """
        with open(self.meta_fname, 'rb') as meta_input:
            metadata = pickle.load(meta_input)
        # TODO: implement type versioning
        try:
            self.total_count, self.max_length, self.keys, self.textins, self.textouts, self.dtype = metadata
        except ValueError:
            self.total_count, self.max_length, self.keys, self.textins, self.textouts, self.dtype = *metadata, np.float64

    def save_batch(self, first_idx, traces, metadata):
        """
        Dump traces and metadata to the file at a specified offset.
        
        :param first_idx: offset in the traces file at which the first trace is to be stored
        :param traces: traces to be stored
        :param metadata: traces metadata: iterable of '(key, input, output)' tuples
        """
        for i, trace in enumerate(traces):
            tracelen = len(trace)
            if tracelen > self.max_length:
                raise ValueError(
                    f"Captured trace longer ({tracelen}) than expected ({self.max_length})."
                )
            self.traces[first_idx + i][:tracelen] = trace[:]
            if tracelen < self.max_length:
                self.traces[first_idx +
                            i][tracelen:] = np.zeros(self.max_length -
                                                     tracelen)
        self.traces.flush()
        # Split metadata
        keys = [meta[0] for meta in metadata]
        textins = [meta[1] for meta in metadata]
        textouts = [meta[2] for meta in metadata]
        self.keys += keys
        self.textins += textins
        self.textouts += textouts
        self.flush()

    @classmethod
    def load(cls, fname, mode='r') -> 'TracesFileProxy_v1':
        """
        Load class instance from files. This operation opens a memmap to the trace file.
        
        :param fname: common prefix for the trace and metadata file name
                      (if the files are in a different directory than the execution context,
                      'fname' should be an absolute or relative path excluding file extensions)
        :returns: class instance
        """
        instance = cls(fname)
        instance._load_meta()
        instance.traces = np.memmap(instance.traces_fname,
                                    dtype=instance.dtype,
                                    mode=mode,
                                    shape=(instance.total_count,
                                           instance.max_length))
        return instance

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
