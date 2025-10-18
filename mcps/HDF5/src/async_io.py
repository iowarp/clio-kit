#!/usr/bin/env python3
# /// script
# dependencies = [
#   "h5py>=3.9.0",
#   "numpy>=1.24.0,<2.0.0",
#   "aiofiles>=23.2.1"
# ]
# requires-python = ">=3.10"
# ///

"""
Asynchronous I/O enhancements for HDF5 operations with true parallel processing.
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import h5py
import numpy as np
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class ReadRequest:
    """A request for reading HDF5 data."""
    file_path: Path
    dataset_path: str
    slice_spec: Optional[Union[slice, Tuple[slice, ...]]] = None
    chunk_size: Optional[int] = None
    priority: int = 0  # Higher priority = processed first

@dataclass 
class ReadResult:
    """Result of an HDF5 read operation."""
    data: np.ndarray
    metadata: Dict[str, Any]
    read_time: float
    file_path: Path
    dataset_path: str

class ParallelHDF5Reader:
    """High-performance parallel HDF5 reader with async interface."""
    
    def __init__(self, max_workers: int = None, chunk_size: int = 1024*1024):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.chunk_size = chunk_size
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._file_locks: Dict[Path, threading.RLock] = {}
        self._file_handles: Dict[Path, h5py.File] = {}
        self._stats = {
            'reads_completed': 0,
            'bytes_read': 0,
            'total_read_time': 0.0,
            'cache_hits': 0,
            'parallel_reads': 0
        }
        
    def __del__(self):
        """Cleanup resources."""
        self.close()
        
    def close(self):
        """Close all file handles and shutdown executor."""
        for file_handle in self._file_handles.values():
            try:
                file_handle.close()
            except:
                pass
        self._file_handles.clear()
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
    
    def _get_file_lock(self, file_path: Path) -> threading.RLock:
        """Get or create a lock for the given file."""
        if file_path not in self._file_locks:
            self._file_locks[file_path] = threading.RLock()
        return self._file_locks[file_path]
    
    def _get_file_handle(self, file_path: Path, mode: str = 'r') -> h5py.File:
        """Get or create a file handle for the given path."""
        lock = self._get_file_lock(file_path)
        
        with lock:
            if file_path not in self._file_handles:
                try:
                    # Open with SWMR (Single Writer Multiple Reader) mode for parallel access
                    self._file_handles[file_path] = h5py.File(
                        file_path, mode, 
                        swmr=True if mode == 'r' else False,
                        libver='latest'
                    )
                except Exception as e:
                    logger.error(f"Failed to open HDF5 file {file_path}: {e}")
                    raise
                    
            return self._file_handles[file_path]
    
    def _read_dataset_chunk(self, request: ReadRequest) -> ReadResult:
        """Read a dataset chunk in a thread-safe manner."""
        start_time = time.time()
        
        try:
            file_handle = self._get_file_handle(request.file_path)
            lock = self._get_file_lock(request.file_path)
            
            with lock:
                if request.dataset_path not in file_handle:
                    raise KeyError(f"Dataset {request.dataset_path} not found in {request.file_path}")
                
                dataset = file_handle[request.dataset_path]
                
                # Apply slice if specified
                if request.slice_spec is not None:
                    data = dataset[request.slice_spec]
                else:
                    data = dataset[()]
                
                # Get metadata
                metadata = {
                    'shape': dataset.shape,
                    'dtype': str(dataset.dtype),
                    'size_bytes': dataset.size * dataset.dtype.itemsize,
                    'chunks': dataset.chunks,
                    'compression': dataset.compression,
                    'shuffle': dataset.shuffle,
                    'fletcher32': dataset.fletcher32,
                    'attributes': dict(dataset.attrs)
                }
                
                read_time = time.time() - start_time
                
                # Update stats
                self._stats['reads_completed'] += 1
                self._stats['bytes_read'] += data.nbytes
                self._stats['total_read_time'] += read_time
                
                return ReadResult(
                    data=data,
                    metadata=metadata,
                    read_time=read_time,
                    file_path=request.file_path,
                    dataset_path=request.dataset_path
                )
                
        except Exception as e:
            logger.error(f"Error reading {request.dataset_path} from {request.file_path}: {e}")
            raise
    
    async def read_dataset(self, file_path: Path, dataset_path: str, 
                          slice_spec: Optional[Union[slice, Tuple[slice, ...]]] = None) -> ReadResult:
        """Read a single dataset asynchronously."""
        request = ReadRequest(
            file_path=file_path,
            dataset_path=dataset_path,
            slice_spec=slice_spec
        )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, 
            self._read_dataset_chunk, 
            request
        )
        
        return result
    
    async def read_datasets_parallel(self, requests: List[ReadRequest]) -> List[ReadResult]:
        """Read multiple datasets in parallel."""
        if not requests:
            return []
        
        # Sort by priority (higher first)
        sorted_requests = sorted(requests, key=lambda r: r.priority, reverse=True)
        
        loop = asyncio.get_event_loop()
        
        # Submit all requests to executor
        futures = [
            loop.run_in_executor(self.executor, self._read_dataset_chunk, request)
            for request in sorted_requests
        ]
        
        self._stats['parallel_reads'] += 1
        
        # Wait for all to complete
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to read {sorted_requests[i].dataset_path}: {result}")
                # Could optionally continue with partial results
                raise result
            else:
                final_results.append(result)
        
        return final_results
    
    async def read_dataset_chunked(self, file_path: Path, dataset_path: str, 
                                  chunk_size: Optional[int] = None) -> List[ReadResult]:
        """Read a large dataset in chunks for memory efficiency."""
        chunk_size = chunk_size or self.chunk_size
        
        # First, get dataset info
        file_handle = self._get_file_handle(file_path)
        dataset = file_handle[dataset_path]
        
        total_size = dataset.size
        if total_size * dataset.dtype.itemsize < chunk_size:
            # Small dataset, read all at once
            return [await self.read_dataset(file_path, dataset_path)]
        
        # Large dataset, create chunk requests
        shape = dataset.shape
        dtype_size = dataset.dtype.itemsize
        
        # Calculate optimal chunking strategy
        if len(shape) == 1:
            # 1D array
            elements_per_chunk = chunk_size // dtype_size
            requests = []
            
            for start in range(0, shape[0], elements_per_chunk):
                end = min(start + elements_per_chunk, shape[0])
                slice_spec = slice(start, end)
                
                requests.append(ReadRequest(
                    file_path=file_path,
                    dataset_path=dataset_path,
                    slice_spec=slice_spec,
                    priority=len(requests)  # Earlier chunks have higher priority
                ))
        
        elif len(shape) == 2:
            # 2D array - chunk by rows
            row_size = shape[1] * dtype_size
            rows_per_chunk = max(1, chunk_size // row_size)
            requests = []
            
            for start_row in range(0, shape[0], rows_per_chunk):
                end_row = min(start_row + rows_per_chunk, shape[0])
                slice_spec = (slice(start_row, end_row), slice(None))
                
                requests.append(ReadRequest(
                    file_path=file_path,
                    dataset_path=dataset_path,
                    slice_spec=slice_spec,
                    priority=len(requests)
                ))
        
        else:
            # Multi-dimensional - chunk along first dimension
            first_dim_size = np.prod(shape[1:]) * dtype_size
            slices_per_chunk = max(1, chunk_size // first_dim_size)
            requests = []
            
            for start in range(0, shape[0], slices_per_chunk):
                end = min(start + slices_per_chunk, shape[0])
                slice_spec = (slice(start, end),) + (slice(None),) * (len(shape) - 1)
                
                requests.append(ReadRequest(
                    file_path=file_path,
                    dataset_path=dataset_path,
                    slice_spec=slice_spec,
                    priority=len(requests)
                ))
        
        # Read all chunks in parallel
        return await self.read_datasets_parallel(requests)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._stats.copy()
        
        if stats['reads_completed'] > 0:
            stats['avg_read_time'] = stats['total_read_time'] / stats['reads_completed']
            stats['throughput_mbps'] = (stats['bytes_read'] / (1024 * 1024)) / stats['total_read_time'] if stats['total_read_time'] > 0 else 0
        else:
            stats['avg_read_time'] = 0
            stats['throughput_mbps'] = 0
            
        stats['active_workers'] = self.max_workers
        stats['open_files'] = len(self._file_handles)
        
        return stats

class AsyncHDF5Manager:
    """High-level async HDF5 manager with caching and batching."""
    
    def __init__(self, cache_size_mb: int = 1024, max_workers: int = None):
        self.reader = ParallelHDF5Reader(max_workers=max_workers)
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        self._cache: Dict[str, Tuple[np.ndarray, float]] = {}  # key -> (data, timestamp)
        self._cache_size_bytes = 0
        self._cache_lock = asyncio.Lock()
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        """Close the manager and cleanup resources."""
        self.reader.close()
        self._cache.clear()
        self._cache_size_bytes = 0
    
    def _make_cache_key(self, file_path: Path, dataset_path: str, 
                       slice_spec: Optional[Union[slice, Tuple[slice, ...]]]) -> str:
        """Create a cache key for the request."""
        slice_str = str(slice_spec) if slice_spec else "full"
        return f"{file_path}::{dataset_path}::{slice_str}"
    
    async def _evict_cache_if_needed(self, needed_bytes: int):
        """Evict items from cache if needed to make space."""
        if self._cache_size_bytes + needed_bytes <= self.cache_size_bytes:
            return
            
        # Sort by timestamp (LRU eviction)
        cache_items = [(k, v[1]) for k, v in self._cache.items()]
        cache_items.sort(key=lambda x: x[1])
        
        # Evict oldest items until we have enough space
        for key, _ in cache_items:
            if self._cache_size_bytes + needed_bytes <= self.cache_size_bytes:
                break
                
            data, _ = self._cache.pop(key)
            self._cache_size_bytes -= data.nbytes
    
    async def read_dataset_cached(self, file_path: Path, dataset_path: str,
                                 slice_spec: Optional[Union[slice, Tuple[slice, ...]]] = None,
                                 use_cache: bool = True) -> ReadResult:
        """Read dataset with caching support."""
        cache_key = self._make_cache_key(file_path, dataset_path, slice_spec)
        
        # Check cache first
        if use_cache:
            async with self._cache_lock:
                if cache_key in self._cache:
                    data, timestamp = self._cache[cache_key]
                    self.reader._stats['cache_hits'] += 1
                    
                    # Create result from cached data
                    return ReadResult(
                        data=data.copy(),  # Return copy to prevent modification
                        metadata={'cached': True, 'cache_timestamp': timestamp},
                        read_time=0.0,
                        file_path=file_path,
                        dataset_path=dataset_path
                    )
        
        # Not in cache, read from file
        result = await self.reader.read_dataset(file_path, dataset_path, slice_spec)
        
        # Add to cache if enabled and data is not too large
        if use_cache and result.data.nbytes < self.cache_size_bytes // 4:  # Don't cache items larger than 1/4 of cache
            async with self._cache_lock:
                await self._evict_cache_if_needed(result.data.nbytes)
                self._cache[cache_key] = (result.data.copy(), time.time())
                self._cache_size_bytes += result.data.nbytes
        
        return result
    
    async def batch_read_datasets(self, requests: List[Tuple[Path, str, Optional[Union[slice, Tuple[slice, ...]]]]],
                                 use_cache: bool = True) -> List[ReadResult]:
        """Read multiple datasets with optimal batching and caching."""
        
        # Check cache for all requests first
        cached_results = {}
        uncached_requests = []
        
        if use_cache:
            async with self._cache_lock:
                for i, (file_path, dataset_path, slice_spec) in enumerate(requests):
                    cache_key = self._make_cache_key(file_path, dataset_path, slice_spec)
                    if cache_key in self._cache:
                        data, timestamp = self._cache[cache_key]
                        cached_results[i] = ReadResult(
                            data=data.copy(),
                            metadata={'cached': True, 'cache_timestamp': timestamp},
                            read_time=0.0,
                            file_path=file_path,
                            dataset_path=dataset_path
                        )
                        self.reader._stats['cache_hits'] += 1
                    else:
                        uncached_requests.append((i, file_path, dataset_path, slice_spec))
        else:
            uncached_requests = [(i, *req) for i, req in enumerate(requests)]
        
        # Read uncached data in parallel
        uncached_results = {}
        if uncached_requests:
            read_requests = [
                ReadRequest(
                    file_path=file_path,
                    dataset_path=dataset_path,
                    slice_spec=slice_spec,
                    priority=len(uncached_requests) - i  # Earlier requests have higher priority
                )
                for i, file_path, dataset_path, slice_spec in uncached_requests
            ]
            
            read_results = await self.reader.read_datasets_parallel(read_requests)
            
            # Cache results and create result mapping
            if use_cache:
                async with self._cache_lock:
                    for (orig_idx, file_path, dataset_path, slice_spec), result in zip(uncached_requests, read_results):
                        uncached_results[orig_idx] = result
                        
                        # Cache if appropriate size
                        if result.data.nbytes < self.cache_size_bytes // 4:
                            cache_key = self._make_cache_key(file_path, dataset_path, slice_spec)
                            await self._evict_cache_if_needed(result.data.nbytes)
                            self._cache[cache_key] = (result.data.copy(), time.time())
                            self._cache_size_bytes += result.data.nbytes
            else:
                for (orig_idx, _, _, _), result in zip(uncached_requests, read_results):
                    uncached_results[orig_idx] = result
        
        # Combine cached and uncached results in original order
        final_results = []
        for i in range(len(requests)):
            if i in cached_results:
                final_results.append(cached_results[i])
            else:
                final_results.append(uncached_results[i])
        
        return final_results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_entries': len(self._cache),
            'cache_size_bytes': self._cache_size_bytes,
            'cache_size_mb': self._cache_size_bytes / (1024 * 1024),
            'cache_utilization': self._cache_size_bytes / self.cache_size_bytes if self.cache_size_bytes > 0 else 0,
            'max_cache_size_mb': self.cache_size_bytes / (1024 * 1024)
        }

# Global instance for easy access
_global_async_manager: Optional[AsyncHDF5Manager] = None

def get_async_manager(cache_size_mb: int = 1024, max_workers: int = None) -> AsyncHDF5Manager:
    """Get the global async HDF5 manager instance."""
    global _global_async_manager
    
    if _global_async_manager is None:
        _global_async_manager = AsyncHDF5Manager(
            cache_size_mb=cache_size_mb,
            max_workers=max_workers
        )
    
    return _global_async_manager

def close_async_manager():
    """Close and cleanup the global async manager."""
    global _global_async_manager
    
    if _global_async_manager:
        _global_async_manager.close()
        _global_async_manager = None