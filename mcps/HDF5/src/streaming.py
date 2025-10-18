#!/usr/bin/env python3
# /// script
# dependencies = [
#   "h5py>=3.9.0",
#   "numpy>=1.24.0,<2.0.0",
#   "aiofiles>=23.2.1",
#   "psutil>=5.9.0"
# ]
# requires-python = ">=3.10"
# ///

"""
Memory-efficient streaming for large HDF5 datasets with adaptive chunking.
"""
import asyncio
import os
from typing import AsyncIterator, Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import h5py
import time
import logging
import psutil
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class StreamChunk:
    """A chunk of data from a streaming operation."""
    data: np.ndarray
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]
    timestamp: float
    
    @property
    def is_last_chunk(self) -> bool:
        return self.chunk_index == self.total_chunks - 1
    
    @property
    def size_bytes(self) -> int:
        return self.data.nbytes
    
    @property
    def progress(self) -> float:
        return (self.chunk_index + 1) / self.total_chunks if self.total_chunks > 0 else 1.0

class StreamingStrategy(ABC):
    """Abstract base class for streaming strategies."""
    
    @abstractmethod
    def calculate_chunk_size(self, dataset_shape: Tuple[int, ...], 
                           dtype: np.dtype, available_memory: int) -> Tuple[int, ...]:
        """Calculate optimal chunk size for streaming."""
        pass
    
    @abstractmethod
    def get_chunk_slices(self, dataset_shape: Tuple[int, ...], 
                        chunk_shape: Tuple[int, ...]) -> List[Tuple[slice, ...]]:
        """Generate slice objects for each chunk."""
        pass

class RowWiseStreamingStrategy(StreamingStrategy):
    """Stream data row by row (along first dimension)."""
    
    def calculate_chunk_size(self, dataset_shape: Tuple[int, ...], 
                           dtype: np.dtype, available_memory: int) -> Tuple[int, ...]:
        """Calculate chunk size for row-wise streaming."""
        if len(dataset_shape) == 1:
            # 1D array - chunk by elements
            elements_per_chunk = available_memory // dtype.itemsize
            chunk_size = min(elements_per_chunk, dataset_shape[0])
            return (chunk_size,)
        
        # Multi-dimensional - chunk along first dimension
        row_size = np.prod(dataset_shape[1:]) * dtype.itemsize
        rows_per_chunk = max(1, available_memory // row_size)
        chunk_rows = min(rows_per_chunk, dataset_shape[0])
        
        return (chunk_rows,) + dataset_shape[1:]
    
    def get_chunk_slices(self, dataset_shape: Tuple[int, ...], 
                        chunk_shape: Tuple[int, ...]) -> List[Tuple[slice, ...]]:
        """Generate row-wise chunk slices."""
        slices = []
        
        if len(dataset_shape) == 1:
            # 1D array
            chunk_size = chunk_shape[0]
            for start in range(0, dataset_shape[0], chunk_size):
                end = min(start + chunk_size, dataset_shape[0])
                slices.append((slice(start, end),))
        else:
            # Multi-dimensional
            chunk_rows = chunk_shape[0]
            for start_row in range(0, dataset_shape[0], chunk_rows):
                end_row = min(start_row + chunk_rows, dataset_shape[0])
                slice_tuple = (slice(start_row, end_row),) + (slice(None),) * (len(dataset_shape) - 1)
                slices.append(slice_tuple)
        
        return slices

class AdaptiveStreamingStrategy(StreamingStrategy):
    """Adaptive streaming that adjusts chunk size based on system performance."""
    
    def __init__(self, initial_chunk_mb: int = 64, min_chunk_mb: int = 16, max_chunk_mb: int = 512):
        self.initial_chunk_mb = initial_chunk_mb
        self.min_chunk_mb = min_chunk_mb
        self.max_chunk_mb = max_chunk_mb
        self.current_chunk_mb = initial_chunk_mb
        self.performance_history = []
        
    def calculate_chunk_size(self, dataset_shape: Tuple[int, ...], 
                           dtype: np.dtype, available_memory: int) -> Tuple[int, ...]:
        """Calculate adaptive chunk size based on performance history."""
        target_bytes = min(self.current_chunk_mb * 1024 * 1024, available_memory // 4)
        
        if len(dataset_shape) == 1:
            elements_per_chunk = target_bytes // dtype.itemsize
            chunk_size = min(elements_per_chunk, dataset_shape[0])
            return (chunk_size,)
        
        # Multi-dimensional - optimize for memory layout
        total_elements = np.prod(dataset_shape)
        target_elements = target_bytes // dtype.itemsize
        
        if target_elements >= total_elements:
            return dataset_shape  # Read entire dataset
        
        # Calculate chunk dimensions (prefer to keep later dimensions intact)
        chunk_shape = list(dataset_shape)
        
        # Reduce first dimension to fit target size
        elements_per_slice = np.prod(dataset_shape[1:])
        max_first_dim = target_elements // elements_per_slice
        chunk_shape[0] = min(max_first_dim, dataset_shape[0])
        
        return tuple(chunk_shape)
    
    def get_chunk_slices(self, dataset_shape: Tuple[int, ...], 
                        chunk_shape: Tuple[int, ...]) -> List[Tuple[slice, ...]]:
        """Generate adaptive chunk slices."""
        # Use row-wise strategy as base
        row_strategy = RowWiseStreamingStrategy()
        return row_strategy.get_chunk_slices(dataset_shape, chunk_shape)
    
    def update_performance(self, chunk_size_mb: float, throughput_mbps: float, memory_pressure: float):
        """Update chunk size based on performance feedback."""
        self.performance_history.append({
            'chunk_size_mb': chunk_size_mb,
            'throughput_mbps': throughput_mbps,
            'memory_pressure': memory_pressure,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
        
        # Adjust chunk size based on performance
        if len(self.performance_history) >= 3:
            recent_performance = self.performance_history[-3:]
            avg_throughput = sum(p['throughput_mbps'] for p in recent_performance) / len(recent_performance)
            avg_memory_pressure = sum(p['memory_pressure'] for p in recent_performance) / len(recent_performance)
            
            # Increase chunk size if throughput is good and memory pressure is low
            if avg_throughput > 100 and avg_memory_pressure < 0.7:
                self.current_chunk_mb = min(self.current_chunk_mb * 1.2, self.max_chunk_mb)
            # Decrease chunk size if memory pressure is high
            elif avg_memory_pressure > 0.8:
                self.current_chunk_mb = max(self.current_chunk_mb * 0.8, self.min_chunk_mb)

class HDF5DataStreamer:
    """Memory-efficient streamer for large HDF5 datasets."""
    
    def __init__(self, file_path: Path, dataset_path: str,
                 strategy: Optional[StreamingStrategy] = None,
                 chunk_size_mb: int = 64,
                 prefetch_chunks: int = 2):
        
        self.file_path = file_path
        self.dataset_path = dataset_path
        self.strategy = strategy or AdaptiveStreamingStrategy(initial_chunk_mb=chunk_size_mb)
        self.prefetch_chunks = prefetch_chunks
        
        # Dataset info (populated on first access)
        self._dataset_shape: Optional[Tuple[int, ...]] = None
        self._dataset_dtype: Optional[np.dtype] = None
        self._dataset_size_bytes: Optional[int] = None
        self._total_chunks: Optional[int] = None
        self._chunk_slices: Optional[List[Tuple[slice, ...]]] = None
        
        # Performance tracking
        self._start_time: Optional[float] = None
        self._bytes_streamed: int = 0
        self._chunks_streamed: int = 0
        
        # Prefetch buffer
        self._prefetch_buffer: Dict[int, StreamChunk] = {}
        self._prefetch_lock = asyncio.Lock()
        
    async def _initialize_dataset_info(self):
        """Initialize dataset information."""
        if self._dataset_shape is not None:
            return  # Already initialized
            
        def get_dataset_info():
            with h5py.File(self.file_path, 'r', swmr=True) as f:
                if self.dataset_path not in f:
                    raise KeyError(f"Dataset {self.dataset_path} not found in {self.file_path}")
                
                dataset = f[self.dataset_path]
                return {
                    'shape': dataset.shape,
                    'dtype': dataset.dtype,
                    'size_bytes': dataset.size * dataset.dtype.itemsize,
                    'chunks': dataset.chunks,
                    'compression': dataset.compression,
                    'attributes': dict(dataset.attrs)
                }
        
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, get_dataset_info)
        
        self._dataset_shape = info['shape']
        self._dataset_dtype = info['dtype']
        self._dataset_size_bytes = info['size_bytes']
        
        # Calculate optimal chunking
        available_memory = self._get_available_memory()
        chunk_shape = self.strategy.calculate_chunk_size(
            self._dataset_shape, self._dataset_dtype, available_memory
        )
        
        self._chunk_slices = self.strategy.get_chunk_slices(self._dataset_shape, chunk_shape)
        self._total_chunks = len(self._chunk_slices)
        
        logger.info(f"Initialized streaming for {self.dataset_path}: "
                   f"shape={self._dataset_shape}, chunks={self._total_chunks}, "
                   f"chunk_shape={chunk_shape}")
    
    def _get_available_memory(self) -> int:
        """Get available memory for streaming."""
        memory = psutil.virtual_memory()
        # Use up to 25% of available memory for streaming
        available = int(memory.available * 0.25)
        # But at least 64MB
        return max(available, 64 * 1024 * 1024)
    
    async def _read_chunk(self, chunk_index: int) -> StreamChunk:
        """Read a single chunk from the dataset."""
        if chunk_index >= self._total_chunks:
            raise IndexError(f"Chunk index {chunk_index} out of range (max: {self._total_chunks-1})")
        
        chunk_slice = self._chunk_slices[chunk_index]
        
        def read_chunk_data():
            with h5py.File(self.file_path, 'r', swmr=True) as f:
                dataset = f[self.dataset_path]
                data = dataset[chunk_slice]
                
                metadata = {
                    'file_path': str(self.file_path),
                    'dataset_path': self.dataset_path,
                    'chunk_slice': chunk_slice,
                    'original_shape': dataset.shape,
                    'chunk_shape': data.shape,
                    'dtype': str(data.dtype)
                }
                
                return data, metadata
        
        loop = asyncio.get_event_loop()
        data, metadata = await loop.run_in_executor(None, read_chunk_data)
        
        return StreamChunk(
            data=data,
            chunk_index=chunk_index,
            total_chunks=self._total_chunks,
            metadata=metadata,
            timestamp=time.time()
        )
    
    async def _prefetch_chunk(self, chunk_index: int):
        """Prefetch a chunk into the buffer."""
        if chunk_index >= self._total_chunks or chunk_index in self._prefetch_buffer:
            return
            
        try:
            chunk = await self._read_chunk(chunk_index)
            
            async with self._prefetch_lock:
                # Only keep a limited number of prefetched chunks
                if len(self._prefetch_buffer) >= self.prefetch_chunks:
                    # Remove oldest chunk
                    oldest_index = min(self._prefetch_buffer.keys())
                    del self._prefetch_buffer[oldest_index]
                
                self._prefetch_buffer[chunk_index] = chunk
                logger.debug(f"Prefetched chunk {chunk_index}")
                
        except Exception as e:
            logger.error(f"Failed to prefetch chunk {chunk_index}: {e}")
    
    async def stream_chunks(self) -> AsyncIterator[StreamChunk]:
        """Stream dataset chunks asynchronously."""
        await self._initialize_dataset_info()
        
        self._start_time = time.time()
        self._bytes_streamed = 0
        self._chunks_streamed = 0
        
        try:
            for chunk_index in range(self._total_chunks):
                # Check prefetch buffer first
                chunk = None
                async with self._prefetch_lock:
                    if chunk_index in self._prefetch_buffer:
                        chunk = self._prefetch_buffer.pop(chunk_index)
                        logger.debug(f"Using prefetched chunk {chunk_index}")
                
                # If not prefetched, read directly
                if chunk is None:
                    chunk = await self._read_chunk(chunk_index)
                
                # Update statistics
                self._bytes_streamed += chunk.size_bytes
                self._chunks_streamed += 1
                
                # Start prefetching next chunks
                for prefetch_offset in range(1, self.prefetch_chunks + 1):
                    next_chunk_index = chunk_index + prefetch_offset
                    if next_chunk_index < self._total_chunks:
                        asyncio.create_task(self._prefetch_chunk(next_chunk_index))
                
                # Update adaptive strategy performance
                if isinstance(self.strategy, AdaptiveStreamingStrategy):
                    elapsed_time = time.time() - self._start_time
                    throughput_mbps = (self._bytes_streamed / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0
                    memory_pressure = 1.0 - (psutil.virtual_memory().available / psutil.virtual_memory().total)
                    
                    self.strategy.update_performance(
                        chunk.size_bytes / (1024 * 1024),
                        throughput_mbps,
                        memory_pressure
                    )
                
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming failed for {self.dataset_path}: {e}")
            raise
        
        finally:
            # Clear prefetch buffer
            async with self._prefetch_lock:
                self._prefetch_buffer.clear()
    
    async def stream_all(self) -> np.ndarray:
        """Stream all chunks and concatenate into a single array."""
        chunks = []
        
        async for chunk in self.stream_chunks():
            chunks.append(chunk.data)
        
        if not chunks:
            raise ValueError("No chunks were streamed")
        
        # Concatenate along the first dimension
        return np.concatenate(chunks, axis=0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get streaming performance statistics."""
        elapsed_time = time.time() - self._start_time if self._start_time else 0
        
        return {
            'dataset_info': {
                'file_path': str(self.file_path),
                'dataset_path': self.dataset_path,
                'shape': self._dataset_shape,
                'dtype': str(self._dataset_dtype) if self._dataset_dtype else None,
                'size_bytes': self._dataset_size_bytes,
                'size_mb': self._dataset_size_bytes / (1024 * 1024) if self._dataset_size_bytes else None
            },
            'streaming_stats': {
                'total_chunks': self._total_chunks,
                'chunks_streamed': self._chunks_streamed,
                'bytes_streamed': self._bytes_streamed,
                'mb_streamed': self._bytes_streamed / (1024 * 1024),
                'elapsed_time_seconds': elapsed_time,
                'throughput_mbps': (self._bytes_streamed / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0,
                'chunks_per_second': self._chunks_streamed / elapsed_time if elapsed_time > 0 else 0
            },
            'prefetch_stats': {
                'prefetch_buffer_size': len(self._prefetch_buffer),
                'max_prefetch_chunks': self.prefetch_chunks
            },
            'system_info': {
                'available_memory_mb': psutil.virtual_memory().available / (1024 * 1024),
                'memory_usage_percent': psutil.virtual_memory().percent
            }
        }

class BatchStreamer:
    """Stream multiple datasets efficiently with resource management."""
    
    def __init__(self, max_concurrent_streams: int = 3, chunk_size_mb: int = 64):
        self.max_concurrent_streams = max_concurrent_streams
        self.chunk_size_mb = chunk_size_mb
        self.active_streams: Dict[str, HDF5DataStreamer] = {}
        self.stream_semaphore = asyncio.Semaphore(max_concurrent_streams)
        
    async def stream_datasets(self, dataset_specs: List[Tuple[Path, str]]) -> AsyncIterator[Tuple[str, StreamChunk]]:
        """Stream multiple datasets concurrently."""
        
        async def stream_single_dataset(file_path: Path, dataset_path: str):
            """Stream a single dataset and yield labeled chunks."""
            dataset_id = f"{file_path}::{dataset_path}"
            
            async with self.stream_semaphore:
                streamer = HDF5DataStreamer(
                    file_path=file_path,
                    dataset_path=dataset_path,
                    chunk_size_mb=self.chunk_size_mb
                )
                
                self.active_streams[dataset_id] = streamer
                
                try:
                    async for chunk in streamer.stream_chunks():
                        yield dataset_id, chunk
                finally:
                    self.active_streams.pop(dataset_id, None)
        
        # Create tasks for all datasets
        tasks = [
            stream_single_dataset(file_path, dataset_path)
            for file_path, dataset_path in dataset_specs
        ]
        
        # Merge all streams
        async def merge_streams():
            async_iterators = [task.__aiter__() for task in tasks]
            
            while async_iterators:
                # Get next chunk from each active iterator
                done_tasks = []
                
                for i, async_iter in enumerate(async_iterators):
                    try:
                        dataset_id, chunk = await async_iter.__anext__()
                        yield dataset_id, chunk
                    except StopAsyncIteration:
                        done_tasks.append(i)
                
                # Remove completed iterators
                for i in reversed(done_tasks):
                    async_iterators.pop(i)
        
        async for dataset_id, chunk in merge_streams():
            yield dataset_id, chunk
    
    def get_active_streams_stats(self) -> Dict[str, Any]:
        """Get statistics for all active streams."""
        stats = {}
        
        for dataset_id, streamer in self.active_streams.items():
            stats[dataset_id] = streamer.get_performance_stats()
        
        return {
            'active_streams': len(self.active_streams),
            'max_concurrent_streams': self.max_concurrent_streams,
            'stream_details': stats
        }

# Convenience functions

async def stream_dataset(file_path: Path, dataset_path: str, 
                        chunk_size_mb: int = 64) -> AsyncIterator[StreamChunk]:
    """Convenience function to stream a single dataset."""
    streamer = HDF5DataStreamer(file_path, dataset_path, chunk_size_mb=chunk_size_mb)
    
    async for chunk in streamer.stream_chunks():
        yield chunk

async def stream_dataset_to_array(file_path: Path, dataset_path: str,
                                 chunk_size_mb: int = 64) -> np.ndarray:
    """Convenience function to stream a dataset into a single array."""
    streamer = HDF5DataStreamer(file_path, dataset_path, chunk_size_mb=chunk_size_mb)
    return await streamer.stream_all()