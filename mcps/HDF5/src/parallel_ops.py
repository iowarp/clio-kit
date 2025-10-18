#!/usr/bin/env python3
# /// script
# dependencies = [
#   "h5py>=3.9.0",
#   "numpy>=1.24.0,<2.0.0",
#   "dask>=2023.0.0",
#   "psutil>=5.9.0"
# ]
# requires-python = ">=3.10"
# ///

"""
Parallel processing and batch operations for HDF5 datasets with CPU optimization.
"""
import asyncio
import concurrent.futures
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import h5py
import time
import logging
from functools import partial
import dask.array as da
import psutil

logger = logging.getLogger(__name__)

@dataclass
class BatchOperation:
    """Represents a batch operation on HDF5 datasets."""
    operation_type: str  # 'read', 'compute', 'analyze', 'transform'
    files: List[Path]
    datasets: List[str]
    function: Callable
    args: Tuple = ()
    kwargs: Dict[str, Any] = None
    priority: int = 0
    chunk_size: Optional[int] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

@dataclass
class BatchResult:
    """Result of a batch operation."""
    operation_id: str
    results: List[Any]
    execution_time: float
    memory_peak: float
    cpu_time: float
    success: bool = True
    error: Optional[str] = None

class CPUOptimizedProcessor:
    """CPU-optimized processor for numerical operations on HDF5 data."""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Optimize for the system
        self.optimal_workers = min(self.cpu_count, 8)  # Don't exceed 8 for most workloads
        self.chunk_size = self._calculate_optimal_chunk_size()
        
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on system resources."""
        # Target ~256MB per chunk, but adjust based on available memory
        base_chunk_mb = 256
        
        # Scale based on available memory
        if self.memory_gb >= 32:
            base_chunk_mb = 512
        elif self.memory_gb >= 16:
            base_chunk_mb = 384
        elif self.memory_gb <= 8:
            base_chunk_mb = 128
            
        return base_chunk_mb * 1024 * 1024  # Convert to bytes
    
    def vectorized_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Compute statistics using vectorized operations."""
        # Use numpy's optimized functions
        flat_data = data.ravel()
        
        # Basic stats - all vectorized
        mean_val = np.mean(flat_data)
        std_val = np.std(flat_data)
        min_val = np.min(flat_data)
        max_val = np.max(flat_data)
        
        # Advanced stats
        median_val = np.median(flat_data)
        
        # Percentiles (vectorized)
        percentiles = np.percentile(flat_data, [25, 75, 90, 95, 99])
        
        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(min_val),
            'max': float(max_val),
            'median': float(median_val),
            'q25': float(percentiles[0]),
            'q75': float(percentiles[1]),
            'q90': float(percentiles[2]),
            'q95': float(percentiles[3]),
            'q99': float(percentiles[4]),
            'size': int(data.size),
            'shape': data.shape,
            'dtype': str(data.dtype)
        }
    
    def vectorized_transform(self, data: np.ndarray, operation: str, **kwargs) -> np.ndarray:
        """Apply vectorized transformations to data."""
        
        if operation == 'normalize':
            # Z-score normalization
            mean = np.mean(data)
            std = np.std(data)
            return (data - mean) / std if std > 0 else data
            
        elif operation == 'minmax_scale':
            # Min-max scaling to [0, 1]
            min_val = np.min(data)
            max_val = np.max(data)
            range_val = max_val - min_val
            return (data - min_val) / range_val if range_val > 0 else data
            
        elif operation == 'log_transform':
            # Log transformation (handle negatives)
            min_val = np.min(data)
            offset = abs(min_val) + 1 if min_val <= 0 else 0
            return np.log(data + offset)
            
        elif operation == 'power_transform':
            power = kwargs.get('power', 2)
            return np.power(data, power)
            
        elif operation == 'clip':
            min_val = kwargs.get('min', np.min(data))
            max_val = kwargs.get('max', np.max(data))
            return np.clip(data, min_val, max_val)
            
        elif operation == 'smooth':
            # Simple moving average smoothing
            window = kwargs.get('window', 3)
            if data.ndim == 1:
                return np.convolve(data, np.ones(window)/window, mode='same')
            else:
                # For multi-dimensional data, apply along last axis
                from scipy import ndimage
                return ndimage.uniform_filter(data, size=window)
                
        else:
            raise ValueError(f"Unknown operation: {operation}")

class ParallelBatchProcessor:
    """High-performance batch processor for HDF5 operations."""
    
    def __init__(self, max_workers: int = None, use_processes: bool = False):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.use_processes = use_processes
        self.cpu_processor = CPUOptimizedProcessor()
        
        # Choose executor type based on workload
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        self.active_operations: Dict[str, asyncio.Task] = {}
        self.operation_counter = 0
        
    def __del__(self):
        self.close()
        
    def close(self):
        """Close the executor and cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
    
    async def execute_batch(self, operation: BatchOperation) -> BatchResult:
        """Execute a batch operation with parallel processing."""
        
        operation_id = f"batch_{self.operation_counter}"
        self.operation_counter += 1
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            if operation.operation_type == 'read':
                results = await self._batch_read(operation)
            elif operation.operation_type == 'compute':
                results = await self._batch_compute(operation)
            elif operation.operation_type == 'analyze':
                results = await self._batch_analyze(operation)
            elif operation.operation_type == 'transform':
                results = await self._batch_transform(operation)
            else:
                raise ValueError(f"Unknown operation type: {operation.operation_type}")
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            return BatchResult(
                operation_id=operation_id,
                results=results,
                execution_time=end_time - start_time,
                memory_peak=(end_memory - start_memory) / (1024*1024),  # MB
                cpu_time=0.0,  # Would need more detailed profiling
                success=True
            )
            
        except Exception as e:
            logger.error(f"Batch operation {operation_id} failed: {e}")
            return BatchResult(
                operation_id=operation_id,
                results=[],
                execution_time=time.time() - start_time,
                memory_peak=0.0,
                cpu_time=0.0,
                success=False,
                error=str(e)
            )
    
    async def _batch_read(self, operation: BatchOperation) -> List[Any]:
        """Execute batch read operations in parallel."""
        
        def read_single_dataset(file_path: Path, dataset_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
            """Read a single dataset - used in parallel execution."""
            try:
                with h5py.File(file_path, 'r', swmr=True) as f:
                    if dataset_path not in f:
                        raise KeyError(f"Dataset {dataset_path} not found in {file_path}")
                    
                    dataset = f[dataset_path]
                    data = dataset[()]
                    
                    metadata = {
                        'shape': dataset.shape,
                        'dtype': str(dataset.dtype),
                        'size_bytes': dataset.size * dataset.dtype.itemsize,
                        'chunks': dataset.chunks,
                        'compression': dataset.compression,
                        'attributes': dict(dataset.attrs)
                    }
                    
                    return data, metadata
                    
            except Exception as e:
                logger.error(f"Failed to read {dataset_path} from {file_path}: {e}")
                raise
        
        # Create tasks for parallel execution
        loop = asyncio.get_event_loop()
        
        tasks = []
        for file_path in operation.files:
            for dataset_path in operation.datasets:
                task = loop.run_in_executor(
                    self.executor,
                    read_single_dataset,
                    file_path,
                    dataset_path
                )
                tasks.append(task)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Read task failed: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def _batch_compute(self, operation: BatchOperation) -> List[Any]:
        """Execute batch compute operations using dask for large datasets."""
        
        def compute_on_dataset(file_path: Path, dataset_path: str, 
                             compute_func: Callable, *args, **kwargs) -> Any:
            """Compute on a single dataset using dask array."""
            try:
                with h5py.File(file_path, 'r', swmr=True) as f:
                    dataset = f[dataset_path]
                    
                    # Create dask array for large datasets
                    if dataset.size > 10**6:  # > 1M elements
                        # Use dask for large datasets
                        dask_array = da.from_array(dataset, chunks=self.cpu_processor.chunk_size)
                        result = compute_func(dask_array, *args, **kwargs)
                        
                        # Compute the result
                        if hasattr(result, 'compute'):
                            return result.compute()
                        else:
                            return result
                    else:
                        # Small datasets - direct computation
                        data = dataset[()]
                        return compute_func(data, *args, **kwargs)
                        
            except Exception as e:
                logger.error(f"Compute failed on {dataset_path} from {file_path}: {e}")
                raise
        
        # Execute computations in parallel
        loop = asyncio.get_event_loop()
        
        tasks = []
        for file_path in operation.files:
            for dataset_path in operation.datasets:
                task = loop.run_in_executor(
                    self.executor,
                    compute_on_dataset,
                    file_path,
                    dataset_path,
                    operation.function,
                    *operation.args,
                    **operation.kwargs
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Compute task failed: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def _batch_analyze(self, operation: BatchOperation) -> List[Any]:
        """Execute batch analysis operations with CPU optimization."""
        
        def analyze_dataset(file_path: Path, dataset_path: str) -> Dict[str, Any]:
            """Analyze a single dataset with optimized statistics."""
            try:
                with h5py.File(file_path, 'r', swmr=True) as f:
                    dataset = f[dataset_path]
                    data = dataset[()]
                    
                    # Use CPU-optimized statistics
                    stats = self.cpu_processor.vectorized_stats(data)
                    
                    # Add file and dataset info
                    stats.update({
                        'file_path': str(file_path),
                        'dataset_path': dataset_path,
                        'analysis_time': time.time()
                    })
                    
                    return stats
                    
            except Exception as e:
                logger.error(f"Analysis failed on {dataset_path} from {file_path}: {e}")
                raise
        
        # Execute analysis in parallel
        loop = asyncio.get_event_loop()
        
        tasks = []
        for file_path in operation.files:
            for dataset_path in operation.datasets:
                task = loop.run_in_executor(
                    self.executor,
                    analyze_dataset,
                    file_path,
                    dataset_path
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Analysis task failed: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def _batch_transform(self, operation: BatchOperation) -> List[Any]:
        """Execute batch transformation operations."""
        
        def transform_dataset(file_path: Path, dataset_path: str,
                            transform_op: str, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
            """Transform a single dataset."""
            try:
                with h5py.File(file_path, 'r', swmr=True) as f:
                    dataset = f[dataset_path]
                    data = dataset[()]
                    
                    # Apply CPU-optimized transformation
                    transformed_data = self.cpu_processor.vectorized_transform(
                        data, transform_op, **kwargs
                    )
                    
                    metadata = {
                        'original_shape': data.shape,
                        'transformed_shape': transformed_data.shape,
                        'original_dtype': str(data.dtype),
                        'transformed_dtype': str(transformed_data.dtype),
                        'transformation': transform_op,
                        'parameters': kwargs,
                        'file_path': str(file_path),
                        'dataset_path': dataset_path
                    }
                    
                    return transformed_data, metadata
                    
            except Exception as e:
                logger.error(f"Transform failed on {dataset_path} from {file_path}: {e}")
                raise
        
        # Get transform operation from kwargs
        transform_op = operation.kwargs.get('operation', 'normalize')
        transform_params = {k: v for k, v in operation.kwargs.items() if k != 'operation'}
        
        # Execute transformations in parallel
        loop = asyncio.get_event_loop()
        
        tasks = []
        for file_path in operation.files:
            for dataset_path in operation.datasets:
                task = loop.run_in_executor(
                    self.executor,
                    transform_dataset,
                    file_path,
                    dataset_path,
                    transform_op,
                    **transform_params
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Transform task failed: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def execute_multiple_batches(self, operations: List[BatchOperation]) -> List[BatchResult]:
        """Execute multiple batch operations concurrently."""
        
        # Sort by priority (higher priority first)
        sorted_operations = sorted(operations, key=lambda op: op.priority, reverse=True)
        
        # Execute all batches concurrently
        tasks = [self.execute_batch(op) for op in sorted_operations]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the batch processor."""
        return {
            'processor_config': {
                'max_workers': self.max_workers,
                'use_processes': self.use_processes,
                'optimal_chunk_size_mb': self.cpu_processor.chunk_size / (1024*1024)
            },
            'system_info': {
                'cpu_count_physical': self.cpu_processor.cpu_count,
                'memory_gb': self.cpu_processor.memory_gb,
                'optimal_workers': self.cpu_processor.optimal_workers
            },
            'active_operations': len(self.active_operations),
            'total_operations': self.operation_counter
        }

# Convenience functions for common operations

async def batch_read_datasets(files: List[Path], datasets: List[str], 
                            max_workers: int = None) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """Convenience function for batch reading datasets."""
    
    processor = ParallelBatchProcessor(max_workers=max_workers)
    
    try:
        operation = BatchOperation(
            operation_type='read',
            files=files,
            datasets=datasets,
            function=None  # Not used for read operations
        )
        
        result = await processor.execute_batch(operation)
        return result.results if result.success else []
        
    finally:
        processor.close()

async def batch_analyze_datasets(files: List[Path], datasets: List[str],
                               max_workers: int = None) -> List[Dict[str, Any]]:
    """Convenience function for batch analyzing datasets."""
    
    processor = ParallelBatchProcessor(max_workers=max_workers)
    
    try:
        operation = BatchOperation(
            operation_type='analyze',
            files=files,
            datasets=datasets,
            function=None  # Analysis uses built-in statistics
        )
        
        result = await processor.execute_batch(operation)
        return result.results if result.success else []
        
    finally:
        processor.close()

async def batch_transform_datasets(files: List[Path], datasets: List[str],
                                 transform_operation: str, max_workers: int = None,
                                 **transform_params) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """Convenience function for batch transforming datasets."""
    
    processor = ParallelBatchProcessor(max_workers=max_workers)
    
    try:
        operation = BatchOperation(
            operation_type='transform',
            files=files,
            datasets=datasets,
            function=None,  # Transform uses built-in operations
            kwargs={'operation': transform_operation, **transform_params}
        )
        
        result = await processor.execute_batch(operation)
        return result.results if result.success else []
        
    finally:
        processor.close()

# Global processor instance
_global_processor: Optional[ParallelBatchProcessor] = None

def get_batch_processor(max_workers: int = None, use_processes: bool = False) -> ParallelBatchProcessor:
    """Get the global batch processor instance."""
    global _global_processor
    
    if _global_processor is None:
        _global_processor = ParallelBatchProcessor(
            max_workers=max_workers,
            use_processes=use_processes
        )
    
    return _global_processor

def close_batch_processor():
    """Close and cleanup the global batch processor."""
    global _global_processor
    
    if _global_processor:
        _global_processor.close()
        _global_processor = None