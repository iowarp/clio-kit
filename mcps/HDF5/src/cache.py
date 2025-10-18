#!/usr/bin/env python3
# /// script
# dependencies = [
#   "h5py>=3.9.0",
#   "numpy>=1.24.0,<2.0.0",
#   "pydantic>=2.4.2,<3.0.0",
#   "psutil>=5.9.0"
# ]
# requires-python = ">=3.10"
# ///

"""
High-performance multi-level caching system with predictive prefetching.
"""
import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import h5py
import psutil
import logging
from collections import defaultdict, deque
import pickle
import hashlib

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache levels in order of speed/size."""
    L1_MEMORY = 1      # Fast, small memory cache
    L2_MEMORY = 2      # Larger memory cache
    L3_DISK = 3        # Disk-based cache (optional)

@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    data: np.ndarray
    metadata: Dict[str, Any]
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    size_bytes: int = 0
    level: CacheLevel = CacheLevel.L1_MEMORY
    
    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = self.data.nbytes

@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    bytes_cached: int = 0
    avg_access_time: float = 0.0
    cache_utilization: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def prefetch_accuracy(self) -> float:
        total = self.prefetch_hits + self.prefetch_misses
        return self.prefetch_hits / total if total > 0 else 0.0

class AccessPattern:
    """Tracks access patterns for predictive prefetching."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.access_history: deque = deque(maxlen=max_history)
        self.sequence_patterns: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
        self.dataset_groups: Dict[str, Set[str]] = defaultdict(set)
        self.access_frequencies: Dict[str, int] = defaultdict(int)
        
    def record_access(self, key: str, related_keys: Optional[List[str]] = None):
        """Record an access and update patterns."""
        self.access_history.append((key, time.time()))
        self.access_frequencies[key] += 1
        
        # Build sequence patterns (last 3 accesses predict next)
        if len(self.access_history) >= 4:
            recent_keys = [k for k, _ in list(self.access_history)[-4:]]
            sequence = tuple(recent_keys[:-1])
            next_key = recent_keys[-1]
            
            if next_key not in self.sequence_patterns[sequence]:
                self.sequence_patterns[sequence].append(next_key)
        
        # Group related datasets
        if related_keys:
            for related in related_keys:
                self.dataset_groups[key].add(related)
                self.dataset_groups[related].add(key)
    
    def predict_next_accesses(self, current_key: str, limit: int = 5) -> List[str]:
        """Predict likely next accesses for prefetching."""
        predictions = []
        
        # 1. Sequence-based prediction
        if len(self.access_history) >= 3:
            recent_keys = [k for k, _ in list(self.access_history)[-3:]]
            sequence = tuple(recent_keys)
            
            if sequence in self.sequence_patterns:
                predictions.extend(self.sequence_patterns[sequence][:limit//2])
        
        # 2. Group-based prediction (related datasets)
        if current_key in self.dataset_groups:
            related = list(self.dataset_groups[current_key])
            # Sort by access frequency
            related.sort(key=lambda k: self.access_frequencies[k], reverse=True)
            predictions.extend(related[:limit//2])
        
        # 3. Frequency-based prediction
        frequent_keys = sorted(
            self.access_frequencies.keys(),
            key=lambda k: self.access_frequencies[k],
            reverse=True
        )
        predictions.extend([k for k in frequent_keys[:limit] if k not in predictions])
        
        return predictions[:limit]

class MultiLevelCache:
    """Multi-level cache system with intelligent eviction and prefetching."""
    
    def __init__(self, 
                 l1_size_mb: int = 512,      # Fast memory cache
                 l2_size_mb: int = 2048,     # Larger memory cache
                 l3_size_mb: int = 8192,     # Disk cache (optional)
                 enable_prefetch: bool = True,
                 prefetch_threads: int = 2):
        
        self.l1_size_bytes = l1_size_mb * 1024 * 1024
        self.l2_size_bytes = l2_size_mb * 1024 * 1024
        self.l3_size_bytes = l3_size_mb * 1024 * 1024
        self.enable_prefetch = enable_prefetch
        
        # Cache storage
        self.l1_cache: Dict[str, CacheEntry] = {}
        self.l2_cache: Dict[str, CacheEntry] = {}
        self.l3_cache: Dict[str, CacheEntry] = {}  # Will implement if needed
        
        # Cache sizes
        self.l1_current_bytes = 0
        self.l2_current_bytes = 0
        self.l3_current_bytes = 0
        
        # Concurrency control
        self.l1_lock = asyncio.Lock()
        self.l2_lock = asyncio.Lock()
        self.l3_lock = asyncio.Lock()
        
        # Performance tracking
        self.stats = CacheStats()
        self.access_pattern = AccessPattern()
        
        # Prefetching
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        self.prefetch_tasks: List[asyncio.Task] = []
        self.prefetch_active = set()  # Keys being prefetched
        
        if enable_prefetch:
            self._start_prefetch_workers(prefetch_threads)
    
    def _start_prefetch_workers(self, num_workers: int):
        """Start background prefetch workers."""
        for i in range(num_workers):
            task = asyncio.create_task(self._prefetch_worker(f"prefetch-{i}"))
            self.prefetch_tasks.append(task)
    
    async def _prefetch_worker(self, worker_name: str):
        """Background worker for prefetching data."""
        logger.debug(f"Started prefetch worker: {worker_name}")
        
        while True:
            try:
                # Get prefetch request
                prefetch_func, key, *args = await self.prefetch_queue.get()
                
                if key in self.prefetch_active:
                    continue  # Already being prefetched
                    
                self.prefetch_active.add(key)
                
                try:
                    # Execute prefetch
                    result = await prefetch_func(*args)
                    if result:
                        logger.debug(f"Prefetched: {key}")
                        self.stats.prefetch_hits += 1
                    else:
                        self.stats.prefetch_misses += 1
                        
                except Exception as e:
                    logger.debug(f"Prefetch failed for {key}: {e}")
                    self.stats.prefetch_misses += 1
                    
                finally:
                    self.prefetch_active.discard(key)
                    self.prefetch_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prefetch worker {worker_name} error: {e}")
    
    def _make_cache_key(self, file_path: Path, dataset_path: str, 
                       slice_spec: Optional[Union[slice, Tuple[slice, ...]]]) -> str:
        """Create a standardized cache key."""
        slice_str = str(slice_spec) if slice_spec else "full"
        key = f"{file_path}::{dataset_path}::{slice_str}"
        return hashlib.md5(key.encode()).hexdigest()
    
    async def _evict_from_cache(self, cache: Dict[str, CacheEntry], 
                               cache_lock: asyncio.Lock,
                               current_bytes: int, max_bytes: int,
                               needed_bytes: int) -> int:
        """Evict entries from cache using LRU + LFU hybrid strategy."""
        
        if current_bytes + needed_bytes <= max_bytes:
            return current_bytes
            
        async with cache_lock:
            # Calculate scores for eviction (lower score = evict first)
            # Score = (access_count * recency_factor) / size_factor
            entries_with_scores = []
            now = time.time()
            
            for key, entry in cache.items():
                recency = max(0.1, 1.0 / (now - entry.last_access + 1))
                size_factor = entry.size_bytes / (1024 * 1024)  # MB
                score = (entry.access_count * recency) / max(1, size_factor)
                entries_with_scores.append((score, key, entry))
            
            # Sort by score (lowest first)
            entries_with_scores.sort(key=lambda x: x[0])
            
            # Evict until we have enough space
            bytes_freed = 0
            for score, key, entry in entries_with_scores:
                if current_bytes - bytes_freed + needed_bytes <= max_bytes:
                    break
                    
                del cache[key]
                bytes_freed += entry.size_bytes
                self.stats.evictions += 1
                logger.debug(f"Evicted {key} (score: {score:.2f}, size: {entry.size_bytes})")
            
            return current_bytes - bytes_freed
    
    async def get(self, file_path: Path, dataset_path: str,
                 slice_spec: Optional[Union[slice, Tuple[slice, ...]]] = None) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get data from cache hierarchy."""
        
        key = self._make_cache_key(file_path, dataset_path, slice_spec)
        self.access_pattern.record_access(key)
        
        start_time = time.time()
        
        # Check L1 cache first
        async with self.l1_lock:
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                entry.access_count += 1
                entry.last_access = time.time()
                self.stats.hits += 1
                self.stats.avg_access_time = (self.stats.avg_access_time + (time.time() - start_time)) / 2
                
                # Trigger prefetching
                if self.enable_prefetch:
                    await self._trigger_prefetch(key)
                
                return entry.data.copy(), entry.metadata
        
        # Check L2 cache
        async with self.l2_lock:
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                entry.access_count += 1
                entry.last_access = time.time()
                self.stats.hits += 1
                
                # Promote to L1 if it fits
                if entry.size_bytes <= self.l1_size_bytes // 4:  # Only promote smaller items
                    await self._promote_to_l1(key, entry)
                
                self.stats.avg_access_time = (self.stats.avg_access_time + (time.time() - start_time)) / 2
                
                # Trigger prefetching
                if self.enable_prefetch:
                    await self._trigger_prefetch(key)
                
                return entry.data.copy(), entry.metadata
        
        # Not found in any cache
        self.stats.misses += 1
        return None
    
    async def put(self, file_path: Path, dataset_path: str, data: np.ndarray,
                 metadata: Dict[str, Any],
                 slice_spec: Optional[Union[slice, Tuple[slice, ...]]] = None):
        """Put data into appropriate cache level."""
        
        key = self._make_cache_key(file_path, dataset_path, slice_spec)
        entry = CacheEntry(
            key=key,
            data=data.copy(),
            metadata=metadata,
            size_bytes=data.nbytes
        )
        
        # Decide which cache level based on size
        if entry.size_bytes <= self.l1_size_bytes // 8:  # Small items go to L1
            await self._put_l1(key, entry)
        elif entry.size_bytes <= self.l2_size_bytes // 4:  # Medium items go to L2
            await self._put_l2(key, entry)
        # Large items might not be cached or go to L3 (disk) if implemented
        
        self.stats.bytes_cached += entry.size_bytes
        
        # Update cache utilization
        total_cached = self.l1_current_bytes + self.l2_current_bytes + self.l3_current_bytes
        total_capacity = self.l1_size_bytes + self.l2_size_bytes + self.l3_size_bytes
        self.stats.cache_utilization = total_cached / total_capacity if total_capacity > 0 else 0
    
    async def _put_l1(self, key: str, entry: CacheEntry):
        """Put entry in L1 cache."""
        entry.level = CacheLevel.L1_MEMORY
        
        # Evict if necessary
        self.l1_current_bytes = await self._evict_from_cache(
            self.l1_cache, self.l1_lock, 
            self.l1_current_bytes, self.l1_size_bytes,
            entry.size_bytes
        )
        
        async with self.l1_lock:
            self.l1_cache[key] = entry
            self.l1_current_bytes += entry.size_bytes
    
    async def _put_l2(self, key: str, entry: CacheEntry):
        """Put entry in L2 cache."""
        entry.level = CacheLevel.L2_MEMORY
        
        # Evict if necessary
        self.l2_current_bytes = await self._evict_from_cache(
            self.l2_cache, self.l2_lock,
            self.l2_current_bytes, self.l2_size_bytes,
            entry.size_bytes
        )
        
        async with self.l2_lock:
            self.l2_cache[key] = entry
            self.l2_current_bytes += entry.size_bytes
    
    async def _promote_to_l1(self, key: str, entry: CacheEntry):
        """Promote entry from L2 to L1."""
        if entry.size_bytes > self.l1_size_bytes // 4:
            return  # Too large for L1
            
        # Remove from L2
        async with self.l2_lock:
            if key in self.l2_cache:
                del self.l2_cache[key]
                self.l2_current_bytes -= entry.size_bytes
        
        # Add to L1
        await self._put_l1(key, entry)
    
    async def _trigger_prefetch(self, accessed_key: str):
        """Trigger prefetching based on access patterns."""
        if not self.enable_prefetch:
            return
            
        predicted_keys = self.access_pattern.predict_next_accesses(accessed_key, limit=3)
        
        for pred_key in predicted_keys:
            if (pred_key not in self.l1_cache and 
                pred_key not in self.l2_cache and
                pred_key not in self.prefetch_active):
                
                # Queue for prefetching
                # Note: This is a simplified version. In practice, you'd need
                # to decode the key back to file_path, dataset_path, slice_spec
                # and have a callback to actually load the data
                await self.prefetch_queue.put((self._dummy_prefetch, pred_key))
    
    async def _dummy_prefetch(self, key: str) -> bool:
        """Dummy prefetch function - replace with actual data loading."""
        # In practice, this would decode the key and load the actual data
        # For now, just simulate some work
        await asyncio.sleep(0.1)
        return False  # Indicate we didn't actually cache anything
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total_entries = len(self.l1_cache) + len(self.l2_cache) + len(self.l3_cache)
        total_bytes = self.l1_current_bytes + self.l2_current_bytes + self.l3_current_bytes
        
        return {
            'cache_stats': {
                'hit_rate': self.stats.hit_rate,
                'total_hits': self.stats.hits,
                'total_misses': self.stats.misses,
                'evictions': self.stats.evictions,
                'prefetch_accuracy': self.stats.prefetch_accuracy,
                'prefetch_hits': self.stats.prefetch_hits,
                'prefetch_misses': self.stats.prefetch_misses,
                'avg_access_time_ms': self.stats.avg_access_time * 1000,
                'cache_utilization': self.stats.cache_utilization
            },
            'cache_levels': {
                'l1': {
                    'entries': len(self.l1_cache),
                    'size_bytes': self.l1_current_bytes,
                    'size_mb': self.l1_current_bytes / (1024 * 1024),
                    'max_size_mb': self.l1_size_bytes / (1024 * 1024),
                    'utilization': self.l1_current_bytes / self.l1_size_bytes if self.l1_size_bytes > 0 else 0
                },
                'l2': {
                    'entries': len(self.l2_cache),
                    'size_bytes': self.l2_current_bytes,
                    'size_mb': self.l2_current_bytes / (1024 * 1024),
                    'max_size_mb': self.l2_size_bytes / (1024 * 1024),
                    'utilization': self.l2_current_bytes / self.l2_size_bytes if self.l2_size_bytes > 0 else 0
                }
            },
            'access_patterns': {
                'total_accesses': len(self.access_pattern.access_history),
                'unique_keys': len(self.access_pattern.access_frequencies),
                'sequence_patterns': len(self.access_pattern.sequence_patterns),
                'dataset_groups': len(self.access_pattern.dataset_groups)
            },
            'system': {
                'total_entries': total_entries,
                'total_size_bytes': total_bytes,
                'total_size_mb': total_bytes / (1024 * 1024),
                'memory_usage_mb': psutil.Process().memory_info().rss / (1024 * 1024)
            }
        }
    
    async def clear(self):
        """Clear all caches."""
        async with self.l1_lock:
            self.l1_cache.clear()
            self.l1_current_bytes = 0
            
        async with self.l2_lock:
            self.l2_cache.clear()
            self.l2_current_bytes = 0
            
        async with self.l3_lock:
            self.l3_cache.clear()
            self.l3_current_bytes = 0
        
        # Reset stats
        self.stats = CacheStats()
        self.access_pattern = AccessPattern()
    
    async def close(self):
        """Close cache and cleanup resources."""
        # Cancel prefetch tasks
        for task in self.prefetch_tasks:
            task.cancel()
            
        # Wait for tasks to finish
        if self.prefetch_tasks:
            await asyncio.gather(*self.prefetch_tasks, return_exceptions=True)
        
        # Clear caches
        await self.clear()

# Global cache instance
_global_cache: Optional[MultiLevelCache] = None

def get_cache(l1_size_mb: int = 512, l2_size_mb: int = 2048, 
              enable_prefetch: bool = True) -> MultiLevelCache:
    """Get the global cache instance."""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = MultiLevelCache(
            l1_size_mb=l1_size_mb,
            l2_size_mb=l2_size_mb,
            enable_prefetch=enable_prefetch
        )
    
    return _global_cache

async def close_cache():
    """Close and cleanup the global cache."""
    global _global_cache
    
    if _global_cache:
        await _global_cache.close()
        _global_cache = None