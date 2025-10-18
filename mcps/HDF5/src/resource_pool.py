#!/usr/bin/env python3
# /// script
# dependencies = [
#   "h5py>=3.9.0",
#   "numpy>=1.24.0,<2.0.0",
#   "psutil>=5.9.0",
#   "aiofiles>=23.2.1"
# ]
# requires-python = ">=3.10"
# ///

"""
Connection pooling and resource management for HDF5 operations.
"""
import asyncio
import threading
import time
import weakref
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import h5py
import psutil
import logging
from contextlib import asynccontextmanager
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """Connection states in the pool."""
    AVAILABLE = "available"
    IN_USE = "in_use"
    CLOSED = "closed"
    ERROR = "error"

@dataclass
class ConnectionInfo:
    """Information about a pooled connection."""
    file_path: Path
    connection_id: str
    h5_file: h5py.File
    state: ConnectionState = ConnectionState.AVAILABLE
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    error_count: int = 0
    thread_id: Optional[int] = None
    
    def mark_used(self):
        """Mark connection as used."""
        self.last_used = time.time()
        self.use_count += 1
        self.state = ConnectionState.IN_USE
        self.thread_id = threading.get_ident()
    
    def mark_available(self):
        """Mark connection as available."""
        self.state = ConnectionState.AVAILABLE
        self.thread_id = None
    
    def mark_error(self):
        """Mark connection as having an error."""
        self.error_count += 1
        self.state = ConnectionState.ERROR
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at
    
    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used
    
    @property
    def is_healthy(self) -> bool:
        return self.state in (ConnectionState.AVAILABLE, ConnectionState.IN_USE) and self.error_count < 3

class HDF5ConnectionPool:
    """Thread-safe connection pool for HDF5 files."""
    
    def __init__(self, 
                 max_connections_per_file: int = 5,
                 max_total_connections: int = 50,
                 connection_timeout: float = 300.0,  # 5 minutes
                 cleanup_interval: float = 60.0,     # 1 minute
                 enable_swmr: bool = True):
        
        self.max_connections_per_file = max_connections_per_file
        self.max_total_connections = max_total_connections
        self.connection_timeout = connection_timeout
        self.cleanup_interval = cleanup_interval
        self.enable_swmr = enable_swmr
        
        # Connection storage
        self._connections: Dict[Path, List[ConnectionInfo]] = defaultdict(list)
        self._connection_semaphores: Dict[Path, asyncio.Semaphore] = {}
        self._total_connections = 0
        
        # Thread safety
        self._pool_lock = asyncio.Lock()
        self._file_locks: Dict[Path, threading.RLock] = {}
        
        # Statistics
        self._stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'connections_closed': 0,
            'connection_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Start cleanup
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start the background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background task to clean up expired connections."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection cleanup error: {e}")
    
    def _get_file_lock(self, file_path: Path) -> threading.RLock:
        """Get or create a lock for the given file."""
        if file_path not in self._file_locks:
            self._file_locks[file_path] = threading.RLock()
        return self._file_locks[file_path]
    
    async def _create_connection(self, file_path: Path) -> ConnectionInfo:
        """Create a new HDF5 connection."""
        connection_id = f"{file_path}_{int(time.time() * 1000000)}"
        
        def open_file():
            """Open HDF5 file in a thread-safe manner."""
            try:
                # Open with SWMR for concurrent access if enabled
                h5_file = h5py.File(
                    file_path, 
                    'r',
                    swmr=self.enable_swmr,
                    libver='latest' if self.enable_swmr else None
                )
                return h5_file
            except Exception as e:
                logger.error(f"Failed to open HDF5 file {file_path}: {e}")
                raise
        
        loop = asyncio.get_event_loop()
        h5_file = await loop.run_in_executor(None, open_file)
        
        connection = ConnectionInfo(
            file_path=file_path,
            connection_id=connection_id,
            h5_file=h5_file
        )
        
        self._stats['connections_created'] += 1
        logger.debug(f"Created new connection {connection_id} for {file_path}")
        
        return connection
    
    async def _get_available_connection(self, file_path: Path) -> Optional[ConnectionInfo]:
        """Get an available connection for the file."""
        connections = self._connections[file_path]
        
        for conn in connections:
            if conn.state == ConnectionState.AVAILABLE and conn.is_healthy:
                return conn
        
        return None
    
    async def _cleanup_connection(self, connection: ConnectionInfo):
        """Cleanup a single connection."""
        try:
            if connection.h5_file and connection.h5_file.id.valid:
                connection.h5_file.close()
            connection.state = ConnectionState.CLOSED
            self._stats['connections_closed'] += 1
            logger.debug(f"Cleaned up connection {connection.connection_id}")
        except Exception as e:
            logger.error(f"Error closing connection {connection.connection_id}: {e}")
    
    async def _cleanup_expired_connections(self):
        """Clean up expired and unhealthy connections."""
        async with self._pool_lock:
            current_time = time.time()
            cleanup_tasks = []
            
            for file_path, connections in list(self._connections.items()):
                healthy_connections = []
                
                for conn in connections:
                    # Check if connection should be cleaned up
                    should_cleanup = (
                        not conn.is_healthy or
                        conn.idle_seconds > self.connection_timeout or
                        conn.state == ConnectionState.ERROR
                    )
                    
                    if should_cleanup and conn.state != ConnectionState.IN_USE:
                        cleanup_tasks.append(self._cleanup_connection(conn))
                        self._total_connections -= 1
                    else:
                        healthy_connections.append(conn)
                
                if healthy_connections:
                    self._connections[file_path] = healthy_connections
                else:
                    # No healthy connections left
                    del self._connections[file_path]
                    if file_path in self._connection_semaphores:
                        del self._connection_semaphores[file_path]
            
            # Execute cleanup tasks
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                logger.debug(f"Cleaned up {len(cleanup_tasks)} expired connections")
    
    @asynccontextmanager
    async def get_connection(self, file_path: Path):
        """Get a connection from the pool (context manager)."""
        if not file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        
        # Get or create semaphore for this file
        if file_path not in self._connection_semaphores:
            async with self._pool_lock:
                if file_path not in self._connection_semaphores:
                    self._connection_semaphores[file_path] = asyncio.Semaphore(
                        self.max_connections_per_file
                    )
        
        semaphore = self._connection_semaphores[file_path]
        
        # Acquire semaphore (limit concurrent connections per file)
        async with semaphore:
            connection = None
            
            try:
                async with self._pool_lock:
                    # Try to get an available connection
                    connection = await self._get_available_connection(file_path)
                    
                    if connection:
                        # Reuse existing connection
                        connection.mark_used()
                        self._stats['connections_reused'] += 1
                        self._stats['cache_hits'] += 1
                    else:
                        # Create new connection if under limits
                        if (len(self._connections[file_path]) < self.max_connections_per_file and
                            self._total_connections < self.max_total_connections):
                            
                            connection = await self._create_connection(file_path)
                            connection.mark_used()
                            self._connections[file_path].append(connection)
                            self._total_connections += 1
                            self._stats['cache_misses'] += 1
                        else:
                            # Wait for an available connection
                            self._stats['cache_misses'] += 1
                            # This is a simplification - in practice you'd want a more sophisticated queuing system
                            raise RuntimeError(f"Connection pool exhausted for {file_path}")
                
                if connection is None:
                    raise RuntimeError(f"Unable to acquire connection for {file_path}")
                
                # Yield the connection
                yield connection
                
            except Exception as e:
                if connection:
                    connection.mark_error()
                    self._stats['connection_errors'] += 1
                logger.error(f"Connection error for {file_path}: {e}")
                raise
                
            finally:
                # Return connection to pool
                if connection and connection.state != ConnectionState.ERROR:
                    connection.mark_available()
    
    async def get_file_handle(self, file_path: Path) -> h5py.File:
        """Get a raw file handle (for backwards compatibility)."""
        async with self.get_connection(file_path) as conn:
            return conn.h5_file
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        file_stats = {}
        total_available = 0
        total_in_use = 0
        total_error = 0
        
        for file_path, connections in self._connections.items():
            available = sum(1 for c in connections if c.state == ConnectionState.AVAILABLE)
            in_use = sum(1 for c in connections if c.state == ConnectionState.IN_USE)
            error = sum(1 for c in connections if c.state == ConnectionState.ERROR)
            
            total_available += available
            total_in_use += in_use
            total_error += error
            
            file_stats[str(file_path)] = {
                'total_connections': len(connections),
                'available': available,
                'in_use': in_use,
                'error': error,
                'avg_age_seconds': sum(c.age_seconds for c in connections) / len(connections) if connections else 0,
                'avg_idle_seconds': sum(c.idle_seconds for c in connections if c.state == ConnectionState.AVAILABLE) / available if available else 0
            }
        
        return {
            'pool_config': {
                'max_connections_per_file': self.max_connections_per_file,
                'max_total_connections': self.max_total_connections,
                'connection_timeout': self.connection_timeout,
                'cleanup_interval': self.cleanup_interval
            },
            'pool_stats': {
                'total_files': len(self._connections),
                'total_connections': self._total_connections,
                'total_available': total_available,
                'total_in_use': total_in_use,
                'total_error': total_error,
                'utilization': self._total_connections / self.max_total_connections if self.max_total_connections > 0 else 0
            },
            'performance_stats': self._stats.copy(),
            'file_details': file_stats,
            'system_info': {
                'memory_usage_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                'open_files': len(psutil.Process().open_files())
            }
        }
    
    async def close_all_connections(self):
        """Close all connections in the pool."""
        self._shutdown = True
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._pool_lock:
            cleanup_tasks = []
            
            for connections in self._connections.values():
                for conn in connections:
                    if conn.state != ConnectionState.CLOSED:
                        cleanup_tasks.append(self._cleanup_connection(conn))
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            self._connections.clear()
            self._connection_semaphores.clear()
            self._total_connections = 0
        
        logger.info("Closed all connections in the pool")

class ResourceManager:
    """High-level resource manager with automatic cleanup and monitoring."""
    
    def __init__(self, 
                 connection_pool: Optional[HDF5ConnectionPool] = None,
                 memory_limit_mb: Optional[int] = None,
                 file_handle_limit: Optional[int] = None):
        
        self.connection_pool = connection_pool or HDF5ConnectionPool()
        
        # System limits
        system_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
        self.memory_limit_mb = memory_limit_mb or int(system_memory * 0.5)  # 50% of system memory
        
        # File handle limits
        try:
            import resource
            max_files = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            self.file_handle_limit = file_handle_limit or max(100, min(1000, max_files // 4))
        except:
            self.file_handle_limit = file_handle_limit or 500
        
        # Resource tracking
        self._active_resources: Dict[str, Any] = {}
        self._resource_lock = asyncio.Lock()
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_monitoring()
        
        # Cleanup callbacks
        self._cleanup_callbacks: List[Callable[[], None]] = []
        
        # Register cleanup on garbage collection
        weakref.finalize(self, self._final_cleanup)
    
    def _start_monitoring(self):
        """Start resource monitoring task."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_resources())
    
    async def _monitor_resources(self):
        """Monitor system resources and trigger cleanup if needed."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check memory usage
                memory_info = psutil.virtual_memory()
                current_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                
                if current_memory_mb > self.memory_limit_mb:
                    logger.warning(f"Memory usage ({current_memory_mb:.1f} MB) exceeds limit ({self.memory_limit_mb} MB)")
                    await self._emergency_cleanup()
                
                # Check file handles
                try:
                    open_files = len(psutil.Process().open_files())
                    if open_files > self.file_handle_limit:
                        logger.warning(f"Open files ({open_files}) exceeds limit ({self.file_handle_limit})")
                        await self._emergency_cleanup()
                except:
                    pass  # Some systems don't support open_files()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    async def _emergency_cleanup(self):
        """Emergency cleanup when resource limits are exceeded."""
        logger.info("Performing emergency resource cleanup")
        
        # Clean up connection pool
        await self.connection_pool._cleanup_expired_connections()
        
        # Run custom cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Cleanup callback error: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback to be called during resource pressure."""
        self._cleanup_callbacks.append(callback)
    
    @asynccontextmanager
    async def get_hdf5_connection(self, file_path: Path):
        """Get an HDF5 connection with resource tracking."""
        async with self.connection_pool.get_connection(file_path) as conn:
            resource_id = f"hdf5_{conn.connection_id}"
            
            async with self._resource_lock:
                self._active_resources[resource_id] = {
                    'type': 'hdf5_connection',
                    'file_path': str(file_path),
                    'connection_id': conn.connection_id,
                    'acquired_at': time.time()
                }
            
            try:
                yield conn.h5_file
            finally:
                async with self._resource_lock:
                    self._active_resources.pop(resource_id, None)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics."""
        pool_stats = self.connection_pool.get_pool_stats()
        
        # System resource information
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        system_stats = {
            'memory': {
                'total_mb': memory.total / (1024 * 1024),
                'available_mb': memory.available / (1024 * 1024),
                'used_percent': memory.percent,
                'process_usage_mb': process.memory_info().rss / (1024 * 1024),
                'limit_mb': self.memory_limit_mb
            },
            'files': {
                'limit': self.file_handle_limit
            },
            'cpu': {
                'count': psutil.cpu_count(),
                'percent': psutil.cpu_percent(interval=None)
            }
        }
        
        try:
            system_stats['files']['open_count'] = len(process.open_files())
        except:
            system_stats['files']['open_count'] = 'unavailable'
        
        return {
            'connection_pool': pool_stats,
            'system_resources': system_stats,
            'active_resources': {
                'count': len(self._active_resources),
                'resources': list(self._active_resources.values())
            },
            'limits': {
                'memory_limit_mb': self.memory_limit_mb,
                'file_handle_limit': self.file_handle_limit
            }
        }
    
    def _final_cleanup(self):
        """Final cleanup called by weakref when object is garbage collected."""
        # This runs in a different thread, so we can't use async
        try:
            if hasattr(self, 'connection_pool'):
                # Schedule cleanup on the event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self.connection_pool.close_all_connections())
                except:
                    pass  # Event loop might not be available
        except Exception as e:
            logger.error(f"Final cleanup error: {e}")
    
    async def close(self):
        """Close the resource manager and cleanup all resources."""
        # Cancel monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Close connection pool
        await self.connection_pool.close_all_connections()
        
        # Clear active resources
        async with self._resource_lock:
            self._active_resources.clear()
        
        logger.info("Resource manager closed")

# Global instances
_global_connection_pool: Optional[HDF5ConnectionPool] = None
_global_resource_manager: Optional[ResourceManager] = None

def get_connection_pool(**kwargs) -> HDF5ConnectionPool:
    """Get the global connection pool instance."""
    global _global_connection_pool
    
    if _global_connection_pool is None:
        _global_connection_pool = HDF5ConnectionPool(**kwargs)
    
    return _global_connection_pool

def get_resource_manager(**kwargs) -> ResourceManager:
    """Get the global resource manager instance."""
    global _global_resource_manager
    
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager(**kwargs)
    
    return _global_resource_manager

async def close_global_resources():
    """Close all global resources."""
    global _global_connection_pool, _global_resource_manager
    
    if _global_resource_manager:
        await _global_resource_manager.close()
        _global_resource_manager = None
    
    if _global_connection_pool:
        await _global_connection_pool.close_all_connections()
        _global_connection_pool = None