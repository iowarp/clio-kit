"""
HDF5 File Scanner Module

This module provides functionality for discovering and validating HDF5 files
in a directory structure.
"""

import os
import h5py
from pathlib import Path
from typing import Generator, Optional, Dict, Any, List, Set
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue
import time
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class HDF5Metadata:
    """Metadata for an HDF5 file."""
    path: Path
    size: int
    last_modified: float
    groups: Set[str]
    datasets: Dict[str, Dict[str, Any]]
    attributes: Dict[str, Any]

class HDF5Scanner:
    """Scanner class for discovering and validating HDF5 files."""
    
    def __init__(self, base_dir: Optional[str] = None, max_workers: int = 4,
                chunk_size: int = 1000, metadata_cache_size: int = 1000):
        """
        Initialize the scanner with a base directory.
        
        Args:
            base_dir: Base directory to scan. Defaults to configured data directory.
            max_workers: Maximum number of parallel workers for scanning
            chunk_size: Number of files to process in each chunk
            metadata_cache_size: Size of the metadata LRU cache
        """
        self.base_dir = Path(base_dir or "/home/akougkas/dev/hdf5-mcp-server/data")
        self.scan_lock = Lock()
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self._metadata_cache = {}
        self._skip_patterns = {'.git', '__pycache__', 'node_modules'}
        
    @lru_cache(maxsize=1000)
    def is_hdf5_file(self, file_path: Path) -> bool:
        """
        Check if a file is a valid HDF5 file with caching.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if file is a valid HDF5 file, False otherwise
        """
        try:
            if not file_path.is_file():
                return False
                
            # Quick signature check first
            with open(file_path, 'rb') as f:
                signature = f.read(8)
                if signature != b'\x89HDF\r\n\x1a\n':
                    return False
            
            # Try opening the file as HDF5
            with h5py.File(file_path, 'r') as _:
                return True
        except (OSError, IOError):
            return False
            
    def _should_skip_dir(self, dir_path: str) -> bool:
        """Check if directory should be skipped."""
        return any(pattern in dir_path for pattern in self._skip_patterns)
        
    def _extract_minimal_metadata(self, file_path: Path) -> Optional[HDF5Metadata]:
        """
        Extract minimal metadata from an HDF5 file.
        
        Args:
            file_path: Path to the HDF5 file
            
        Returns:
            HDF5Metadata object or None if extraction fails
        """
        try:
            stat = file_path.stat()
            with h5py.File(file_path, 'r') as f:
                groups = set()
                datasets = {}
                
                def visitor(name: str, obj: Any) -> None:
                    if isinstance(obj, h5py.Group):
                        groups.add(name)
                    elif isinstance(obj, h5py.Dataset):
                        datasets[name] = {
                            'shape': obj.shape,
                            'dtype': str(obj.dtype),
                            'chunks': obj.chunks
                        }
                
                # Visit all objects but limit depth
                f.visititems(visitor)
                
                return HDF5Metadata(
                    path=file_path,
                    size=stat.st_size,
                    last_modified=stat.st_mtime,
                    groups=groups,
                    datasets=datasets,
                    attributes=dict(f.attrs)
                )
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
            return None
            
    def scan_directory(self) -> Generator[HDF5Metadata, None, None]:
        """
        Recursively scan directory for HDF5 files using a generator.
        
        Yields:
            HDF5Metadata objects for each discovered HDF5 file
        """
        with self.scan_lock:
            try:
                for root, dirs, files in os.walk(self.base_dir):
                    # Skip unwanted directories
                    dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]
                    
                    for file in files:
                        file_path = Path(root) / file
                        
                        # Check cache first
                        if file_path in self._metadata_cache:
                            yield self._metadata_cache[file_path]
                            continue
                            
                        if self.is_hdf5_file(file_path):
                            metadata = self._extract_minimal_metadata(file_path)
                            if metadata:
                                self._metadata_cache[file_path] = metadata
                                yield metadata
                                
            except Exception as e:
                logger.error(f"Error scanning directory: {e}")
                raise
                
    def parallel_scan(self) -> List[HDF5Metadata]:
        """
        Perform optimized parallel scanning of directories for HDF5 files.
        
        Returns:
            List of HDF5Metadata objects for discovered files
        """
        discovered_files: List[HDF5Metadata] = []
        path_queue = queue.Queue()
        
        def scan_chunk() -> List[HDF5Metadata]:
            """Process a chunk of files."""
            results = []
            while True:
                try:
                    path = path_queue.get_nowait()
                    if self.is_hdf5_file(path):
                        metadata = self._extract_minimal_metadata(path)
                        if metadata:
                            results.append(metadata)
                except queue.Empty:
                    break
            return results
            
        try:
            # Collect all file paths first
            all_paths = []
            for root, dirs, files in os.walk(self.base_dir):
                dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]
                all_paths.extend(Path(root) / f for f in files)
                
            # Process files in chunks
            for i in range(0, len(all_paths), self.chunk_size):
                chunk = all_paths[i:i + self.chunk_size]
                
                # Fill the queue with chunk paths
                for path in chunk:
                    path_queue.put(path)
                    
                # Process chunk in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(scan_chunk) 
                             for _ in range(self.max_workers)]
                    
                    for future in as_completed(futures):
                        discovered_files.extend(future.result())
                        
        except Exception as e:
            logger.error(f"Error in parallel scan: {e}")
            raise
            
        return discovered_files
        
    def get_metadata(self, file_path: Path) -> Optional[HDF5Metadata]:
        """
        Get metadata for a specific HDF5 file.
        
        Args:
            file_path: Path to the HDF5 file
            
        Returns:
            HDF5Metadata object or None if not found/invalid
        """
        if file_path in self._metadata_cache:
            return self._metadata_cache[file_path]
            
        if self.is_hdf5_file(file_path):
            metadata = self._extract_minimal_metadata(file_path)
            if metadata:
                self._metadata_cache[file_path] = metadata
                return metadata
        return None 