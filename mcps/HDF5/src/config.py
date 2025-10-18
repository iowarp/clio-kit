"""
Configuration handling for the HDF5 MCP server.
Implements a hierarchical configuration system with validation and runtime updates.
"""
#!/usr/bin/env python3
# /// script
# dependencies = [
#   "mcp>=1.4.0",
#   "h5py>=3.8.0,<3.9.0",
#   "numpy>=1.24.0,<2.0.0",
#   "pydantic>=2.0.0",
#   "python-dotenv>=0.19.0"
# ]
# requires-python = ">=3.10"
# ///

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv
import logging
from pydantic import BaseModel, field_validator
from functools import lru_cache

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

class ServerConfig(BaseModel):
    """Server-specific configuration settings."""
    name: str = "HDF5 MCP Server"
    version: str = "0.1.0"
    request_timeout: float = 30.0

class AsyncConfig(BaseModel):
    """Asynchronous processing configuration."""
    max_workers: int = 4
    task_queue_size: int = 1000
    batch_size: int = 100
    max_batch_wait_time: float = 0.5

class TransportConfig(BaseModel):
    """Transport configuration."""
    enable_stdio: bool = True
    enable_sse: bool = False
    sse_host: str = "localhost"
    sse_port: int = 8765
    max_connections: int = 100
    enable_batching: bool = True
    batch_timeout: float = 0.1
    max_batch_size: int = 50

class HDF5Config(BaseModel):
    """HDF5-specific configuration."""
    data_dir: Path = Path("data")
    file_path: Optional[Path] = None
    chunk_size: int = 1024 * 1024 * 64  # 64MB chunks for better performance
    cache_size: int = 1024 * 1024 * 500  # 500MB default
    prefetch_enabled: bool = True
    parallel_threshold: int = 1024 * 1024 * 100  # 100MB
    
    @field_validator('data_dir', 'file_path')
    @classmethod
    def validate_paths(cls, v):
        if v is not None:
            return Path(v).resolve()
        return v

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(levelname)s:%(name)s:%(message)s"
    file: Optional[str] = None

class Config:
    """Full application configuration."""
    
    def __init__(self):
        self.server = ServerConfig()
        self.async_config = AsyncConfig()
        self.transport = TransportConfig()
        self.hdf5 = HDF5Config()
        self.logging = LoggingConfig()
        self._load_config()
        
    def _load_config(self) -> None:
        """Load configuration from multiple sources."""
        # Order of precedence (highest to lowest):
        # 1. Environment variables
        # 2. Config file (if exists)
        # 3. Default values (from models)
        
        # First load from config file if it exists
        config_file = Path(os.environ.get('HDF5_MCP_CONFIG', 'config.json'))
        if config_file.exists():
            with open(config_file) as f:
                config_dict = json.load(f)
                self._update_from_dict(config_dict)
        
        # Then override with environment variables
        self._load_from_env()
        
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # HDF5 config
        if 'HDF5_MCP_DATA_DIR' in os.environ:
            self.hdf5.data_dir = Path(os.environ['HDF5_MCP_DATA_DIR']).resolve()
        
        if 'HDF5_MCP_STORAGE_ENDPOINT' in os.environ:
            storage_endpoint = os.environ['HDF5_MCP_STORAGE_ENDPOINT']
            # Handle file:// protocol prefix
            if storage_endpoint.startswith('file://'):
                storage_endpoint = storage_endpoint[7:]
            self.hdf5.data_dir = Path(storage_endpoint).resolve()
            
        if 'HDF5_MCP_CACHE_SIZE_MB' in os.environ:
            try:
                # Value in MB
                cache_size_mb = int(os.environ['HDF5_MCP_CACHE_SIZE_MB'])
                self.hdf5.cache_size = cache_size_mb * 1024 * 1024
            except ValueError:
                logger.warning(f"Invalid cache size: {os.environ['HDF5_MCP_CACHE_SIZE_MB']}")
        
        if 'HDF5_MCP_CHUNK_SIZE_MB' in os.environ:
            try:
                chunk_size_mb = int(os.environ['HDF5_MCP_CHUNK_SIZE_MB'])
                self.hdf5.chunk_size = chunk_size_mb * 1024 * 1024
            except ValueError:
                logger.warning(f"Invalid chunk size: {os.environ['HDF5_MCP_CHUNK_SIZE_MB']}")
        
        if 'HDF5_MCP_PARALLEL_THRESHOLD_MB' in os.environ:
            try:
                threshold_mb = int(os.environ['HDF5_MCP_PARALLEL_THRESHOLD_MB'])
                self.hdf5.parallel_threshold = threshold_mb * 1024 * 1024
            except ValueError:
                logger.warning(f"Invalid parallel threshold: {os.environ['HDF5_MCP_PARALLEL_THRESHOLD_MB']}")
        
        if 'HDF5_MCP_PREFETCH_ENABLED' in os.environ:
            self.hdf5.prefetch_enabled = os.environ['HDF5_MCP_PREFETCH_ENABLED'].lower() in ('true', '1', 'yes')
        
        # Transport config
        if 'HDF5_MCP_ENABLE_SSE' in os.environ:
            self.transport.enable_sse = os.environ['HDF5_MCP_ENABLE_SSE'].lower() in ('true', '1', 'yes')
        
        if 'HDF5_MCP_PORT' in os.environ:
            try:
                self.transport.sse_port = int(os.environ['HDF5_MCP_PORT'])
            except ValueError:
                logger.warning(f"Invalid port: {os.environ['HDF5_MCP_PORT']}")
        
        if 'HDF5_MCP_HOST' in os.environ:
            self.transport.sse_host = os.environ['HDF5_MCP_HOST']
        
        if 'HDF5_MCP_MAX_CONNECTIONS' in os.environ:
            try:
                self.transport.max_connections = int(os.environ['HDF5_MCP_MAX_CONNECTIONS'])
            except ValueError:
                logger.warning(f"Invalid max connections: {os.environ['HDF5_MCP_MAX_CONNECTIONS']}")
        
        # Async config
        if 'HDF5_MCP_MAX_WORKERS' in os.environ:
            try:
                self.async_config.max_workers = int(os.environ['HDF5_MCP_MAX_WORKERS'])
            except ValueError:
                logger.warning(f"Invalid max workers: {os.environ['HDF5_MCP_MAX_WORKERS']}")
        
        # Logging config
        if 'HDF5_MCP_LOG_LEVEL' in os.environ:
            self.logging.level = os.environ['HDF5_MCP_LOG_LEVEL']
            
    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from a dictionary."""
        if 'server' in config_dict:
            self.server = ServerConfig(**config_dict['server'])
        if 'async_config' in config_dict:
            self.async_config = AsyncConfig(**config_dict['async_config'])
        if 'transport' in config_dict:
            self.transport = TransportConfig(**config_dict['transport'])
        if 'hdf5' in config_dict:
            self.hdf5 = HDF5Config(**config_dict['hdf5'])
        if 'logging' in config_dict:
            self.logging = LoggingConfig(**config_dict['logging'])
            
    def update_runtime(self, section: str, key: str, value: Any) -> None:
        """
        Update a configuration value at runtime.
        
        Args:
            section: Configuration section (e.g., 'server', 'hdf5')
            key: Configuration key within the section
            value: New value to set
        """
        if not hasattr(self, section):
            raise ValueError(f"Invalid configuration section: {section}")
            
        section_config = getattr(self, section)
        if not hasattr(section_config, key):
            raise ValueError(f"Invalid configuration key: {key} in section {section}")
            
        # Convert value to correct type if needed
        field_type = type(getattr(section_config, key))
        if not isinstance(value, field_type):
            value = field_type(value)
            
        # Update the value
        setattr(section_config, key, value)
        logger.debug(f"Updated runtime config: {section}.{key} = {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            'server': self.server.dict(),
            'async_config': self.async_config.dict(),
            'transport': self.transport.dict(),
            'hdf5': self.hdf5.dict(),
            'logging': self.logging.dict()
        }

# Global config instance
_CONFIG = None

# No longer using lru_cache for get_config
def get_config() -> Config:
    """Get the application configuration singleton."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG

def _configure_logging(level_name: str) -> None:
    """Configure logging with the specified level."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    level = level_map.get(level_name.upper(), DEFAULT_LOG_LEVEL)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=get_config().logging.format
    )
    
    # If a log file is specified, add a file handler
    log_file = get_config().logging.file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter(get_config().logging.format)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    
    logger.debug(f"Logging configured with level {level_name}")

# Initialize logging when module is loaded
_configure_logging(get_config().logging.level)

# =========================================================================
# Public API functions
# =========================================================================

def get_storage_path() -> Path:
    """Get the current storage path for HDF5 files."""
    return get_config().hdf5.data_dir

def set_storage_path(path: Union[str, Path]) -> None:
    """
    Set the storage path for HDF5 files.
    
    Args:
        path: Directory path for HDF5 files
    """
    path_obj = Path(path) if isinstance(path, str) else path
    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created storage directory: {path_obj}")
    
    get_config().update_runtime('hdf5', 'data_dir', path_obj.resolve())
    logger.info(f"Storage path set to: {path_obj.resolve()}")

def get_cache_size() -> int:
    """
    Get the current cache size in MB.
    
    Returns:
        Cache size in megabytes
    """
    return get_config().hdf5.cache_size // (1024 * 1024)  # Convert bytes to MB

def set_cache_size(size_mb: int) -> None:
    """
    Set the cache size for HDF5 operations.
    
    Args:
        size_mb: Cache size in megabytes
    """
    if size_mb <= 0:
        raise ValueError("Cache size must be positive")
    
    # Convert MB to bytes
    size_bytes = size_mb * 1024 * 1024
    
    # Get config and update cache size
    config = get_config()
    config.update_runtime('hdf5', 'cache_size', size_bytes)
    logger.info(f"Cache size set to {size_mb} MB ({size_bytes} bytes)")

def get_log_level() -> str:
    """
    Get the current log level.
    
    Returns:
        Current log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    return get_config().logging.level

def set_log_level(level: str) -> None:
    """
    Set the log level for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    level_upper = level.upper()
    
    if level_upper not in valid_levels:
        valid_levels_str = ", ".join(valid_levels)
        raise ValueError(f"Invalid log level: {level}. Must be one of: {valid_levels_str}")
    
    get_config().update_runtime('logging', 'level', level_upper)
    # Reconfigure logging with the new level
    _configure_logging(level_upper)
    logger.info(f"Log level set to {level_upper}")
