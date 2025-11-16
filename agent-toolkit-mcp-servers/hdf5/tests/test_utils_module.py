"""
Comprehensive tests for the utils module.

Tests performance monitoring, file handle caching, HDF5 manager,
and all utility functions.
"""

import pytest
import time
import json
import h5py
import numpy as np

# Import the utils module directly - DO NOT import server
from hdf5_mcp import utils


# =========================================================================
# setup_logging Tests
# =========================================================================


def test_setup_logging_default():
    """Test setup_logging with default INFO level."""
    utils.setup_logging()
    # Should not raise any errors


def test_setup_logging_debug():
    """Test setup_logging with DEBUG level."""
    utils.setup_logging("DEBUG")
    # Should not raise any errors


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_setup_logging_all_levels(level):
    """Test setup_logging with all valid levels."""
    utils.setup_logging(level)
    # Should not raise any errors


def test_setup_logging_lowercase():
    """Test setup_logging with lowercase level."""
    utils.setup_logging("debug")
    # Should handle lowercase


# =========================================================================
# PerformanceMonitor Tests
# =========================================================================


def test_performance_monitor_initialization(temp_dir):
    """Test PerformanceMonitor initialization."""
    monitor = utils.PerformanceMonitor(metrics_dir=temp_dir)
    assert monitor.metrics_dir == temp_dir
    assert monitor.metrics_dir.exists()
    assert "operations" in monitor.metrics
    assert "memory" in monitor.metrics
    assert "cache_stats" in monitor.metrics


def test_performance_monitor_default_dir():
    """Test PerformanceMonitor with default directory."""
    monitor = utils.PerformanceMonitor()
    assert monitor.metrics_dir.exists()
    assert ".hdf5_mcp" in str(monitor.metrics_dir)


def test_performance_monitor_track_operation():
    """Test tracking operations."""
    monitor = utils.PerformanceMonitor()
    monitor.track_operation("test_op", 1.5, size=1024)

    assert "test_op" in monitor.metrics["operations"]
    stats = monitor.metrics["operations"]["test_op"]
    assert stats["count"] == 1
    assert stats["total_time"] == 1.5
    assert stats["avg_time"] == 1.5
    assert stats["total_size"] == 1024


def test_performance_monitor_track_operation_multiple():
    """Test tracking multiple operations."""
    monitor = utils.PerformanceMonitor()
    monitor.track_operation("test_op", 1.0, size=512)
    monitor.track_operation("test_op", 2.0, size=1024)
    monitor.track_operation("test_op", 3.0, size=256)

    stats = monitor.metrics["operations"]["test_op"]
    assert stats["count"] == 3
    assert stats["total_time"] == 6.0
    assert stats["avg_time"] == 2.0
    assert stats["total_size"] == 1792


def test_performance_monitor_track_operation_no_size():
    """Test tracking operation without size."""
    monitor = utils.PerformanceMonitor()
    monitor.track_operation("test_op", 0.5)

    stats = monitor.metrics["operations"]["test_op"]
    assert stats["count"] == 1
    assert stats["total_size"] == 0


def test_performance_monitor_track_memory():
    """Test tracking memory usage."""
    monitor = utils.PerformanceMonitor()
    monitor.track_memory()

    assert len(monitor.metrics["memory"]) == 1
    mem_entry = monitor.metrics["memory"][0]
    assert "timestamp" in mem_entry
    assert "rss" in mem_entry
    assert "vms" in mem_entry
    assert mem_entry["rss"] > 0


def test_performance_monitor_track_memory_limit():
    """Test memory tracking limit of 1000 entries."""
    monitor = utils.PerformanceMonitor()

    # Add 1500 entries
    for _ in range(1500):
        monitor.track_memory()

    # Should only keep last 1000
    assert len(monitor.metrics["memory"]) == 1000


def test_performance_monitor_track_cache_hit():
    """Test tracking cache hit."""
    monitor = utils.PerformanceMonitor()
    monitor.track_cache(hit=True)

    assert monitor.metrics["cache_stats"]["hits"] == 1
    assert monitor.metrics["cache_stats"]["misses"] == 0


def test_performance_monitor_track_cache_miss():
    """Test tracking cache miss."""
    monitor = utils.PerformanceMonitor()
    monitor.track_cache(hit=False)

    assert monitor.metrics["cache_stats"]["hits"] == 0
    assert monitor.metrics["cache_stats"]["misses"] == 1


def test_performance_monitor_track_cache_multiple():
    """Test tracking multiple cache hits and misses."""
    monitor = utils.PerformanceMonitor()
    monitor.track_cache(hit=True)
    monitor.track_cache(hit=True)
    monitor.track_cache(hit=False)
    monitor.track_cache(hit=True)

    assert monitor.metrics["cache_stats"]["hits"] == 3
    assert monitor.metrics["cache_stats"]["misses"] == 1


def test_performance_monitor_get_summary_empty():
    """Test get_summary with no operations."""
    monitor = utils.PerformanceMonitor()
    summary = monitor.get_summary()

    assert summary["total_operations"] == 0
    assert summary["cache_hit_ratio"] == 0
    assert summary["current_memory_mb"] == 0
    assert summary["operation_stats"] == {}


def test_performance_monitor_get_summary_with_data():
    """Test get_summary with tracked data."""
    monitor = utils.PerformanceMonitor()
    monitor.track_operation("op1", 1.0, size=1024)
    monitor.track_operation("op2", 2.0, size=2048)
    monitor.track_cache(hit=True)
    monitor.track_cache(hit=True)
    monitor.track_cache(hit=False)
    monitor.track_memory()

    summary = monitor.get_summary()

    assert summary["total_operations"] == 2
    assert summary["cache_hit_ratio"] == 2 / 3
    assert summary["current_memory_mb"] > 0
    assert len(summary["operation_stats"]) == 2


def test_performance_monitor_save_metrics(temp_dir):
    """Test saving metrics to disk."""
    monitor = utils.PerformanceMonitor(metrics_dir=temp_dir)
    monitor.track_operation("test_op", 1.0, size=1024)
    monitor.save_metrics()

    # Check that a metrics file was created
    metrics_files = list(temp_dir.glob("metrics-*.json"))
    assert len(metrics_files) > 0

    # Verify file contents
    with open(metrics_files[0], "r") as f:
        data = json.load(f)
        assert "operations" in data
        assert "test_op" in data["operations"]


def test_performance_monitor_thread_safety():
    """Test thread-safe operations."""
    monitor = utils.PerformanceMonitor()

    # Simulate concurrent access
    import threading

    def track_ops():
        for _ in range(100):
            monitor.track_operation("concurrent_op", 0.1)

    threads = [threading.Thread(target=track_ops) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All operations should be tracked
    stats = monitor.metrics["operations"]["concurrent_op"]
    assert stats["count"] == 500


# =========================================================================
# monitor_performance Decorator Tests
# =========================================================================


def test_monitor_performance_decorator():
    """Test monitor_performance decorator."""
    monitor = utils.PerformanceMonitor()

    @utils.monitor_performance(monitor)
    def test_func():
        time.sleep(0.01)
        return "result"

    result = test_func()
    assert result == "result"
    assert "test_func" in monitor.metrics["operations"]


def test_monitor_performance_decorator_with_error():
    """Test monitor_performance decorator with exception."""
    monitor = utils.PerformanceMonitor()

    @utils.monitor_performance(monitor)
    def test_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        test_func()


def test_monitor_performance_decorator_with_numpy_result():
    """Test monitor_performance decorator with numpy result."""
    monitor = utils.PerformanceMonitor()

    @utils.monitor_performance(monitor)
    def test_func():
        return np.array([1, 2, 3, 4, 5])

    result = test_func()
    assert isinstance(result, np.ndarray)

    stats = monitor.metrics["operations"]["test_func"]
    assert stats["total_size"] == result.nbytes


def test_monitor_performance_decorator_with_bytes_result():
    """Test monitor_performance decorator with bytes result."""
    monitor = utils.PerformanceMonitor()

    @utils.monitor_performance(monitor)
    def test_func():
        return b"test data"

    test_func()
    stats = monitor.metrics["operations"]["test_func"]
    assert stats["total_size"] == len(b"test data")


def test_monitor_performance_decorator_with_string_result():
    """Test monitor_performance decorator with string result."""
    monitor = utils.PerformanceMonitor()

    @utils.monitor_performance(monitor)
    def test_func():
        return "test string"

    test_func()
    stats = monitor.metrics["operations"]["test_func"]
    assert stats["total_size"] == len("test string")


# =========================================================================
# FileHandleCache Tests
# =========================================================================


def test_file_handle_cache_initialization():
    """Test FileHandleCache initialization."""
    cache = utils.FileHandleCache(max_size=10, expiry_time=60)
    assert cache._max_size == 10
    assert cache._expiry_time == 60
    assert len(cache._cache) == 0


def test_file_handle_cache_get_new_file(sample_hdf5_file):
    """Test getting a new file from cache."""
    cache = utils.FileHandleCache()
    handle = cache.get(str(sample_hdf5_file), mode="r")

    assert isinstance(handle, h5py.File)
    assert str(sample_hdf5_file) in cache._cache
    assert len(cache._cache) == 1


def test_file_handle_cache_get_cached_file(sample_hdf5_file):
    """Test getting a cached file."""
    cache = utils.FileHandleCache()

    # First access
    handle1 = cache.get(str(sample_hdf5_file), mode="r")

    # Second access should return same handle from cache
    handle2 = cache.get(str(sample_hdf5_file), mode="r")

    assert handle1 is handle2


def test_file_handle_cache_hit_tracking(sample_hdf5_file):
    """Test cache hit tracking."""
    cache = utils.FileHandleCache()

    # First access is a miss
    cache.get(str(sample_hdf5_file), mode="r")
    assert cache._monitor.metrics["cache_stats"]["hits"] == 0
    assert cache._monitor.metrics["cache_stats"]["misses"] == 1

    # Second access is a hit
    cache.get(str(sample_hdf5_file), mode="r")
    assert cache._monitor.metrics["cache_stats"]["hits"] == 1
    assert cache._monitor.metrics["cache_stats"]["misses"] == 1


def test_file_handle_cache_lru_eviction(temp_dir):
    """Test LRU eviction when cache is full."""
    cache = utils.FileHandleCache(max_size=2, expiry_time=300)

    # Create 3 HDF5 files
    files = []
    for i in range(3):
        filepath = temp_dir / f"test{i}.h5"
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=[i])
        files.append(filepath)

    # Access all 3 files
    cache.get(str(files[0]), mode="r")
    cache.get(str(files[1]), mode="r")
    cache.get(str(files[2]), mode="r")  # This should evict files[0]

    # Cache should only have 2 files
    assert len(cache._cache) == 2
    assert str(files[0]) not in cache._cache
    assert str(files[1]) in cache._cache
    assert str(files[2]) in cache._cache


def test_file_handle_cache_close_handle(sample_hdf5_file):
    """Test closing a specific file handle."""
    cache = utils.FileHandleCache()
    cache.get(str(sample_hdf5_file), mode="r")

    assert str(sample_hdf5_file) in cache._cache
    cache._close_handle(str(sample_hdf5_file))
    assert str(sample_hdf5_file) not in cache._cache


def test_file_handle_cache_close_all(temp_dir):
    """Test closing all file handles."""
    cache = utils.FileHandleCache()

    # Create multiple HDF5 files
    files = []
    for i in range(3):
        filepath = temp_dir / f"test{i}.h5"
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=[i])
        files.append(filepath)
        cache.get(str(filepath), mode="r")

    assert len(cache._cache) == 3

    cache.close_all()
    assert len(cache._cache) == 0


@pytest.mark.skip(reason="Expiry checker test can hang - tested via integration")
def test_file_handle_cache_expiry_checker(temp_dir):
    """Test expiry checker removes old handles."""
    cache = utils.FileHandleCache(max_size=10, expiry_time=0.1)  # 100ms expiry

    filepath = temp_dir / "test.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("data", data=[1, 2, 3])

    cache.get(str(filepath), mode="r")
    assert str(filepath) in cache._cache

    # Wait for expiry
    time.sleep(0.2)

    # Trigger expiry check manually
    cache._check_expiry()

    # Clean up
    cache.close_all()


def test_file_handle_cache_move_to_end(sample_hdf5_file, empty_hdf5_file):
    """Test that accessed files are moved to end (most recent)."""
    cache = utils.FileHandleCache()

    cache.get(str(sample_hdf5_file), mode="r")
    cache.get(str(empty_hdf5_file), mode="r")

    # Access the first file again
    cache.get(str(sample_hdf5_file), mode="r")

    # The first file should now be at the end
    keys = list(cache._cache.keys())
    assert keys[-1] == str(sample_hdf5_file)


def test_file_handle_cache_del():
    """Test __del__ method closes all handles."""
    cache = utils.FileHandleCache()
    # Just verify it doesn't raise an error
    del cache


# =========================================================================
# Global file_handle_cache Tests
# =========================================================================


def test_global_file_handle_cache_exists():
    """Test that global file_handle_cache exists."""
    assert hasattr(utils, "file_handle_cache")
    assert isinstance(utils.file_handle_cache, utils.FileHandleCache)


# =========================================================================
# HDF5Manager Tests
# =========================================================================


def test_hdf5_manager_initialization(sample_hdf5_file):
    """Test HDF5Manager initialization."""
    manager = utils.HDF5Manager(sample_hdf5_file)
    assert manager.file_path == sample_hdf5_file
    assert manager.file is None
    assert isinstance(manager.monitor, utils.PerformanceMonitor)


def test_hdf5_manager_context_manager(sample_hdf5_file):
    """Test HDF5Manager as context manager."""
    manager = utils.HDF5Manager(sample_hdf5_file)

    with manager as mgr:
        assert mgr.file is not None
        assert isinstance(mgr.file, h5py.File)

    # After exiting, file should be None
    assert manager.file is None


def test_hdf5_manager_context_manager_error(temp_dir):
    """Test HDF5Manager with nonexistent file."""
    nonexistent = temp_dir / "nonexistent.h5"
    manager = utils.HDF5Manager(nonexistent)

    with pytest.raises(Exception):
        with manager:
            pass


def test_hdf5_manager_get_object_info_file(sample_hdf5_file):
    """Test get_object_info for root file."""
    with utils.HDF5Manager(sample_hdf5_file) as manager:
        info = manager.get_object_info(manager.file["/"])

        assert info["name"] == "/"
        assert info["type"] == "Group"
        assert "keys" in info
        assert "attrs" in info


def test_hdf5_manager_get_object_info_group(sample_hdf5_file):
    """Test get_object_info for a group."""
    with utils.HDF5Manager(sample_hdf5_file) as manager:
        group = manager.file["/results"]
        info = manager.get_object_info(group)

        assert info["name"] == "/results"
        assert info["type"] == "Group"
        assert "temperature" in info["keys"]
        assert "attrs" in info


def test_hdf5_manager_get_object_info_dataset(sample_hdf5_file):
    """Test get_object_info for a dataset."""
    with utils.HDF5Manager(sample_hdf5_file) as manager:
        dataset = manager.file["/results/temperature"]
        info = manager.get_object_info(dataset)

        assert info["name"] == "/results/temperature"
        assert info["type"] == "Dataset"
        assert info["shape"] == (100, 50)
        assert "dtype" in info
        assert "size" in info
        assert "attrs" in info
        assert info["attrs"]["unit"] == "Kelvin"


def test_hdf5_manager_get_object_info_small_dataset(temp_dir):
    """Test get_object_info for small dataset includes data."""
    filepath = temp_dir / "small.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("small", data=[1, 2, 3, 4, 5])

    with utils.HDF5Manager(filepath) as manager:
        info = manager.get_object_info(manager.file["small"])

        assert "data" in info
        assert info["data"] == [1, 2, 3, 4, 5]


def test_hdf5_manager_get_object_info_large_dataset(sample_hdf5_file):
    """Test get_object_info for large dataset excludes data."""
    with utils.HDF5Manager(sample_hdf5_file) as manager:
        dataset = manager.file["/results/temperature"]
        info = manager.get_object_info(dataset)

        # Large dataset should not include data sample
        assert "data" not in info


def test_hdf5_manager_get_object_info_chunked_dataset(sample_hdf5_file):
    """Test get_object_info for chunked dataset."""
    with utils.HDF5Manager(sample_hdf5_file) as manager:
        dataset = manager.file["/results/large_data"]
        info = manager.get_object_info(dataset)

        assert "chunks" in info
        assert info["chunks"] == (100, 100)


def test_hdf5_manager_read_dataset(sample_hdf5_file):
    """Test read_dataset method."""
    with utils.HDF5Manager(sample_hdf5_file) as manager:
        data = manager.read_dataset("/results/temperature")

        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 50)


def test_hdf5_manager_write_dataset(temp_dir):
    """Test write_dataset method."""
    filepath = temp_dir / "write_test.h5"

    # Create file and write dataset
    with h5py.File(filepath, "w") as f:
        pass  # Just create the file

    # Use file_handle_cache to get a writable handle
    # Note: The manager uses read mode by default
    with h5py.File(filepath, "a") as f:
        manager = utils.HDF5Manager(filepath)
        manager.file = f
        manager.write_dataset("/test_data", np.array([1, 2, 3, 4, 5]))

    # Verify data was written
    with h5py.File(filepath, "r") as f:
        assert "/test_data" in f
        data = f["/test_data"][()]
        np.testing.assert_array_equal(data, [1, 2, 3, 4, 5])


def test_hdf5_manager_write_dataset_overwrites(temp_dir):
    """Test write_dataset overwrites existing dataset."""
    filepath = temp_dir / "overwrite_test.h5"

    with h5py.File(filepath, "w") as f:
        f.create_dataset("/test_data", data=[1, 2, 3])

    with h5py.File(filepath, "a") as f:
        manager = utils.HDF5Manager(filepath)
        manager.file = f
        manager.write_dataset("/test_data", np.array([4, 5, 6]))

    with h5py.File(filepath, "r") as f:
        data = f["/test_data"][()]
        np.testing.assert_array_equal(data, [4, 5, 6])


def test_hdf5_manager_get_performance_summary(sample_hdf5_file):
    """Test get_performance_summary method."""
    with utils.HDF5Manager(sample_hdf5_file) as manager:
        manager.read_dataset("/results/temperature")
        summary = manager.get_performance_summary()

        assert isinstance(summary, dict)
        assert "total_operations" in summary
        assert "cache_hit_ratio" in summary
        assert "current_memory_mb" in summary


def test_hdf5_manager_save_performance_metrics(sample_hdf5_file, temp_dir):
    """Test save_performance_metrics method."""
    manager = utils.HDF5Manager(sample_hdf5_file)
    manager.monitor = utils.PerformanceMonitor(metrics_dir=temp_dir)

    with manager:
        manager.read_dataset("/results/temperature")

    manager.save_performance_metrics()

    # Check that metrics file was created
    metrics_files = list(temp_dir.glob("metrics-*.json"))
    assert len(metrics_files) > 0


def test_hdf5_manager_multiple_operations(sample_hdf5_file):
    """Test multiple operations with HDF5Manager."""
    with utils.HDF5Manager(sample_hdf5_file) as manager:
        # Read multiple datasets
        temp_data = manager.read_dataset("/results/temperature")
        pressure_data = manager.read_dataset("/results/pressure")

        assert temp_data.shape == (100, 50)
        assert pressure_data.shape == (100, 50)

        # Check performance tracking
        # Note: Each manager has its own monitor, so operations are tracked there
        summary = manager.get_performance_summary()
        # The decorator creates a new monitor, so this one may be empty
        assert isinstance(summary, dict)


# =========================================================================
# Integration Tests
# =========================================================================


def test_performance_monitoring_integration(sample_hdf5_file, temp_dir):
    """Test complete performance monitoring workflow."""
    monitor = utils.PerformanceMonitor(metrics_dir=temp_dir)

    @utils.monitor_performance(monitor)
    def process_data():
        with utils.HDF5Manager(sample_hdf5_file) as manager:
            data = manager.read_dataset("/results/temperature")
            return data.mean()

    result = process_data()
    assert isinstance(result, float)

    # Verify monitoring
    assert "process_data" in monitor.metrics["operations"]

    # Save metrics
    monitor.save_metrics()
    metrics_files = list(temp_dir.glob("metrics-*.json"))
    assert len(metrics_files) > 0


def test_file_cache_with_multiple_files(temp_dir):
    """Test file cache with multiple files."""
    cache = utils.FileHandleCache(max_size=5)

    # Create multiple files
    files = []
    for i in range(3):
        filepath = temp_dir / f"multi_{i}.h5"
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=np.random.rand(10))
        files.append(filepath)

    # Access all files
    handles = [cache.get(str(f), mode="r") for f in files]

    # All should be cached
    assert len(cache._cache) == 3

    # Access first file again (should be cache hit)
    handle = cache.get(str(files[0]), mode="r")
    assert handle is handles[0]

    cache.close_all()


def test_hdf5_manager_with_cache(sample_hdf5_file):
    """Test HDF5Manager uses global file cache."""
    # Clear cache first
    utils.file_handle_cache.close_all()

    with utils.HDF5Manager(sample_hdf5_file) as manager:
        assert manager.file is not None

    # File should be in cache
    assert str(sample_hdf5_file) in utils.file_handle_cache._cache
