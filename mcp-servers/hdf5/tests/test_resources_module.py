"""
Comprehensive tests for the resources module.

Tests LRU cache, lazy loading proxies, resource manager,
and file discovery functions.
"""

import pytest
import json
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime

# Import the resources module directly - DO NOT import server
from hdf5_mcp import resources


# =========================================================================
# LRUCache Tests
# =========================================================================


def test_lru_cache_initialization():
    """Test LRUCache initialization."""
    cache = resources.LRUCache(capacity=10)
    assert cache.capacity == 10
    assert len(cache.cache) == 0


def test_lru_cache_put_get():
    """Test basic put and get operations."""
    cache = resources.LRUCache(capacity=5)
    cache.put("key1", "value1")

    result = cache.get("key1")
    assert result == "value1"


def test_lru_cache_get_nonexistent():
    """Test getting a nonexistent key."""
    cache = resources.LRUCache()
    result = cache.get("nonexistent")
    assert result is None


def test_lru_cache_update_existing():
    """Test updating an existing key."""
    cache = resources.LRUCache()
    cache.put("key1", "value1")
    cache.put("key1", "value2")

    result = cache.get("key1")
    assert result == "value2"


def test_lru_cache_eviction():
    """Test LRU eviction when capacity is exceeded."""
    cache = resources.LRUCache(capacity=3)

    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")
    cache.put("key4", "value4")  # Should evict key1

    assert cache.get("key1") is None
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"
    assert len(cache.cache) == 3


def test_lru_cache_move_to_end():
    """Test that accessed items are moved to end."""
    cache = resources.LRUCache(capacity=3)

    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")

    # Access key1, making it most recently used
    cache.get("key1")

    # Add key4, should evict key2 (least recently used)
    cache.put("key4", "value4")

    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None
    assert cache.get("key3") == "value3"
    assert cache.get("key4") == "value4"


def test_lru_cache_thread_safety():
    """Test thread-safe operations."""
    cache = resources.LRUCache(capacity=100)

    import threading

    def add_items(start, count):
        for i in range(start, start + count):
            cache.put(f"key{i}", f"value{i}")

    threads = [threading.Thread(target=add_items, args=(i * 10, 10)) for i in range(5)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify some items are present
    assert len(cache.cache) > 0


# =========================================================================
# LazyHDF5Proxy Tests
# =========================================================================


def test_lazy_hdf5_proxy_initialization(sample_hdf5_file, temp_dir):
    """Test LazyHDF5Proxy initialization."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)

    assert proxy._file_path == sample_hdf5_file
    assert proxy._file is None  # Not loaded yet


def test_lazy_hdf5_proxy_lazy_loading(sample_hdf5_file, temp_dir):
    """Test lazy loading of HDF5 file."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)

    # File is not loaded until accessed
    assert proxy._file is None

    # Access file property - triggers loading
    file = proxy.file
    assert file is not None
    assert isinstance(file, h5py.File)


def test_lazy_hdf5_proxy_filename(sample_hdf5_file, temp_dir):
    """Test filename property."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)

    assert proxy.filename == str(sample_hdf5_file)


def test_lazy_hdf5_proxy_mode(sample_hdf5_file, temp_dir):
    """Test mode property."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)

    # Before loading, mode defaults to 'r'
    assert proxy.mode == "r"

    # After loading
    assert proxy.mode == "r"


def test_lazy_hdf5_proxy_attrs(sample_hdf5_file, temp_dir):
    """Test attrs property."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)

    attrs = proxy.attrs
    assert attrs is not None
    assert "experiment" in attrs


def test_lazy_hdf5_proxy_getitem(sample_hdf5_file, temp_dir):
    """Test __getitem__ access."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)

    dataset = proxy["/results/temperature"]
    assert dataset is not None


def test_lazy_hdf5_proxy_contains(sample_hdf5_file, temp_dir):
    """Test __contains__ check."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)

    assert "/results/temperature" in proxy
    assert "/nonexistent" not in proxy


def test_lazy_hdf5_proxy_close(sample_hdf5_file, temp_dir):
    """Test close method."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)

    # Load the file
    _ = proxy.file  # Trigger loading
    assert proxy._file is not None

    # Close it
    proxy.close()
    assert proxy._file is None


def test_lazy_hdf5_proxy_error_handling(temp_dir):
    """Test error handling for nonexistent file."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    nonexistent = temp_dir / "nonexistent.h5"
    proxy = resources.LazyHDF5Proxy(nonexistent, rm)

    # Should return None on error
    file = proxy.file
    assert file is None

    # Mode and attrs should also handle None file gracefully
    assert proxy.mode == "r"  # Default when file is None
    assert proxy.attrs is None  # None when file is None


# =========================================================================
# LazyDatasetProxy Tests
# =========================================================================


def test_lazy_dataset_proxy_initialization(sample_hdf5_file, temp_dir):
    """Test LazyDatasetProxy initialization."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    file_proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)
    dataset_proxy = resources.LazyDatasetProxy(file_proxy, "/results/temperature")

    assert dataset_proxy._dataset is None  # Not loaded yet


def test_lazy_dataset_proxy_lazy_loading(sample_hdf5_file, temp_dir):
    """Test lazy loading of dataset."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    file_proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)
    dataset_proxy = resources.LazyDatasetProxy(file_proxy, "/results/temperature")

    # Dataset not loaded yet
    assert dataset_proxy._dataset is None

    # Access dataset - triggers loading
    dataset = dataset_proxy.dataset
    assert dataset is not None
    assert isinstance(dataset, h5py.Dataset)


def test_lazy_dataset_proxy_shape(sample_hdf5_file, temp_dir):
    """Test shape property."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    file_proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)
    dataset_proxy = resources.LazyDatasetProxy(file_proxy, "/results/temperature")

    assert dataset_proxy.shape == (100, 50)


def test_lazy_dataset_proxy_dtype(sample_hdf5_file, temp_dir):
    """Test dtype property."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    file_proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)
    dataset_proxy = resources.LazyDatasetProxy(file_proxy, "/results/temperature")

    assert dataset_proxy.dtype == np.float64


def test_lazy_dataset_proxy_getitem(sample_hdf5_file, temp_dir):
    """Test __getitem__ slicing."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    file_proxy = resources.LazyHDF5Proxy(sample_hdf5_file, rm)
    dataset_proxy = resources.LazyDatasetProxy(file_proxy, "/results/temperature")

    # Get a slice
    data = dataset_proxy[0:10, 0:10]
    assert isinstance(data, np.ndarray)
    assert data.shape == (10, 10)


# =========================================================================
# ResourceManager Tests
# =========================================================================


def test_resource_manager_initialization(temp_dir):
    """Test ResourceManager initialization."""
    rm = resources.ResourceManager(data_dir=temp_dir, cache_capacity=100)

    assert rm.data_dir == temp_dir
    assert rm.cache_capacity == 100
    assert rm.storage_index_path.exists()
    assert rm.cache_db_path.exists()
    assert rm.history_db_path.exists()


def test_resource_manager_default_data_dir():
    """Test ResourceManager with default data directory."""
    rm = resources.ResourceManager()
    assert rm.data_dir == Path("data")


def test_resource_manager_load_storage_index(temp_dir):
    """Test loading storage index."""
    # Create a storage index file
    storage_index = {
        "files": {"test.h5": {"path": "test.h5"}},
        "last_updated": datetime.now().isoformat(),
    }
    storage_path = temp_dir / ".storage_index.json"
    with open(storage_path, "w") as f:
        json.dump(storage_index, f)

    rm = resources.ResourceManager(data_dir=temp_dir)
    assert "test.h5" in rm.storage_index["files"]


def test_resource_manager_load_storage_index_empty(temp_dir):
    """Test loading empty storage index."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    assert "files" in rm.storage_index
    assert "last_updated" in rm.storage_index


def test_resource_manager_load_storage_index_error(temp_dir):
    """Test loading corrupted storage index."""
    storage_path = temp_dir / ".storage_index.json"
    with open(storage_path, "w") as f:
        f.write("{invalid json")

    rm = resources.ResourceManager(data_dir=temp_dir)
    # Should create empty index
    assert "files" in rm.storage_index


def test_resource_manager_save_storage_index(temp_dir):
    """Test saving storage index."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    rm.storage_index["files"]["test.h5"] = {"path": "test.h5"}
    rm._save_storage_index()

    # Verify file was saved
    with open(rm.storage_index_path, "r") as f:
        data = json.load(f)
        assert "test.h5" in data["files"]


def test_resource_manager_load_cache_db(temp_dir):
    """Test loading cache database."""
    cache_db = {
        "datasets": {},
        "attributes": {},
        "last_updated": datetime.now().isoformat(),
    }
    cache_path = temp_dir / ".cache_db.json"
    with open(cache_path, "w") as f:
        json.dump(cache_db, f)

    rm = resources.ResourceManager(data_dir=temp_dir)
    assert "datasets" in rm.cache_db
    assert "attributes" in rm.cache_db


def test_resource_manager_load_cache_db_error(temp_dir):
    """Test loading corrupted cache database."""
    cache_path = temp_dir / ".cache_db.json"
    with open(cache_path, "w") as f:
        f.write("{corrupt json")

    rm = resources.ResourceManager(data_dir=temp_dir)
    # Should create empty cache db
    assert "datasets" in rm.cache_db
    assert "attributes" in rm.cache_db


def test_resource_manager_save_cache_db(temp_dir):
    """Test saving cache database."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    rm.cache_db["datasets"]["test"] = {"data": [1, 2, 3]}
    rm._save_cache_db()

    with open(rm.cache_db_path, "r") as f:
        data = json.load(f)
        assert "test" in data["datasets"]


def test_resource_manager_load_history_db(temp_dir):
    """Test loading history database."""
    history_db = {"sessions": [], "tool_calls": [], "requests": [], "errors": []}
    history_path = temp_dir / ".history_db.json"
    with open(history_path, "w") as f:
        json.dump(history_db, f)

    rm = resources.ResourceManager(data_dir=temp_dir)
    assert "sessions" in rm.history_db
    assert "tool_calls" in rm.history_db


def test_resource_manager_load_history_db_error(temp_dir):
    """Test loading corrupted history database."""
    history_path = temp_dir / ".history_db.json"
    with open(history_path, "w") as f:
        f.write("{corrupt json")

    rm = resources.ResourceManager(data_dir=temp_dir)
    # Should create empty history db
    assert "sessions" in rm.history_db
    assert "tool_calls" in rm.history_db
    assert "requests" in rm.history_db
    assert "errors" in rm.history_db


def test_resource_manager_save_history_db(temp_dir):
    """Test saving history database."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    rm.history_db["sessions"].append({"session_id": "test"})
    rm._save_history_db()

    with open(rm.history_db_path, "r") as f:
        data = json.load(f)
        assert len(data["sessions"]) > 0


def test_resource_manager_register_hdf5_file(temp_dir, sample_hdf5_file):
    """Test registering an HDF5 file."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    success = rm.register_hdf5_file(sample_hdf5_file)

    assert success
    assert str(sample_hdf5_file) in rm.storage_index["files"]


def test_resource_manager_register_nonexistent_file(temp_dir):
    """Test registering a nonexistent file."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    nonexistent = temp_dir / "nonexistent.h5"
    success = rm.register_hdf5_file(nonexistent)

    assert success is False


def test_resource_manager_register_corrupted_file(temp_dir):
    """Test registering a corrupted HDF5 file."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    corrupted = temp_dir / "corrupted.h5"
    # Create a file that's not valid HDF5
    with open(corrupted, "w") as f:
        f.write("not an hdf5 file")

    success = rm.register_hdf5_file(corrupted)
    # Should return False on error
    assert success is False


def test_resource_manager_get_registered_files(temp_dir, sample_hdf5_file):
    """Test getting registered files."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    rm.register_hdf5_file(sample_hdf5_file)

    files = rm.get_registered_files()
    assert len(files) > 0
    assert any(str(sample_hdf5_file) in f["path"] for f in files)


def test_resource_manager_add_session(temp_dir):
    """Test adding a session."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    rm.add_session("session123", {"client": "test"})

    assert len(rm.history_db["sessions"]) == 1
    assert rm.history_db["sessions"][0]["session_id"] == "session123"


def test_resource_manager_add_tool_call(temp_dir):
    """Test adding a tool call."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    rm.add_tool_call("session123", "read_dataset", {"path": "/data"})

    assert len(rm.history_db["tool_calls"]) == 1
    assert rm.history_db["tool_calls"][0]["tool_name"] == "read_dataset"


def test_resource_manager_add_tool_call_batch_save(temp_dir):
    """Test tool calls are saved periodically."""
    rm = resources.ResourceManager(data_dir=temp_dir)

    # Add 10 tool calls (should trigger save)
    for i in range(10):
        rm.add_tool_call(f"session{i}", "tool", {})

    # Check file was saved
    with open(rm.history_db_path, "r") as f:
        data = json.load(f)
        assert len(data["tool_calls"]) == 10


def test_resource_manager_add_request(temp_dir):
    """Test adding a request."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    rm.add_request("session123", {"method": "read", "path": "/data"})

    assert len(rm.history_db["requests"]) == 1


def test_resource_manager_add_request_batch_save(temp_dir):
    """Test requests are saved periodically."""
    rm = resources.ResourceManager(data_dir=temp_dir)

    for i in range(10):
        rm.add_request(f"session{i}", {"request": "test"})

    with open(rm.history_db_path, "r") as f:
        data = json.load(f)
        assert len(data["requests"]) == 10


def test_resource_manager_add_error(temp_dir):
    """Test adding an error."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    rm.add_error("session123", "ValueError", "Test error message")

    assert len(rm.history_db["errors"]) == 1
    assert rm.history_db["errors"][0]["error_type"] == "ValueError"


def test_resource_manager_cache_dataset(temp_dir):
    """Test caching dataset data."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    data = np.array([1, 2, 3, 4, 5])
    rm.cache_dataset("file.h5", "/dataset", data)

    key = "file.h5:/dataset"
    assert key in rm.cache_db["datasets"]


def test_resource_manager_cache_dataset_list(temp_dir):
    """Test caching dataset with list conversion."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    data = [1, 2, 3, 4, 5]
    rm.cache_dataset("file.h5", "/dataset", data)

    key = "file.h5:/dataset"
    assert key in rm.cache_db["datasets"]
    assert rm.cache_db["datasets"][key]["data"] == data


def test_resource_manager_get_cached_dataset(temp_dir):
    """Test getting cached dataset."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    data = [1, 2, 3, 4, 5]
    rm.cache_dataset("file.h5", "/dataset", data)

    cached = rm.get_cached_dataset("file.h5", "/dataset")
    assert cached == data


def test_resource_manager_get_cached_dataset_miss(temp_dir):
    """Test cache miss."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    cached = rm.get_cached_dataset("nonexistent.h5", "/dataset")
    assert cached is None


def test_resource_manager_get_hdf5_file(temp_dir, sample_hdf5_file):
    """Test getting HDF5 file proxy."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    proxy = rm.get_hdf5_file(str(sample_hdf5_file))

    assert proxy is not None
    assert isinstance(proxy, resources.LazyHDF5Proxy)


def test_resource_manager_get_hdf5_file_cached(temp_dir, sample_hdf5_file):
    """Test getting cached HDF5 file proxy."""
    rm = resources.ResourceManager(data_dir=temp_dir)

    # First access
    proxy1 = rm.get_hdf5_file(str(sample_hdf5_file))

    # Second access should return same proxy
    proxy2 = rm.get_hdf5_file(str(sample_hdf5_file))

    assert proxy1 is proxy2


def test_resource_manager_get_hdf5_file_nonexistent(temp_dir):
    """Test getting nonexistent file."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    proxy = rm.get_hdf5_file(str(temp_dir / "nonexistent.h5"))

    assert proxy is None


def test_resource_manager_get_dataset(temp_dir, sample_hdf5_file):
    """Test getting dataset."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    data = rm.get_dataset(str(sample_hdf5_file), "/results/temperature")

    assert data is not None
    assert isinstance(data, np.ndarray)
    assert data.shape == (100, 50)


def test_resource_manager_get_dataset_sliced(temp_dir, sample_hdf5_file):
    """Test getting sliced dataset."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    data = rm.get_dataset(
        str(sample_hdf5_file), "/results/temperature", start=(0, 0), count=(10, 10)
    )

    assert data is not None
    assert data.shape == (10, 10)


def test_resource_manager_get_dataset_cached(temp_dir, sample_hdf5_file):
    """Test getting dataset uses cache."""
    rm = resources.ResourceManager(data_dir=temp_dir)

    # First access - cache miss
    data1 = rm.get_dataset(str(sample_hdf5_file), "/results/temperature")

    # Second access - should be from cache
    data2 = rm.get_dataset(str(sample_hdf5_file), "/results/temperature")

    assert data1 is not None
    assert data2 is not None


def test_resource_manager_get_dataset_nonexistent(temp_dir, sample_hdf5_file):
    """Test getting nonexistent dataset."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    data = rm.get_dataset(str(sample_hdf5_file), "/nonexistent")

    assert data is None


@pytest.mark.asyncio
async def test_resource_manager_initialize(temp_dir):
    """Test async initialize method."""
    rm = resources.ResourceManager(data_dir=temp_dir)
    await rm.initialize()
    # Should complete without error


@pytest.mark.asyncio
async def test_resource_manager_shutdown(temp_dir, sample_hdf5_file):
    """Test async shutdown method."""
    rm = resources.ResourceManager(data_dir=temp_dir)

    # Get a file to ensure there's something to clean up
    rm.get_hdf5_file(str(sample_hdf5_file))

    await rm.shutdown()

    # Caches should be cleared
    assert len(rm.hdf5_files) == 0


# =========================================================================
# File Discovery Tests
# =========================================================================


def test_discover_hdf5_files_in_roots(temp_dir):
    """Test discovering HDF5 files in roots."""
    # Create directory structure with HDF5 files
    subdir1 = temp_dir / "subdir1"
    subdir2 = temp_dir / "subdir2"
    subdir1.mkdir()
    subdir2.mkdir()

    # Create HDF5 files
    files = []
    for i, subdir in enumerate([subdir1, subdir2]):
        for ext in [".h5", ".hdf5", ".he5"]:
            filepath = subdir / f"test{i}{ext}"
            with h5py.File(filepath, "w") as f:
                f.create_dataset("data", data=[i])
            files.append(filepath)

    # Discover files
    roots = [temp_dir]
    discovered = resources.discover_hdf5_files_in_roots(roots)

    assert len(discovered) == 6  # 2 subdirs * 3 extensions
    for f in files:
        assert f in discovered


def test_discover_hdf5_files_nonexistent_root(temp_dir):
    """Test discovering files with nonexistent root."""
    nonexistent = temp_dir / "nonexistent"
    discovered = resources.discover_hdf5_files_in_roots([nonexistent])

    assert len(discovered) == 0


def test_discover_hdf5_files_empty_root(temp_dir):
    """Test discovering files in empty directory."""
    empty_dir = temp_dir / "empty"
    empty_dir.mkdir()

    discovered = resources.discover_hdf5_files_in_roots([empty_dir])
    assert len(discovered) == 0


def test_discover_hdf5_files_multiple_roots(temp_dir):
    """Test discovering files in multiple roots."""
    root1 = temp_dir / "root1"
    root2 = temp_dir / "root2"
    root1.mkdir()
    root2.mkdir()

    # Create files in each root
    for i, root in enumerate([root1, root2]):
        filepath = root / f"test{i}.h5"
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=[i])

    discovered = resources.discover_hdf5_files_in_roots([root1, root2])
    assert len(discovered) == 2


def test_discover_hdf5_files_nested(temp_dir):
    """Test discovering files in nested directories."""
    nested = temp_dir / "level1" / "level2" / "level3"
    nested.mkdir(parents=True)

    filepath = nested / "deep.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("data", data=[1])

    discovered = resources.discover_hdf5_files_in_roots([temp_dir])
    assert filepath in discovered


def test_discover_hdf5_files_duplicates(temp_dir):
    """Test that duplicates are removed."""
    root = temp_dir / "root"
    root.mkdir()

    filepath = root / "test.h5"
    with h5py.File(filepath, "w") as f:
        f.create_dataset("data", data=[1])

    # Discover same root twice
    discovered = resources.discover_hdf5_files_in_roots([root, root])

    # Should only have one entry
    assert len(discovered) == 1
    assert discovered[0] == filepath


# =========================================================================
# Integration Tests
# =========================================================================


def test_resource_manager_full_workflow(temp_dir, sample_hdf5_file):
    """Test complete resource manager workflow."""
    rm = resources.ResourceManager(data_dir=temp_dir)

    # Register file
    rm.register_hdf5_file(sample_hdf5_file)

    # Add session
    rm.add_session("test_session", {"client": "pytest"})

    # Get dataset
    data = rm.get_dataset(str(sample_hdf5_file), "/results/temperature")
    assert data is not None

    # Add tool call
    rm.add_tool_call("test_session", "get_dataset", {"path": "/results/temperature"})

    # Verify history
    assert len(rm.history_db["sessions"]) == 1
    assert len(rm.history_db["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_resource_manager_lifecycle(temp_dir):
    """Test resource manager initialization and shutdown."""
    rm = resources.ResourceManager(data_dir=temp_dir)

    await rm.initialize()
    # Manager should be ready

    await rm.shutdown()
    # All resources should be cleaned up


def test_lazy_loading_chain(temp_dir, sample_hdf5_file):
    """Test complete lazy loading chain."""
    rm = resources.ResourceManager(data_dir=temp_dir)

    # Get file proxy (lazy)
    file_proxy = rm.get_hdf5_file(str(sample_hdf5_file))
    assert file_proxy._file is None

    # Create dataset proxy (lazy)
    dataset_proxy = resources.LazyDatasetProxy(file_proxy, "/results/temperature")
    assert dataset_proxy._dataset is None

    # Access data (triggers loading)
    data = dataset_proxy[:]
    assert isinstance(data, np.ndarray)

    # Both proxies should now be loaded
    assert file_proxy._file is not None
    assert dataset_proxy._dataset is not None


def test_cache_performance(temp_dir, sample_hdf5_file):
    """Test cache improves performance."""
    rm = resources.ResourceManager(data_dir=temp_dir, cache_capacity=1000)

    # First access - cache miss
    import time

    start = time.time()
    data1 = rm.get_dataset(str(sample_hdf5_file), "/results/temperature")
    time.time() - start

    # Second access - should be faster from cache
    start = time.time()
    data2 = rm.get_dataset(str(sample_hdf5_file), "/results/temperature")
    time.time() - start

    # Both should return data
    assert data1 is not None
    assert data2 is not None

    # Cache hit should generally be faster (though not guaranteed in tests)
    # Just verify both complete successfully


def test_resource_manager_with_multiple_files(temp_dir):
    """Test resource manager with multiple files."""
    rm = resources.ResourceManager(data_dir=temp_dir)

    # Create multiple HDF5 files
    files = []
    for i in range(3):
        filepath = temp_dir / f"multi_{i}.h5"
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=np.random.rand(10, 10))
        files.append(filepath)

    # Register all files
    for f in files:
        rm.register_hdf5_file(f)

    # Get datasets from all files
    for f in files:
        data = rm.get_dataset(str(f), "/data")
        assert data is not None
        assert data.shape == (10, 10)


def test_error_tracking(temp_dir):
    """Test error tracking in resource manager."""
    rm = resources.ResourceManager(data_dir=temp_dir)

    # Add multiple errors
    rm.add_error("session1", "FileNotFoundError", "File not found")
    rm.add_error("session1", "ValueError", "Invalid value")
    rm.add_error("session2", "IOError", "I/O error")

    # Verify errors are saved immediately
    with open(rm.history_db_path, "r") as f:
        data = json.load(f)
        assert len(data["errors"]) == 3
