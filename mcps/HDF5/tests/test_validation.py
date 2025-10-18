"""Validation tests for HDF5 MCP v2.0."""
import pytest
import ast
from pathlib import Path


def test_all_files_syntax_valid():
    """Test all Python files have valid syntax."""
    src_dir = Path(__file__).parent.parent / 'src' / 'hdf5_mcp'
    python_files = list(src_dir.rglob('*.py'))

    assert len(python_files) > 15, f"Expected 15+ Python files, found {len(python_files)}"

    for py_file in python_files:
        with open(py_file) as f:
            try:
                ast.parse(f.read())
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file.name}: {e}")


def test_sse_protocol_versions():
    """Test SSE transport supports MCP protocol versions."""
    from hdf5_mcp.transports.sse_transport import SUPPORTED_VERSIONS

    assert '2025-06-18' in SUPPORTED_VERSIONS
    assert '2025-03-26' in SUPPORTED_VERSIONS


def test_sse_localhost_binding():
    """Test SSE enforces localhost-only binding for security."""
    from hdf5_mcp.transports.sse_transport import SSETransport
    from hdf5_mcp.transports.base import TransportConfig, TransportType

    # Test default is localhost
    transport = SSETransport()
    assert transport.config.host == '127.0.0.1'

    # Test 0.0.0.0 gets overridden
    config = TransportConfig(transport_type=TransportType.SSE, host='0.0.0.0')
    transport = SSETransport(config)
    assert transport.config.host == '127.0.0.1'


def test_documentation_exists():
    """Test all required docs exist."""
    docs_dir = Path(__file__).parent.parent / 'docs'

    required_docs = [
        'TRANSPORTS.md',
        'ARCHITECTURE.md',
        'EXAMPLES.md',
        'TOOLS.md',
        'MIGRATION.md'
    ]

    for doc in required_docs:
        doc_path = docs_dir / doc
        assert doc_path.exists(), f"Missing documentation: {doc}"


def test_entry_point_configured():
    """Test pyproject.toml has correct entry point."""
    import tomllib

    pyproject = Path(__file__).parent.parent / 'pyproject.toml'
    with open(pyproject, 'rb') as f:
        config = tomllib.load(f)

    assert 'project' in config
    assert 'scripts' in config['project']
    assert 'hdf5-mcp' in config['project']['scripts']
    assert config['project']['scripts']['hdf5-mcp'] == 'main:main'


def test_version_is_2_0():
    """Test version is 2.0.0."""
    import tomllib

    pyproject = Path(__file__).parent.parent / 'pyproject.toml'
    with open(pyproject, 'rb') as f:
        config = tomllib.load(f)

    assert config['project']['version'] == '2.0.0'


def test_dependencies_include_aiohttp():
    """Test aiohttp is in dependencies (needed for SSE)."""
    import tomllib

    pyproject = Path(__file__).parent.parent / 'pyproject.toml'
    with open(pyproject, 'rb') as f:
        config = tomllib.load(f)

    deps = config['project']['dependencies']
    assert any('aiohttp' in dep for dep in deps), "aiohttp must be in dependencies for SSE transport"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
