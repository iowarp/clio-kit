"""
Tests for the main package __init__.py file.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_package_metadata():
    """Test that package metadata is accessible."""
    import src

    assert hasattr(src, "__version__")
    assert hasattr(src, "__author__")
    assert src.__version__ == "1.0.0"
    assert src.__author__ == "IoWarp Scientific MCPs"


def test_package_docstring():
    """Test that package has a docstring."""
    import src

    assert src.__doc__ is not None
    assert "Slurm MCP Server" in src.__doc__
    assert "Model Context Protocol" in src.__doc__
