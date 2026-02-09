"""Tests for package initialization and metadata."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestPackageMetadata:
    """Test package-level metadata and initialization."""

    def test_version_exists(self):
        """Test that package version is defined."""
        import __init__ as pkg

        assert hasattr(pkg, "__version__")

    def test_version_format(self):
        """Test that version follows semantic versioning format."""
        import __init__ as pkg

        version = pkg.__version__
        assert isinstance(version, str)
        assert len(version.split(".")) == 3

    def test_version_value(self):
        """Test that version has expected value."""
        import __init__ as pkg

        assert pkg.__version__ == "1.0.0"

    def test_docstring_exists(self):
        """Test that package has a docstring."""
        import __init__ as pkg

        assert pkg.__doc__ is not None
        assert len(pkg.__doc__) > 0

    def test_docstring_content(self):
        """Test that docstring mentions NDP."""
        import __init__ as pkg

        assert "National Data Platform" in pkg.__doc__ or "NDP" in pkg.__doc__
        assert "MCP" in pkg.__doc__
