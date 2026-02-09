"""
Comprehensive tests for ParaView Visualization Engine
Tests all methods in paraview_capabilities.py for 100% coverage
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import logging


@pytest.fixture
def mock_paraview():
    """Mock paraview.simple module"""
    with patch.dict(
        "sys.modules",
        {
            "paraview": MagicMock(),
            "paraview.simple": MagicMock(),
            "paraview.servermanager": MagicMock(),
            "paraview.collaboration": MagicMock(),
            "adios2": MagicMock(),
            "vtk": MagicMock(),
            "vtk.util": MagicMock(),
            "vtk.util.numpy_support": MagicMock(),
        },
    ):
        yield


@pytest.fixture
def engine(mock_paraview):
    """Create VisualizationEngine instance with mocked ParaView"""
    from paraview_mcp.implementation.paraview_capabilities import VisualizationEngine

    return VisualizationEngine("test_host", 9999)


class TestVisualizationEngineInit:
    """Test VisualizationEngine initialization"""

    def test_init_default_params(self, mock_paraview):
        """Test initialization with default parameters"""
        from paraview_mcp.implementation.paraview_capabilities import (
            VisualizationEngine,
        )

        engine = VisualizationEngine()

        assert engine.connection is None
        assert engine.server_host == "localhost"
        assert engine.server_port == 11111
        assert engine.primary_data_source is None
        assert engine._source_metadata == {}
        assert engine._active_visualizations == {}
        assert engine._data_directory == ""

    def test_init_custom_params(self, mock_paraview):
        """Test initialization with custom parameters"""
        from paraview_mcp.implementation.paraview_capabilities import (
            VisualizationEngine,
        )

        engine = VisualizationEngine("custom_host", 12345)

        assert engine.server_host == "custom_host"
        assert engine.server_port == 12345
        assert isinstance(engine.logger, logging.Logger)


class TestGetSourceName:
    """Test _get_source_name helper method"""

    def test_get_source_name_valid_proxy(self, engine, mock_paraview):
        """Test getting name from valid proxy"""
        from paraview.simple import GetSources

        mock_proxy = Mock()
        GetSources.return_value = {("TestSource", ""): mock_proxy}

        name = engine._get_source_name(mock_proxy)
        assert name == "TestSource"

    def test_get_source_name_none_proxy(self, engine, mock_paraview):
        """Test getting name from None proxy"""
        name = engine._get_source_name(None)
        assert name == ""

    def test_get_source_name_not_found(self, engine, mock_paraview):
        """Test getting name when proxy not in sources dict"""
        from paraview.simple import GetSources

        GetSources.return_value = {("OtherSource", ""): Mock()}

        name = engine._get_source_name(Mock())
        assert name == ""

    def test_get_source_name_exception(self, engine, mock_paraview):
        """Test exception handling in _get_source_name"""
        from paraview.simple import GetSources

        GetSources.side_effect = Exception("Test error")

        name = engine._get_source_name(Mock())
        assert name == ""


class TestConnection:
    """Test connection management"""

    def test_connect_success(self, engine, mock_paraview):
        """Test successful connection"""
        from paraview.simple import Connect, GetActiveView

        mock_connection = Mock()
        mock_view = Mock()
        Connect.return_value = mock_connection
        GetActiveView.return_value = mock_view

        with patch("importlib.util.find_spec", return_value=Mock()):
            result = engine.connect("test_server", 11111)

            assert result is True
            assert engine.connection == mock_connection
            Connect.assert_called_once()

    def test_connect_no_paraview_module(self, mock_paraview):
        """Test connection when paraview.simple not available"""
        with patch("importlib.util.find_spec", return_value=None):
            from paraview_mcp.implementation.paraview_capabilities import (
                VisualizationEngine,
            )

            engine = VisualizationEngine()

            result = engine.connect()
            assert result is False

    def test_connect_exception(self, engine, mock_paraview):
        """Test connection failure with exception"""
        from paraview.simple import Connect

        Connect.side_effect = Exception("Connection failed")

        result = engine.connect()
        assert result is False


class TestReadDatafile:
    """Test read_datafile method"""

    def test_read_datafile_no_path(self, engine, mock_paraview):
        """Test reading with no file path"""
        success, message, reader, name = engine.read_datafile("")

        assert success is False
        assert "No file path" in message
        assert reader is None
        assert name == ""

    def test_read_datafile_not_exist(self, engine, mock_paraview):
        """Test reading non-existent file"""
        with patch("os.path.exists", return_value=False):
            success, message, reader, name = engine.read_datafile("/fake/path.vtk")

            assert success is False
            assert "does not exist" in message
            assert reader is None

    def test_read_datafile_vtk_success(self, engine, mock_paraview):
        """Test reading VTK file successfully"""
        from paraview.simple import OpenDataFile, Show, GetActiveView

        mock_reader = Mock()
        mock_view = Mock()
        mock_display = Mock()

        OpenDataFile.return_value = mock_reader
        GetActiveView.return_value = mock_view
        Show.return_value = mock_display

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000),
            patch("os.path.dirname", return_value="/test/dir"),
            patch("os.path.basename", return_value="test.vtk"),
            patch("os.path.splitext", return_value=("test", ".vtk")),
            patch.object(engine, "_get_source_name", return_value="test.vtk"),
        ):
            success, message, reader, name = engine.read_datafile("/test/dir/test.vtk")

            assert success is True
            assert "Successfully loaded" in message
            assert reader == mock_reader
            assert name == "test.vtk"
            assert engine.primary_data_source == mock_reader
            assert engine._data_directory == "/test/dir"

    def test_read_datafile_bp5_native_success(self, engine, mock_paraview):
        """Test reading BP5 file with native support"""
        from paraview.simple import OpenDataFile, Show, GetActiveView

        mock_reader = Mock()
        mock_view = Mock()

        OpenDataFile.return_value = mock_reader
        GetActiveView.return_value = mock_view
        Show.return_value = Mock()

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000),
            patch("os.path.dirname", return_value="/test"),
            patch("os.path.basename", return_value="test.bp5"),
            patch("os.path.splitext", return_value=("test", ".bp5")),
            patch.object(
                engine,
                "_check_paraview_adios2_support",
                return_value={"has_support": True},
            ),
            patch.object(engine, "_get_source_name", return_value="test.bp5"),
        ):
            success, message, reader, name = engine.read_datafile("/test/test.bp5")

            assert success is True
            assert "ADIOS2/BP5" in message

    def test_read_datafile_bp5_no_adios2(self, engine, mock_paraview):
        """Test reading BP5 without ADIOS2 available"""
        from paraview_mcp.implementation import paraview_capabilities

        original_adios2 = paraview_capabilities.ADIOS2_AVAILABLE
        paraview_capabilities.ADIOS2_AVAILABLE = False

        try:
            with (
                patch("os.path.exists", return_value=True),
                patch("os.path.getsize", return_value=1000),
                patch("os.path.basename", return_value="test.bp"),
                patch("os.path.splitext", return_value=("test", ".bp")),
                patch.object(
                    engine,
                    "_check_paraview_adios2_support",
                    return_value={"has_support": False},
                ),
            ):
                from paraview.simple import OpenDataFile

                OpenDataFile.return_value = None

                success, message, reader, name = engine.read_datafile("/test/test.bp")

                assert success is False
                assert "ADIOS2 not available" in message
        finally:
            paraview_capabilities.ADIOS2_AVAILABLE = original_adios2

    def test_read_datafile_raw_success(self, engine, mock_paraview):
        """Test reading RAW file"""
        from paraview.simple import Show, GetActiveView

        mock_reader = Mock()
        mock_view = Mock()

        GetActiveView.return_value = mock_view
        Show.return_value = Mock()

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000),
            patch("os.path.dirname", return_value="/test"),
            patch("os.path.basename", return_value="volume_256x256x256_uint8.raw"),
            patch("os.path.splitext", return_value=("volume", ".raw")),
            patch.object(engine, "_configure_raw_reader", return_value=mock_reader),
            patch.object(engine, "_get_source_name", return_value="volume.raw"),
        ):
            success, message, reader, name = engine.read_datafile(
                "/test/volume_256x256x256_uint8.raw"
            )

            assert success is True
            assert "RAW volume format" in message

    def test_read_datafile_raw_config_failure(self, engine, mock_paraview):
        """Test RAW file with configuration failure"""
        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000),
            patch("os.path.basename", return_value="bad.raw"),
            patch("os.path.splitext", return_value=("bad", ".raw")),
            patch.object(engine, "_configure_raw_reader", return_value=None),
        ):
            success, message, reader, name = engine.read_datafile("/test/bad.raw")

            assert success is False
            assert "Failed to configure RAW reader" in message

    def test_read_datafile_reader_none(self, engine, mock_paraview):
        """Test when reader creation returns None"""
        from paraview.simple import OpenDataFile

        OpenDataFile.return_value = None

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000),
            patch("os.path.basename", return_value="test.vtk"),
            patch("os.path.splitext", return_value=("test", ".vtk")),
        ):
            success, message, reader, name = engine.read_datafile("/test/test.vtk")

            assert success is False
            assert "Failed to create reader" in message

    def test_read_datafile_no_active_view(self, engine, mock_paraview):
        """Test when no active view available"""
        from paraview.simple import OpenDataFile, GetActiveView

        OpenDataFile.return_value = Mock()
        GetActiveView.return_value = None

        with (
            patch("os.path.exists", return_value=True),
            patch("os.path.getsize", return_value=1000),
            patch("os.path.basename", return_value="test.vtk"),
            patch("os.path.splitext", return_value=("test", ".vtk")),
        ):
            success, message, reader, name = engine.read_datafile("/test/test.vtk")

            assert success is False
            assert "No active ParaView view" in message

    def test_read_datafile_exception(self, engine, mock_paraview):
        """Test exception handling in read_datafile"""
        with patch("os.path.exists", side_effect=Exception("Test error")):
            success, message, reader, name = engine.read_datafile("/test/test.vtk")

            assert success is False
            assert "Error reading datafile" in message


class TestConfigureRawReader:
    """Test RAW file reader configuration"""

    def test_configure_raw_valid_dims_and_type(self, engine, mock_paraview):
        """Test configuring RAW reader with valid dimensions and datatype"""
        from paraview.simple import OpenDataFile

        mock_reader = Mock()
        OpenDataFile.return_value = mock_reader

        result = engine._configure_raw_reader(
            "/test/volume_256x256x256_uint8.raw", "volume_256x256x256_uint8.raw"
        )

        assert result == mock_reader
        assert mock_reader.DataExtent == [0, 255, 0, 255, 0, 255]
        assert mock_reader.FileDimensionality == 3
        assert mock_reader.DataScalarType == "unsigned char"

    def test_configure_raw_only_dims(self, engine, mock_paraview):
        """Test RAW config with dimensions but no datatype"""
        from paraview.simple import OpenDataFile

        mock_reader = Mock()
        OpenDataFile.return_value = mock_reader

        result = engine._configure_raw_reader(
            "/test/volume_128x128x128.raw", "volume_128x128x128.raw"
        )

        assert result == mock_reader
        assert mock_reader.DataExtent == [0, 127, 0, 127, 0, 127]
        assert mock_reader.DataScalarType == "unsigned char"  # Default

    def test_configure_raw_all_datatypes(self, engine, mock_paraview):
        """Test all supported datatypes"""
        from paraview.simple import OpenDataFile

        datatype_tests = {
            "uint8": "unsigned char",
            "uint16": "unsigned short",
            "int8": "char",
            "int16": "short",
            "float32": "float",
            "float64": "double",
        }

        for dtype, expected_vtk_type in datatype_tests.items():
            mock_reader = Mock()
            OpenDataFile.return_value = mock_reader

            _ = engine._configure_raw_reader(
                f"/test/vol_10x10x10_{dtype}.raw", f"vol_10x10x10_{dtype}.raw"
            )

            assert mock_reader.DataScalarType == expected_vtk_type

    def test_configure_raw_none_reader(self, engine, mock_paraview):
        """Test when OpenDataFile returns None"""
        from paraview.simple import OpenDataFile

        OpenDataFile.return_value = None

        result = engine._configure_raw_reader("/test/test.raw", "test.raw")
        assert result is None


class TestSaveContourAsStl:
    """Test STL saving functionality"""

    def test_save_stl_success(self, engine, mock_paraview):
        """Test successful STL save"""
        from paraview.simple import GetActiveSource, SaveData

        mock_source = Mock()
        GetActiveSource.return_value = mock_source
        engine._data_directory = "/test/dir"

        with patch("os.path.join", return_value="/test/dir/output.stl"):
            success, message, path = engine.save_contour_as_stl("output.stl")

            assert success is True
            assert "/test/dir/output.stl" in message
            SaveData.assert_called_once()

    def test_save_stl_no_active_source(self, engine, mock_paraview):
        """Test STL save with no active source"""
        from paraview.simple import GetActiveSource

        GetActiveSource.return_value = None

        success, message, path = engine.save_contour_as_stl()

        assert success is False
        assert "No active source" in message

    def test_save_stl_no_data_directory(self, engine, mock_paraview):
        """Test STL save with no data directory"""
        from paraview.simple import GetActiveSource

        GetActiveSource.return_value = Mock()
        engine._data_directory = ""

        success, message, path = engine.save_contour_as_stl()

        assert success is False
        assert "No data directory" in message

    def test_save_stl_exception(self, engine, mock_paraview):
        """Test STL save exception handling"""
        from paraview.simple import GetActiveSource, SaveData

        GetActiveSource.return_value = Mock()
        engine._data_directory = "/test"
        SaveData.side_effect = Exception("Save failed")

        with patch("os.path.join", return_value="/test/out.stl"):
            success, message, path = engine.save_contour_as_stl()

            assert success is False
            assert "Error saving STL" in message


class TestCreateSource:
    """Test geometric source creation"""

    @pytest.mark.parametrize(
        "source_type,expected_class",
        [
            ("sphere", "Sphere"),
            ("cone", "Cone"),
            ("cylinder", "Cylinder"),
            ("plane", "Plane"),
            ("box", "Box"),
        ],
    )
    def test_create_source_all_types(
        self, engine, mock_paraview, source_type, expected_class
    ):
        """Test creating all source types"""
        from paraview.simple import GetActiveView, Show

        mock_view = Mock()

        GetActiveView.return_value = mock_view
        Show.return_value = Mock()

        # Mock the specific source constructor directly at module level
        import paraview.simple

        setattr(paraview.simple, expected_class, Mock(return_value=Mock()))

        with patch.object(
            engine, "_get_source_name", return_value=f"{expected_class}1"
        ):
            success, message, source, name = engine.create_source(source_type)

            assert success is True
            assert expected_class.lower() in message.lower()
            assert source is not None
            assert name == f"{expected_class}1"

    def test_create_source_invalid_type(self, engine, mock_paraview):
        """Test creating invalid source type"""
        success, message, source, name = engine.create_source("invalid")

        assert success is False
        assert "Unsupported source type" in message
        assert source is None

    def test_create_source_exception(self, engine, mock_paraview):
        """Test source creation exception"""
        from paraview.simple import Sphere

        Sphere.side_effect = Exception("Creation failed")

        success, message, source, name = engine.create_source("sphere")

        assert success is False
        assert "Error creating source" in message


class TestSetActiveSource:
    """Test set_active_source method"""

    def test_set_active_source_exact_match(self, engine, mock_paraview):
        """Test setting active source with exact name match"""
        from paraview.simple import GetSources, SetActiveSource

        mock_proxy = Mock()
        GetSources.return_value = {("Contour1", ""): mock_proxy}

        success, message = engine.set_active_source("Contour1")

        assert success is True
        assert "Contour1" in message
        SetActiveSource.assert_called_once_with(mock_proxy)

    def test_set_active_source_not_found(self, engine, mock_paraview):
        """Test setting non-existent source"""
        from paraview.simple import GetSources

        GetSources.return_value = {("Other1", ""): Mock()}

        success, message = engine.set_active_source("NonExistent")

        assert success is False
        assert "No source found" in message

    def test_set_active_source_empty_pipeline(self, engine, mock_paraview):
        """Test with empty pipeline"""
        from paraview.simple import GetSources

        GetSources.return_value = {}

        success, message = engine.set_active_source("Any")

        assert success is False
        assert "No sources available" in message

    def test_set_active_source_exception(self, engine, mock_paraview):
        """Test exception handling"""
        from paraview.simple import GetSources

        GetSources.side_effect = Exception("Test error")

        success, message = engine.set_active_source("Test")

        assert success is False
        assert "Error setting active source" in message


class TestGetActiveSourceNamesByType:
    """Test get_active_source_names_by_type method"""

    def test_get_sources_all(self, engine, mock_paraview):
        """Test getting all sources"""
        from paraview.simple import GetSources

        mock_sphere = Mock()
        mock_sphere.__class__.__name__ = "Sphere"
        mock_cone = Mock()
        mock_cone.__class__.__name__ = "Cone"

        GetSources.return_value = {
            ("Sphere1", ""): mock_sphere,
            ("Cone1", ""): mock_cone,
        }

        success, message, names = engine.get_active_source_names_by_type(None)

        assert success is True
        assert len(names) == 2
        assert "Sphere1" in names
        assert "Cone1" in names

    def test_get_sources_filtered(self, engine, mock_paraview):
        """Test getting filtered sources"""
        from paraview.simple import GetSources

        mock_sphere = Mock()
        mock_sphere.__class__.__name__ = "Sphere"
        mock_cone = Mock()
        mock_cone.__class__.__name__ = "Cone"

        GetSources.return_value = {
            ("Sphere1", ""): mock_sphere,
            ("Cone1", ""): mock_cone,
        }

        success, message, names = engine.get_active_source_names_by_type("Sphere")

        assert success is True
        assert len(names) == 1
        assert "Sphere1" in names

    def test_get_sources_empty_pipeline(self, engine, mock_paraview):
        """Test with empty pipeline"""
        from paraview.simple import GetSources

        GetSources.return_value = {}

        success, message, names = engine.get_active_source_names_by_type()

        assert success is True
        assert names == []
        assert "No sources available" in message

    def test_get_sources_no_match(self, engine, mock_paraview):
        """Test when no sources match filter"""
        from paraview.simple import GetSources

        mock_sphere = Mock()
        mock_sphere.__class__.__name__ = "Sphere"

        GetSources.return_value = {("Sphere1", ""): mock_sphere}

        success, message, names = engine.get_active_source_names_by_type("Cone")

        assert success is True
        assert names == []
        assert "No sources of type 'Cone'" in message

    def test_get_sources_exception(self, engine, mock_paraview):
        """Test exception handling"""
        from paraview.simple import GetSources

        GetSources.side_effect = Exception("Test error")

        success, message, names = engine.get_active_source_names_by_type()

        assert success is False
        assert "Error getting source names" in message


# Continue in next file due to length...
