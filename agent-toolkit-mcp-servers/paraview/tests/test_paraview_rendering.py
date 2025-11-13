"""
Tests for ParaView rendering and display operations
"""

import pytest
from unittest.mock import Mock, MagicMock, patch


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
    """Create VisualizationEngine instance"""
    from paraview_mcp.implementation.paraview_capabilities import VisualizationEngine

    engine = VisualizationEngine()
    engine.primary_data_source = Mock()
    return engine


class TestVolumeRendering:
    """Test volume rendering operations"""

    def test_create_volume_rendering_enable(self, engine, mock_paraview):
        """Test enabling volume rendering"""
        from paraview.simple import GetActiveView, GetDisplayProperties

        mock_display = Mock()
        mock_display.GetRepresentationType.return_value = "Surface"

        GetActiveView.return_value = Mock()
        GetDisplayProperties.return_value = mock_display

        with patch.object(engine, "_get_source_name", return_value="TestSource"):
            success, message, name = engine.create_volume_rendering(enable=True)

            assert success is True
            assert "enabled" in message.lower()
            assert mock_display.Visibility == 1

    def test_create_volume_rendering_disable(self, engine, mock_paraview):
        """Test disabling volume rendering"""
        from paraview.simple import GetActiveView, GetDisplayProperties

        mock_display = Mock()
        mock_display.GetRepresentationType.return_value = "Volume"

        GetActiveView.return_value = Mock()
        GetDisplayProperties.return_value = mock_display

        with patch.object(engine, "_get_source_name", return_value="TestSource"):
            success, message, name = engine.create_volume_rendering(enable=False)

            assert success is True
            assert "hidden" in message.lower()
            assert mock_display.Visibility == 0

    def test_create_volume_rendering_no_source(self, engine, mock_paraview):
        """Test volume rendering with no source"""
        engine.primary_data_source = None

        success, message, name = engine.create_volume_rendering()

        assert success is False
        assert "No original data" in message

    def test_create_volume_rendering_exception(self, engine, mock_paraview):
        """Test exception handling"""
        from paraview.simple import SetActiveSource

        SetActiveSource.side_effect = Exception("Test error")

        success, message, name = engine.create_volume_rendering()

        assert success is False
        assert "Error toggling volume rendering" in message


class TestToggleVisibility:
    """Test visibility toggle"""

    def test_toggle_visibility_enable(self, engine, mock_paraview):
        """Test enabling visibility"""
        from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties

        mock_source = Mock()
        mock_display = Mock()
        mock_display.Visibility = 0  # Initialize to 0

        GetActiveSource.return_value = mock_source
        GetActiveView.return_value = Mock()
        GetDisplayProperties.return_value = mock_display

        with patch.object(engine, "_get_source_name", return_value="TestSource"):
            success, message, name = engine.toggle_visibility(enable=True)

            assert success is True
            assert "visible" in message.lower()
            assert mock_display.Visibility == 1

    def test_toggle_visibility_disable(self, engine, mock_paraview):
        """Test disabling visibility"""
        from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties

        mock_source = Mock()
        mock_display = Mock()
        mock_display.Visibility = 1  # Initialize to 1

        GetActiveSource.return_value = mock_source
        GetActiveView.return_value = Mock()
        GetDisplayProperties.return_value = mock_display

        with patch.object(engine, "_get_source_name", return_value="TestSource"):
            success, message, name = engine.toggle_visibility(enable=False)

            assert success is True
            assert "hidden" in message.lower()
            assert mock_display.Visibility == 0

    def test_toggle_visibility_no_source(self, engine, mock_paraview):
        """Test with no active source"""
        from paraview.simple import GetActiveSource

        GetActiveSource.return_value = None

        success, message, name = engine.toggle_visibility()

        assert success is False
        assert "No data selected" in message

    def test_toggle_visibility_exception(self, engine, mock_paraview):
        """Test exception handling"""
        from paraview.simple import GetActiveSource

        GetActiveSource.side_effect = Exception("Test error")

        success, message, name = engine.toggle_visibility()

        assert success is False
        assert "Error toggling visibility" in message


class TestColorBy:
    """Test color by field functionality"""

    def test_color_by_point_data(self, engine, mock_paraview):
        """Test coloring by point data field"""
        from paraview.simple import (
            GetActiveSource,
            GetActiveView,
            GetDisplayProperties,
        )

        mock_source = Mock()
        mock_display = Mock()
        mock_display.GetRepresentationType.return_value = "Surface"

        # Mock data info
        mock_data_info = Mock()
        mock_point_info = Mock()
        mock_cell_info = Mock()
        mock_array_info = Mock()

        mock_array_info.GetName.return_value = "temperature"
        mock_point_info.GetNumberOfArrays.return_value = 1
        mock_point_info.GetArrayInformation.return_value = mock_array_info
        mock_cell_info.GetNumberOfArrays.return_value = 0

        mock_data_info.GetPointDataInformation.return_value = mock_point_info
        mock_data_info.GetCellDataInformation.return_value = mock_cell_info
        mock_source.GetDataInformation.return_value = mock_data_info

        GetActiveSource.return_value = mock_source
        GetActiveView.return_value = Mock()
        GetDisplayProperties.return_value = mock_display

        success, message = engine.color_by("temperature")

        assert success is True
        assert "'temperature'" in message
        assert "POINTS" in message

    def test_color_by_cell_data(self, engine, mock_paraview):
        """Test coloring by cell data field"""
        from paraview.simple import (
            GetActiveSource,
            GetActiveView,
            GetDisplayProperties,
        )

        mock_source = Mock()
        mock_display = Mock()
        mock_display.GetRepresentationType.return_value = "Surface"

        # Mock data info - field in cell data only
        mock_data_info = Mock()
        mock_point_info = Mock()
        mock_cell_info = Mock()
        mock_point_array = Mock()
        mock_cell_array = Mock()

        mock_point_array.GetName.return_value = "other_field"
        mock_cell_array.GetName.return_value = "pressure"

        mock_point_info.GetNumberOfArrays.return_value = 1
        mock_point_info.GetArrayInformation.return_value = mock_point_array
        mock_cell_info.GetNumberOfArrays.return_value = 1
        mock_cell_info.GetArrayInformation.return_value = mock_cell_array

        mock_data_info.GetPointDataInformation.return_value = mock_point_info
        mock_data_info.GetCellDataInformation.return_value = mock_cell_info
        mock_source.GetDataInformation.return_value = mock_data_info

        GetActiveSource.return_value = mock_source
        GetActiveView.return_value = Mock()
        GetDisplayProperties.return_value = mock_display

        success, message = engine.color_by("pressure")

        assert success is True
        assert "'pressure'" in message
        assert "CELLS" in message

    def test_color_by_field_not_found(self, engine, mock_paraview):
        """Test coloring by non-existent field"""
        from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties

        mock_source = Mock()
        mock_display = Mock()
        mock_display.GetRepresentationType.return_value = "Surface"

        mock_data_info = Mock()
        mock_point_info = Mock()
        mock_cell_info = Mock()
        mock_array_info = Mock()

        mock_array_info.GetName.return_value = "other_field"
        mock_point_info.GetNumberOfArrays.return_value = 1
        mock_point_info.GetArrayInformation.return_value = mock_array_info
        mock_cell_info.GetNumberOfArrays.return_value = 0

        mock_data_info.GetPointDataInformation.return_value = mock_point_info
        mock_data_info.GetCellDataInformation.return_value = mock_cell_info
        mock_source.GetDataInformation.return_value = mock_data_info

        GetActiveSource.return_value = mock_source
        GetActiveView.return_value = Mock()
        GetDisplayProperties.return_value = mock_display

        success, message = engine.color_by("nonexistent")

        assert success is False
        assert "Field 'nonexistent' not found" in message
        assert "other_field" in message

    def test_color_by_no_source(self, engine, mock_paraview):
        """Test with no active source"""
        from paraview.simple import GetActiveSource

        GetActiveSource.return_value = None

        success, message = engine.color_by("test")

        assert success is False
        assert "No active source" in message

    def test_color_by_no_arrays(self, engine, mock_paraview):
        """Test with no data arrays"""
        from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties

        mock_source = Mock()
        mock_display = Mock()
        mock_display.GetRepresentationType.return_value = "Surface"

        mock_data_info = Mock()
        mock_point_info = Mock()
        mock_cell_info = Mock()

        mock_point_info.GetNumberOfArrays.return_value = 0
        mock_cell_info.GetNumberOfArrays.return_value = 0

        mock_data_info.GetPointDataInformation.return_value = mock_point_info
        mock_data_info.GetCellDataInformation.return_value = mock_cell_info
        mock_source.GetDataInformation.return_value = mock_data_info

        GetActiveSource.return_value = mock_source
        GetActiveView.return_value = Mock()
        GetDisplayProperties.return_value = mock_display

        success, message = engine.color_by("test")

        assert success is False
        assert "does not have any data arrays" in message

    def test_color_by_invalid_representation(self, engine, mock_paraview):
        """Test with incompatible representation type"""
        from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties

        mock_source = Mock()
        mock_display = Mock()
        mock_display.GetRepresentationType.return_value = "Outline"

        GetActiveSource.return_value = mock_source
        GetActiveView.return_value = Mock()
        GetDisplayProperties.return_value = mock_display

        success, message = engine.color_by("test")

        assert success is False
        assert "cannot be colored by fields" in message

    def test_color_by_exception(self, engine, mock_paraview):
        """Test exception handling"""
        from paraview.simple import GetActiveSource

        GetActiveSource.side_effect = Exception("Test error")

        success, message = engine.color_by("test")

        assert success is False
        assert "Error coloring by field" in message


class TestSetColorMapPreset:
    """Test color map preset setting"""

    def test_set_color_map_preset_success(self, engine, mock_paraview):
        """Test setting color map preset successfully"""
        from paraview.simple import (
            GetActiveSource,
            GetActiveView,
            GetDisplayProperties,
            ApplyPreset,
        )

        mock_source = Mock()
        mock_display = Mock()
        mock_color_tf = Mock()

        mock_display.LookupTable = mock_color_tf

        GetActiveSource.return_value = mock_source
        GetActiveView.return_value = Mock()
        GetDisplayProperties.return_value = mock_display

        # Call the preset version by passing only preset_name
        success, message = engine.set_color_map_preset("Viridis")

        assert success is True
        assert "Viridis" in message
        ApplyPreset.assert_called_once_with(mock_color_tf, "Viridis", True)

    def test_set_color_map_preset_no_source(self, engine, mock_paraview):
        """Test with no active source"""
        from paraview.simple import GetActiveSource

        GetActiveSource.return_value = None

        success, message = engine.set_color_map_preset("Viridis")

        assert success is False
        assert "No active source" in message

    def test_set_color_map_preset_no_lookup_table(self, engine, mock_paraview):
        """Test with no lookup table"""
        from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties

        mock_display = Mock()
        mock_display.LookupTable = None

        GetActiveSource.return_value = Mock()
        GetActiveView.return_value = Mock()
        GetDisplayProperties.return_value = mock_display

        success, message = engine.set_color_map_preset("Viridis")

        assert success is False
        assert "No active color transfer function" in message

    def test_set_color_map_preset_exception(self, engine, mock_paraview):
        """Test exception handling"""
        from paraview.simple import GetActiveSource

        GetActiveSource.side_effect = Exception("Test error")

        success, message = engine.set_color_map_preset("Viridis")

        assert success is False
        assert "Test error" in message


class TestSetRepresentationType:
    """Test set representation type"""

    def test_set_representation_type_success(self, engine, mock_paraview):
        """Test setting representation type successfully"""
        from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties

        mock_source = Mock()
        mock_display = Mock()

        GetActiveSource.return_value = mock_source
        GetActiveView.return_value = Mock()
        GetDisplayProperties.return_value = mock_display

        success, message = engine.set_representation_type("Wireframe")

        assert success is True
        assert "Wireframe" in message
        mock_display.SetRepresentationType.assert_called_once_with("Wireframe")

    def test_set_representation_type_no_source(self, engine, mock_paraview):
        """Test with no active source"""
        from paraview.simple import GetActiveSource

        GetActiveSource.return_value = None

        success, message = engine.set_representation_type("Surface")

        assert success is False
        assert "No active source" in message

    def test_set_representation_type_exception(self, engine, mock_paraview):
        """Test exception handling"""
        from paraview.simple import GetActiveSource

        GetActiveSource.side_effect = Exception("Test error")

        success, message = engine.set_representation_type("Points")

        assert success is False
        assert "Error setting representation type" in message


class TestEditVolumeOpacity:
    """Test opacity transfer function editing"""

    def test_edit_volume_opacity_success(self, engine, mock_paraview):
        """Test editing opacity successfully"""
        from paraview.simple import GetOpacityTransferFunction

        mock_opacity_tf = Mock()
        GetOpacityTransferFunction.return_value = mock_opacity_tf

        opacity_points = [(0.0, 0.0), (50.0, 0.3), (100.0, 1.0)]
        success, message = engine.edit_volume_opacity("density", opacity_points)

        assert success is True
        assert "density" in message
        # Check that Points was set with flattened values
        expected_points = [
            0.0,
            0.0,
            0.5,
            0.0,
            50.0,
            0.3,
            0.5,
            0.0,
            100.0,
            1.0,
            0.5,
            0.0,
        ]
        assert mock_opacity_tf.Points == expected_points

    def test_edit_volume_opacity_empty_points(self, engine, mock_paraview):
        """Test with empty opacity points"""
        success, message = engine.edit_volume_opacity("density", [])

        assert success is False
        assert "No opacity points" in message

    def test_edit_volume_opacity_no_transfer_function(self, engine, mock_paraview):
        """Test when transfer function not found"""
        from paraview.simple import GetOpacityTransferFunction

        GetOpacityTransferFunction.return_value = None

        success, message = engine.edit_volume_opacity("density", [(0, 0)])

        assert success is False
        assert "Could not find" in message

    def test_edit_volume_opacity_exception(self, engine, mock_paraview):
        """Test exception handling"""
        from paraview.simple import GetOpacityTransferFunction

        GetOpacityTransferFunction.side_effect = Exception("Test error")

        success, message = engine.edit_volume_opacity("test", [(0, 0)])

        assert success is False
        assert "Error editing opacity" in message


class TestSetColorMapTransfer:
    """Test color transfer function setting"""

    def test_set_color_map_transfer_success(self, engine, mock_paraview):
        """Test setting color transfer function successfully"""
        from paraview.simple import GetColorTransferFunction

        mock_color_tf = Mock()
        GetColorTransferFunction.return_value = mock_color_tf

        color_points = [
            (0.0, (0.0, 0.0, 1.0)),
            (50.0, (0.0, 1.0, 0.0)),
            (100.0, (1.0, 0.0, 0.0)),
        ]
        success, message = engine.set_color_map("density", color_points)

        assert success is True
        assert "density" in message
        expected_rgb = [0.0, 0.0, 0.0, 1.0, 50.0, 0.0, 1.0, 0.0, 100.0, 1.0, 0.0, 0.0]
        assert mock_color_tf.RGBPoints == expected_rgb

    def test_set_color_map_transfer_empty_points(self, engine, mock_paraview):
        """Test with empty color points"""
        success, message = engine.set_color_map("density", [])

        assert success is False
        assert "No color points" in message

    def test_set_color_map_transfer_invalid_rgb(self, engine, mock_paraview):
        """Test with invalid RGB values"""
        from paraview.simple import GetColorTransferFunction

        GetColorTransferFunction.return_value = Mock()

        color_points = [(0.0, (0.0, 1.0))]  # Only 2 values instead of 3
        success, message = engine.set_color_map("density", color_points)

        assert success is False
        assert "Invalid RGB" in message

    def test_set_color_map_transfer_no_transfer_function(self, engine, mock_paraview):
        """Test when transfer function not found"""
        from paraview.simple import GetColorTransferFunction

        GetColorTransferFunction.return_value = None

        success, message = engine.set_color_map("density", [(0, (0, 0, 0))])

        assert success is False
        assert "Could not find" in message

    def test_set_color_map_transfer_exception(self, engine, mock_paraview):
        """Test exception handling"""
        from paraview.simple import GetColorTransferFunction

        GetColorTransferFunction.side_effect = Exception("Test error")

        success, message = engine.set_color_map("test", [(0, (0, 0, 0))])

        assert success is False
        assert "Error setting color map" in message
