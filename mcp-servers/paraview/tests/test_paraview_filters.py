"""
Tests for ParaView filter operations (isosurface, slice, streamlines, etc.)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock


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


class TestCreateIsosurface:
    """Test isosurface creation"""

    def test_create_isosurface_new(self, engine, mock_paraview):
        """Test creating new isosurface"""
        from paraview.simple import Contour, Show, GetActiveView

        mock_contour = Mock()
        mock_view = Mock()

        Contour.return_value = mock_contour
        GetActiveView.return_value = mock_view
        Show.return_value = Mock()

        with patch.object(engine, "_get_source_name", return_value="Contour1"):
            success, message, contour, name = engine.create_isosurface(50.0)

            assert success is True
            assert "Created isosurface" in message
            assert mock_contour.Isosurfaces == [50.0]
            assert name == "Contour1"

    def test_create_isosurface_with_field(self, engine, mock_paraview):
        """Test creating isosurface with specific field"""
        from paraview.simple import Contour, Show, GetActiveView

        mock_contour = Mock()
        Contour.return_value = mock_contour
        GetActiveView.return_value = Mock()
        Show.return_value = Mock()

        with patch.object(engine, "_get_source_name", return_value="Contour1"):
            success, message, contour, name = engine.create_isosurface(
                50.0, field="density"
            )

            assert success is True
            assert mock_contour.ContourBy == ["POINTS", "density"]

    def test_create_isosurface_update_existing(self, engine, mock_paraview):
        """Test updating existing isosurface"""
        from paraview.simple import Show, GetActiveView

        mock_contour = Mock()
        engine.isosurface_filter = mock_contour
        GetActiveView.return_value = Mock()
        Show.return_value = Mock()

        with patch.object(engine, "_get_source_name", return_value="Contour1"):
            success, message, contour, name = engine.create_isosurface(75.0)

            assert success is True
            assert "Updated isosurface" in message
            assert mock_contour.Isosurfaces == [75.0]

    def test_create_isosurface_no_source(self, engine, mock_paraview):
        """Test isosurface with no active source"""
        from paraview.simple import GetActiveSource

        engine.primary_data_source = None
        GetActiveSource.return_value = None

        success, message, contour, name = engine.create_isosurface(50.0)

        assert success is False
        assert "No active source" in message

    def test_create_isosurface_exception(self, engine, mock_paraview):
        """Test isosurface exception handling"""
        from paraview.simple import Contour

        Contour.side_effect = Exception("Contour failed")

        success, message, contour, name = engine.create_isosurface(50.0)

        assert success is False
        assert "Error creating/updating isosurface" in message


class TestComputeSurfaceArea:
    """Test surface area computation"""

    def test_compute_surface_area_success(self, engine, mock_paraview):
        """Test successful surface area computation"""
        from paraview.simple import GetActiveSource, IntegrateVariables
        import paraview.servermanager as sm

        mock_source = Mock()
        mock_integrate = Mock()
        mock_data = Mock()
        mock_area_array = Mock()

        GetActiveSource.return_value = mock_source
        IntegrateVariables.return_value = mock_integrate
        sm.Fetch.return_value = mock_data
        mock_data.GetCellData.return_value.GetArray.return_value = mock_area_array
        mock_area_array.GetValue.return_value = 123.45

        success, message, area = engine.compute_surface_area()

        assert success is True
        assert area == 123.45
        assert "123.45" in message

    def test_compute_surface_area_no_source(self, engine, mock_paraview):
        """Test with no active source"""
        from paraview.simple import GetActiveSource

        GetActiveSource.return_value = None

        success, message, area = engine.compute_surface_area()

        assert success is False
        assert "No active source" in message
        assert area == 0.0

    def test_compute_surface_area_no_data(self, engine, mock_paraview):
        """Test with no integrated data"""
        from paraview.simple import GetActiveSource, IntegrateVariables
        import paraview.servermanager as sm

        GetActiveSource.return_value = Mock()
        IntegrateVariables.return_value = Mock()
        sm.Fetch.return_value = None

        success, message, area = engine.compute_surface_area()

        assert success is False
        assert "Could not fetch" in message

    def test_compute_surface_area_no_area_array(self, engine, mock_paraview):
        """Test with no Area array"""
        from paraview.simple import GetActiveSource, IntegrateVariables
        import paraview.servermanager as sm

        mock_data = Mock()
        GetActiveSource.return_value = Mock()
        IntegrateVariables.return_value = Mock()
        sm.Fetch.return_value = mock_data
        mock_data.GetCellData.return_value.GetArray.return_value = None

        success, message, area = engine.compute_surface_area()

        assert success is False
        assert "No 'Area' array" in message

    def test_compute_surface_area_exception(self, engine, mock_paraview):
        """Test exception handling"""
        from paraview.simple import GetActiveSource

        GetActiveSource.side_effect = Exception("Test error")

        success, message, area = engine.compute_surface_area()

        assert success is False
        assert "Error computing surface area" in message


class TestCreateSlice:
    """Test slice creation"""

    def test_create_slice_with_origin(self, engine, mock_paraview):
        """Test creating slice with specified origin"""
        from paraview.simple import Slice, Show, GetActiveView

        # Create a mock slice type object
        mock_slice_type_obj = Mock()
        mock_slice = Mock()

        # Make SliceType always return the mock object when accessed
        type(mock_slice).SliceType = PropertyMock(return_value=mock_slice_type_obj)

        mock_view = Mock()

        Slice.return_value = mock_slice
        GetActiveView.return_value = mock_view
        Show.return_value = Mock()

        with patch.object(engine, "_get_source_name", return_value="Slice1"):
            success, message, slice_filter, name = engine.create_slice(
                1.0, 2.0, 3.0, 0, 0, 1
            )

            assert success is True
            assert "Created slice" in message
            assert name == "Slice1"
            # Check that Origin and Normal were set correctly
            assert mock_slice_type_obj.Origin == [1.0, 2.0, 3.0]
            assert mock_slice_type_obj.Normal == [0, 0, 1]

    def test_create_slice_auto_origin(self, engine, mock_paraview):
        """Test creating slice with auto-calculated origin"""
        from paraview.simple import Slice, Show, GetActiveView

        # Create a mock slice type object
        mock_slice_type_obj = Mock()
        mock_slice = Mock()

        # Make SliceType always return the mock object when accessed
        type(mock_slice).SliceType = PropertyMock(return_value=mock_slice_type_obj)

        mock_info = Mock()
        mock_info.GetBounds.return_value = [0, 10, 0, 20, 0, 30]

        engine.primary_data_source.GetDataInformation.return_value = mock_info
        Slice.return_value = mock_slice
        GetActiveView.return_value = Mock()
        Show.return_value = Mock()

        with patch.object(engine, "_get_source_name", return_value="Slice1"):
            success, message, slice_filter, name = engine.create_slice()

            assert success is True
            # Origin should be center: [(0+10)/2, (0+20)/2, (0+30)/2] = [5, 10, 15]
            assert mock_slice_type_obj.Origin == [5.0, 10.0, 15.0]

    def test_create_slice_no_source(self, engine, mock_paraview):
        """Test slice with no source"""
        from paraview.simple import GetActiveSource

        engine.primary_data_source = None
        GetActiveSource.return_value = None

        success, message, slice_filter, name = engine.create_slice()

        assert success is False
        assert "No active source" in message

    def test_create_slice_exception(self, engine, mock_paraview):
        """Test slice exception handling"""
        from paraview.simple import Slice

        Slice.side_effect = Exception("Slice failed")

        success, message, slice_filter, name = engine.create_slice(1, 2, 3)

        assert success is False
        assert "Error creating slice" in message


class TestStreamTracer:
    """Test stream tracer creation"""

    def test_create_stream_tracer_auto_field(self, engine, mock_paraview):
        """Test stream tracer with auto-detected vector field"""
        from paraview.simple import (
            StreamTracer,
            Show,
            Tube,
            ColorBy,
            GetDisplayProperties,
        )

        mock_tracer = Mock()
        mock_tube = Mock()
        mock_info = Mock()
        mock_point_info = Mock()
        mock_array_info = Mock()

        # Mock vector field detection
        mock_array_info.GetNumberOfComponents.return_value = 3
        mock_array_info.GetName.return_value = "velocity"
        mock_point_info.GetNumberOfArrays.return_value = 1
        mock_point_info.GetArrayInformation.return_value = mock_array_info
        mock_info.GetPointDataInformation.return_value = mock_point_info
        mock_info.GetBounds.return_value = [0, 10, 0, 10, 0, 10]

        engine.primary_data_source.GetDataInformation.return_value = mock_info
        StreamTracer.return_value = mock_tracer
        Tube.return_value = mock_tube
        Show.return_value = Mock()
        ColorBy.return_value = None
        GetDisplayProperties.return_value = Mock()

        with patch.object(engine, "_get_source_name", return_value="Tube1"):
            success, message, tube, name = engine.create_stream_tracer(
                number_of_streamlines=100
            )

            assert success is True
            assert "'velocity'" in message
            assert name == "Tube1"

    def test_create_stream_tracer_custom_field(self, engine, mock_paraview):
        """Test stream tracer with custom vector field"""
        from paraview.simple import (
            StreamTracer,
            Tube,
            Show,
            ColorBy,
            GetDisplayProperties,
        )

        mock_info = Mock()
        mock_info.GetBounds.return_value = [0, 10, 0, 10, 0, 10]

        engine.primary_data_source.GetDataInformation.return_value = mock_info
        StreamTracer.return_value = Mock()
        Tube.return_value = Mock()
        Show.return_value = Mock()
        ColorBy.return_value = None
        GetDisplayProperties.return_value = Mock()

        with patch.object(engine, "_get_source_name", return_value="Tube1"):
            success, message, tube, name = engine.create_stream_tracer(
                number_of_streamlines=50,
                vector_field="custom_velocity",
                integration_direction="FORWARD",
            )

            assert success is True
            assert "'custom_velocity'" in message

    def test_create_stream_tracer_no_source(self, engine, mock_paraview):
        """Test stream tracer with no source"""
        from paraview.simple import GetActiveSource

        engine.primary_data_source = None
        GetActiveSource.return_value = None

        success, message, tube, name = engine.create_stream_tracer(
            number_of_streamlines=10
        )

        assert success is False
        assert "No active source" in message

    def test_create_stream_tracer_no_vector_fields(self, engine, mock_paraview):
        """Test when no vector fields available"""

        mock_info = Mock()
        mock_point_info = Mock()
        mock_point_info.GetNumberOfArrays.return_value = 0

        mock_info.GetPointDataInformation.return_value = mock_point_info
        engine.primary_data_source.GetDataInformation.return_value = mock_info

        success, message, tube, name = engine.create_stream_tracer(
            number_of_streamlines=10
        )

        assert success is False
        assert "No arrays found" in message

    def test_create_stream_tracer_single_component_fallback(
        self, engine, mock_paraview
    ):
        """Test fallback to single component array"""
        from paraview.simple import (
            StreamTracer,
            Tube,
            Show,
            ColorBy,
            GetDisplayProperties,
        )

        mock_info = Mock()
        mock_point_info = Mock()
        mock_array_info = Mock()

        # Only single-component array available
        mock_array_info.GetNumberOfComponents.return_value = 1
        mock_array_info.GetName.return_value = "pressure"
        mock_point_info.GetNumberOfArrays.return_value = 1
        mock_point_info.GetArrayInformation.return_value = mock_array_info
        mock_info.GetPointDataInformation.return_value = mock_point_info
        mock_info.GetBounds.return_value = [0, 10, 0, 10, 0, 10]

        engine.primary_data_source.GetDataInformation.return_value = mock_info
        StreamTracer.return_value = Mock()
        Tube.return_value = Mock()
        Show.return_value = Mock()
        ColorBy.return_value = None
        GetDisplayProperties.return_value = Mock()

        with patch.object(engine, "_get_source_name", return_value="Tube1"):
            success, message, tube, name = engine.create_stream_tracer(
                number_of_streamlines=10
            )

            assert success is True
            assert "'pressure'" in message

    def test_create_stream_tracer_exception(self, engine, mock_paraview):
        """Test exception handling"""

        mock_info = Mock()
        mock_info.GetPointDataInformation.side_effect = Exception("Test error")

        engine.primary_data_source.GetDataInformation.return_value = mock_info

        success, message, tube, name = engine.create_stream_tracer(10)

        assert success is False
        assert "Error creating stream tracer" in message


class TestPlotOverLine:
    """Test plot over line filter"""

    def test_plot_over_line_custom_points(self, engine, mock_paraview):
        """Test plot over line with custom points"""
        from paraview.simple import (
            GetActiveSource,
            PlotOverLine,
            Show,
            GetActiveView,
            CreateView,
            AssignViewToLayout,
        )

        mock_source = Mock()
        mock_plot = Mock()
        mock_plot.Point1 = [0, 0, 0]
        mock_plot.Point2 = [10, 10, 10]

        GetActiveSource.return_value = mock_source
        PlotOverLine.return_value = mock_plot
        GetActiveView.return_value = Mock()
        CreateView.return_value = Mock()
        Show.return_value = Mock()
        AssignViewToLayout.return_value = None

        success, message, plot = engine.plot_over_line([0, 0, 0], [10, 10, 10], 200)

        assert success is True
        assert "Plot over line created" in message
        assert mock_plot.Resolution == 200

    def test_plot_over_line_auto_points(self, engine, mock_paraview):
        """Test plot over line with auto points"""
        from paraview.simple import (
            GetActiveSource,
            PlotOverLine,
            Show,
            GetActiveView,
            CreateView,
            AssignViewToLayout,
        )

        mock_plot = Mock()
        mock_plot.Point1 = [0, 0, 0]
        mock_plot.Point2 = [1, 1, 1]

        GetActiveSource.return_value = Mock()
        PlotOverLine.return_value = mock_plot
        GetActiveView.return_value = Mock()
        CreateView.return_value = Mock()
        Show.return_value = Mock()
        AssignViewToLayout.return_value = None

        success, message, plot = engine.plot_over_line()

        assert success is True

    def test_plot_over_line_no_source(self, engine, mock_paraview):
        """Test with no active source"""
        from paraview.simple import GetActiveSource

        GetActiveSource.return_value = None

        success, message, plot = engine.plot_over_line()

        assert success is False
        assert "No active source" in message

    def test_plot_over_line_exception(self, engine, mock_paraview):
        """Test exception handling"""
        from paraview.simple import GetActiveSource, PlotOverLine

        GetActiveSource.return_value = Mock()
        PlotOverLine.side_effect = Exception("Plot failed")

        success, message, plot = engine.plot_over_line()

        assert success is False
        assert "Error creating plot over line" in message


class TestWarpByVector:
    """Test warp by vector filter"""

    def test_warp_by_vector_auto_field(self, engine, mock_paraview):
        """Test warp with auto-detected vector field"""
        from paraview.simple import GetActiveSource, WarpByVector, Show, GetActiveView

        mock_source = Mock()
        mock_warp = Mock()
        mock_info = Mock()
        mock_point_info = Mock()
        mock_array_info = Mock()

        # Mock multi-component array
        mock_array_info.GetNumberOfComponents.return_value = 3
        mock_array_info.GetName.return_value = "displacement"
        mock_point_info.GetNumberOfArrays.return_value = 1
        mock_point_info.GetArrayInformation.return_value = mock_array_info

        mock_info.GetPointDataInformation.return_value = mock_point_info
        mock_source.GetDataInformation.return_value = mock_info

        GetActiveSource.return_value = mock_source
        WarpByVector.return_value = mock_warp
        GetActiveView.return_value = Mock()
        Show.return_value = Mock()

        success, message, warp = engine.warp_by_vector()

        assert success is True
        assert "'displacement'" in message
        assert mock_warp.Vectors == ["POINTS", "displacement"]

    def test_warp_by_vector_custom_field(self, engine, mock_paraview):
        """Test warp with custom vector field"""
        from paraview.simple import GetActiveSource, WarpByVector, Show, GetActiveView

        mock_warp = Mock()

        GetActiveSource.return_value = Mock()
        WarpByVector.return_value = mock_warp
        GetActiveView.return_value = Mock()
        Show.return_value = Mock()

        success, message, warp = engine.warp_by_vector("custom_vector", 2.5)

        assert success is True
        assert "custom_vector" in message
        assert mock_warp.ScaleFactor == 2.5

    def test_warp_by_vector_no_source(self, engine, mock_paraview):
        """Test with no active source"""
        from paraview.simple import GetActiveSource

        GetActiveSource.return_value = None

        success, message, warp = engine.warp_by_vector()

        assert success is False
        assert "No active source" in message

    def test_warp_by_vector_no_vector_field(self, engine, mock_paraview):
        """Test when no vector field found"""
        from paraview.simple import GetActiveSource

        mock_source = Mock()
        mock_info = Mock()
        mock_point_info = Mock()
        mock_array_info = Mock()

        # Only single component arrays
        mock_array_info.GetNumberOfComponents.return_value = 1
        mock_point_info.GetNumberOfArrays.return_value = 1
        mock_point_info.GetArrayInformation.return_value = mock_array_info
        mock_info.GetPointDataInformation.return_value = mock_point_info
        mock_source.GetDataInformation.return_value = mock_info

        GetActiveSource.return_value = mock_source

        success, message, warp = engine.warp_by_vector()

        assert success is False
        assert "No vector field found" in message

    def test_warp_by_vector_exception(self, engine, mock_paraview):
        """Test exception handling"""
        from paraview.simple import GetActiveSource, WarpByVector

        GetActiveSource.return_value = Mock()
        WarpByVector.side_effect = Exception("Warp failed")

        success, message, warp = engine.warp_by_vector("test_field")

        assert success is False
        assert "Error creating warp by vector" in message
