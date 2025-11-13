"""
ParaView MCP Server

This script runs as a standalone process and:
1. Connects to ParaView using its Python API over network
2. Exposes key ParaView functionality through the MCP protocol
3. Updates visualizations in the existing ParaView viewport

Usage:
1. Start pvserver with --multi-clients flag (e.g., pvserver --multi-clients --server-port=11111)
2. Start ParaView app and connect to the server
3. Configure Claude Desktop to use this script

"""
import os
import sys
import json
import logging
import argparse

from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add current directory to path for relative imports
sys.path.insert(0, os.path.dirname(__file__))

# Default prompt that instructs Claude how to interact with ParaView
default_prompt = """
When using ParaView through this interface, please follow these guidelines:

1. IMPORTANT: Only call strictly necessary ParaView functions per reply (and please limit the total number of call per reply). This ensures operations execute in a more interative manner and no excessive calls to related but non-essential functions. 

2. The only execute multiple repeated function call when given a target goal (e.g., identify a specific object), where different parameters need to used (e.g., isosurface with different isovalue). Avoid repeated calling of color map function unless user specific ask for color map design.

3. Paraview will be connect to mcp server on starup so no need to connect first.


"""
# Initialize MCP server
mcp: FastMCP = FastMCP("ParaView")

# ParaView manager will be initialized when needed
pv_manager = None
server_host = "localhost"
server_port = 11111

def get_pv_manager():
    """Lazy initialization of ParaView manager with server connection"""
    global pv_manager, server_host, server_port
    if pv_manager is None:
        try:
            from implementation.paraview_capabilities import VisualizationEngine
            pv_manager = VisualizationEngine(server_host, server_port)
            
            # Connect to the ParaView server
            logger.info(f"Connecting visualization engine to {server_host}:{server_port}")
            success = pv_manager.connect(server_host, server_port)
            if not success:
                logger.error(f"Failed to connect to ParaView server at {server_host}:{server_port}")
                logger.error("Make sure pvserver is running with: pvserver --multi-clients --server-port=11111")
                raise RuntimeError(f"Could not connect to ParaView server at {server_host}:{server_port}")
            else:
                logger.info("Successfully connected to ParaView server")
                
        except ImportError as e:
            logger.error(f"Failed to import ParaView: {e}")
            raise RuntimeError(f"ParaView not properly installed: {e}")
    return pv_manager

# ============================================================================
# MCP Tools for ParaView
# ============================================================================

@mcp.tool(
    name="load_scientific_data",
    description="Load scientific datasets from various file formats into ParaView for visualization and analysis. Supports VTK, EXODUS, CSV, RAW, BP5, and other scientific data formats. This enhanced function provides comprehensive file format detection and automatic configuration for optimal data loading.",
)
async def read_datafile_tool(file_path: str) -> str:
    """
    Read and load data from a file into ParaView with advanced format detection and error handling.
    
    This function provides robust data loading capabilities with:
    - Automatic file format detection based on file extension
    - Special handling for volume data formats (RAW, BP5/ADIOS2)
    - Comprehensive error reporting and troubleshooting guidance
    - Automatic camera positioning and display configuration
    
    Args:
        file_path (str): Absolute path to the data file. Supports multiple formats:
                        - VTK formats (.vtk, .vti, .vtr, .vts, .vtu, .vtp)
                        - EXODUS formats (.e, .exo, .exodus)
                        - CSV files (.csv)
                        - RAW volume files (.raw)
                        - ADIOS2/BP5 files (.bp, .bp5)
                        - Legacy formats and other scientific data formats
    
    Returns:
        str: Detailed status message including:
             - Success/failure status
             - Source registration name for pipeline operations
             - Error details with troubleshooting guidance if loading fails
             - File format detection information
    
    Raises:
        FileNotFoundError: If the specified file path does not exist
        UnsupportedFormatError: If the file format is not supported by ParaView
        
    Example:
        >>> read_datafile("/path/to/volume_data.vti")
        "Successfully loaded data from /path/to/volume_data.vti. Source registered as 'volume_data.vti'."
    """
    logger.info(f"Reading datafile from {file_path}")
    
    # Validate file path exists before attempting to load
    import os
    if not os.path.exists(file_path):
        return f"Error: File not found at path '{file_path}'. Please verify the file path is correct and the file exists."
    
    # Get file size for logging and diagnostics
    try:
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
    except OSError as e:
        logger.warning(f"Could not determine file size: {e}")
    
    # Attempt to load the data with comprehensive error handling
    success, message, _, source_name = get_pv_manager().read_datafile(file_path)
    
    if success:
        return f"{message}. Source registered as '{source_name}'. Use this name for pipeline operations."
    else:
        # Provide enhanced error information
        file_ext = os.path.splitext(file_path)[1].lower()
        error_guidance = ""
        
        if file_ext in ['.bp', '.bp5']:
            error_guidance = "\nFor ADIOS2/BP5 files: Ensure ADIOS2 is installed and ParaView has ADIOS2 support enabled."
        elif file_ext == '.raw':
            error_guidance = "\nFor RAW files: Ensure filename follows format 'name_XxYxZ_datatype.raw' (e.g., 'volume_256x256x256_uint8.raw')."
        elif file_ext in ['.vtk', '.vti', '.vtr', '.vts', '.vtu', '.vtp']:
            error_guidance = "\nFor VTK files: Check if the file is corrupted or uses an unsupported VTK version."
        
        return f"{message}{error_guidance}"

@mcp.tool()
def save_contour_as_stl(stl_filename: str = "contour.stl") -> str:
    """
    Save the currently active contour (or any surface/mesh source) as an STL file
    in the same folder as the originally loaded data.

    Args:
        stl_filename: The STL file name to use, defaults to 'contour.stl'.

    Returns:
        A status message (string).
    """
    success, message, path = pv_manager.save_contour_as_stl(stl_filename)
    return message

@mcp.tool(name="create_geometric_shape")
def create_source(source_type: str) -> str:
    """
    Create a new geometric source.
    
    Args:
        source_type: Type of source to create (Sphere, Cone, Cylinder, Plane, Box)
    
    Returns:
        Status message
    """
    success, message, _, source_name = pv_manager.create_source(source_type)
    if success:
        return f"{message}. Source registered as '{source_name}'."
    else:
        return message

@mcp.tool(name="generate_isosurface")
def create_isosurface(value: float, field: str = None) -> str:
    """
    Create an isosurface visualization of the active source.
    
    Args:
        value: Isovalue
        field: Optional field name to contour by
    
    Returns:
        Status message
    """
    success, message, contour_obj, contour_name = pv_manager.create_isosurface(value, field)
    if success:
        # Return a user-friendly message that also includes the name
        return f"{message}. Filter registered as '{contour_name}'."
    else:
        return message

@mcp.tool(name="create_data_slice")
def create_slice(origin_x: float = None, origin_y: float = None, origin_z: float = None,
                 normal_x: float = 0, normal_y: float = 0, normal_z: float = 1) -> str:
    """
    Create a slice through the loaded volume data.
    
    Args:
        origin_x, origin_y, origin_z: Coordinates for the slice plane's origin. If None,
            defaults to the data set's center.
        normal_x, normal_y, normal_z: Normal vector for the slice plane (default [0, 0, 1]).
    
    Returns:
        A string message containing success/failure details, plus the pipeline name.
    """
    success, message, slice_filter, slice_name = pv_manager.create_slice(
        origin_x,
        origin_y,
        origin_z,
        normal_x,
        normal_y,
        normal_z
    )

    # Return either an error message or a success message including the slice's name
    return message if success else f"Error creating slice: {message}"

@mcp.tool(name="configure_volume_display")
def toggle_volume_rendering(enable: bool = True) -> str:
    """
    Toggle the visibility of volume rendering for the active source.
    
    Args:
        enable (bool): Whether to show (True) or hide (False) volume rendering.
                      If True, shows volume rendering (switching to 'Volume' representation if needed).
                      If False, hides the volume but preserves the volume representation settings.
    
    Returns:
        Status message
    """
       
    success, message, source_name = pv_manager.create_volume_rendering(enable)
    if success:
        # Return a user-friendly message that also includes the name
        return f"{message}. Source registered as '{source_name}'."
    else:
        return message

@mcp.tool()
def toggle_visibility(enable: bool = True) -> str:
    """
    Toggle the visibility for the active source.
    
    Args:
        enable (bool): Whether to show (True) or hide (False) the active source.
                      If True, makes the active source visible.
                      If False, hides the active source but preserves the representation settings.
    
    Returns:
        Status message
    """
       
    success, message, source_name = pv_manager.toggle_visibility(enable)
    if success:
        # Return a user-friendly message that also includes the name
        return f"{message}. Source registered as '{source_name}'."
    else:
        return message


@mcp.tool()
def set_active_source(name: str) -> str:
    """
    Set the active pipeline object by its name.

    Usage:
      set_active_source("Contour1")

    Returns a status message.
    """
    success, message = pv_manager.set_active_source(name)
    return message

@mcp.tool()
def get_active_source_names_by_type(source_type: str = None) -> str:
    """
    Get a list of source names filtered by their type.

    Args:
        source_type (str, optional): Filter sources by type (e.g., 'Sphere', 'Contour', etc.).
                                  If None, returns all sources.

    Returns:
        A string message containing the source names or error message.
    """
    success, message, source_names = pv_manager.get_active_source_names_by_type(source_type)
    
    if success and source_names:
        sources_list = "\n- ".join(source_names)
        result = f"{message}:\n- {sources_list}"
        return result
    else:
        return message

# @mcp.tool()
# def edit_volume_opacity(field_name: str, opacity_points: list) -> str:
#     """
#     Edit ONLY the opacity transfer function for the specified field,
#     ensuring we pass only (value, alpha) pairs.

#     [Tips: only needed by volume rendering particularly finetuning the result, likely not needed when the color is ideal, usually the lower value should always have lower opacity]

#     Args:
#         field_name (str): The data array (field) name whose opacity we're adjusting.
#         opacity_points (list of [value, alpha] pairs):
#             Example: [[0.0, 0.0], [50.0, 0.3], [100.0, 1.0]]

#     Returns:
#         A status message (success or error)
#     """
#     success, message = pv_manager.edit_volume_opacity(field_name, opacity_points)
#     return message

# Compatible with OpenAI tool using
@mcp.tool()
def edit_volume_opacity(field_name: str, opacity_points: list[dict[str, float]]) -> str:
    """
    Edit ONLY the opacity transfer function for the specified field.

    Args:
        field_name (str): The scalar field to modify.
        opacity_points (list): A list of dicts like:
            [{"value": 0.0, "alpha": 0.0}, {"value": 50.0, "alpha": 0.3}]

    Returns:
        A status message (success or error)
    """
    formatted_points = [[pt["value"], pt["alpha"]] for pt in opacity_points]
    success, message = pv_manager.edit_volume_opacity(field_name, formatted_points)
    return message

# @mcp.tool()
# def set_color_map(field_name: str, color_points: list) -> str:
#     """
#     Sets the color transfer function for the specified field.

#     [Tips: only volume rendering should be using the set_color_map function, the lower values range corresponds to lower density objects, whereas higher values indicate high physical density. When design the color mapping try to assess the object of interest's density first from the default colormap (low value assigned to blue, high value assigned to red) and re-assign customized color accordingly, the order of the color may need to be adjust based on the rendering result. The more solid object should have higher density (!high value range). And a screen_shot should always be taken once this function is called to assess how to adjust the color_map again.]

#     Args:
#         field_name (str): The name of the field/array (as it appears in ParaView).
#         color_points (list of [value, [r, g, b]]):
#             e.g., [[0.0, [0.0, 0.0, 1.0]], [50.0, [0.0, 1.0, 0.0]], [100.0, [1.0, 0.0, 0.0]]]
#             Each element is (value, (r, g, b)) with r,g,b in [0,1].

#     Returns:
#         A status message as a string (e.g., success or error).
#     """
#     success, message = pv_manager.set_color_map(field_name, color_points)
#     return message

# Compatible with OpenAI tool using
@mcp.tool()
def set_color_map(field_name: str, color_points: list[dict]) -> str:
    """
    Sets the color transfer function for the specified field.

    [Tips: only volume rendering should be using the set_color_map function, the lower values range corresponds to lower density objects, whereas higher values indicate high physical density. When design the color mapping try to assess the object of interest's density first from the default colormap (low value assigned to blue, high value assigned to red) and re-assign customized color accordingly, the order of the color may need to be adjust based on the rendering result. The more solid object should have higher density (!high value range). And a screen_shot should always be taken once this function is called to assess how to adjust the color_map again.]

    Args:
        field_name (str): The name of the field/array (as it appears in ParaView).
        color_points (list of dicts): Each element should be a dict:
            {"value": float, "rgb": [r, g, b]} where r,g,b ∈ [0,1].

            Example:
            [
                {"value": 0.0, "rgb": [0.0, 0.0, 1.0]},
                {"value": 50.0, "rgb": [0.0, 1.0, 0.0]},
                {"value": 100.0, "rgb": [1.0, 0.0, 0.0]}
            ]

    Returns:
        A status message (success or error).
    """
    # Transform color_points to expected internal format: list[tuple[float, tuple[float, float, float]]]
    try:
        formatted_points = [(pt["value"], tuple(pt["rgb"])) for pt in color_points]
    except Exception as e:
        return f"Invalid format for color_points: {e}"

    success, message = pv_manager.set_color_map(field_name, formatted_points)
    return message


@mcp.tool(name="apply_field_coloring")
def color_by(field: str, component: int = -1) -> str:
    """
    Color the active visualization by a specific field.
    This function first checks if the active source can be colored by fields
    (i.e., it's a dataset with arrays) before attempting to apply colors.
    [tips] Volume rendering should not use this function 

    Args:
        field: Field name to color by
        component: Component to color by (-1 for magnitude)
    
    Returns:
        Status message
    """
    success, message = pv_manager.color_by(field, component)
    return message

@mcp.tool()
def compute_surface_area() -> str:
    """
    Compute the surface area of the currently active dataset.
    NOTE: Must be a surface mesh or 'Area' array won't exist.
    """
    success, message, area_value = pv_manager.compute_surface_area()
    return message

# @mcp.tool()
# def set_color_map_preset(preset_name: str) -> str:
#     """
#     Set the color map (lookup table) for the current visualization.
#     [tips: this should only be call at the beginning of the volume rendering]

#     Args:
#         preset_name: Name of the color map preset (e.g., "Rainbow", "Cool to Warm", "viridis")
    
#     Returns:
#         Status message
#     """
#     success, message = pv_manager.set_color_map(preset_name)
#     return message

@mcp.tool()
def set_representation_type(rep_type: str) -> str:
    """
    Set the representation type for the active source.
    
    [Tips: This function should not be used for volume rendering]

    Args:
        rep_type: Representation type (Surface, Wireframe, Points, etc.)
    
    Returns:
        Status message
    """
    success, message = pv_manager.set_representation_type(rep_type)
    return message

@mcp.tool()
def get_pipeline() -> str:
    """
    Get the current pipeline structure.
    
    Returns:
        Description of the current pipeline
    """
    success, message = pv_manager.get_pipeline()
    return message

@mcp.tool()
def get_available_arrays() -> str:
    """
    Get a list of available arrays in the active source.

    [tips: normally volume rendering would not require this information]
    
    Returns:
        List of available arrays
    """
    success, message = pv_manager.get_available_arrays()
    return message

@mcp.tool(name="generate_flow_streamlines")
def create_streamline(seed_point_number: int, vector_field: str = None,
                     integration_direction: str = "BOTH", max_steps: int = 1000,
                     initial_step: float = 0.1, maximum_step: float = 50.0) -> str:
    """
    Create streamlines from the loaded vector volume using the StreamTracer filter.
    This function automatically generates seed points based on the data bounds.
    
    Args:
        seed_point_number (int): The number of seed points to automatically generate.
        vector_field (str, optional): The name of the vector field to use for tracing. 
                                    If None, the first vector field will be chosen automatically.
        integration_direction (str): Integration direction ("FORWARD", "BACKWARD", or "BOTH"; default: "BOTH").
        max_steps (int): Maximum number of integration steps (default: 1000).
        initial_step (float): Initial integration step length (default: 0.1).
        maximum_step (float): Maximum streamline length (default: 50.0).
        
    Returns:
        str: Status message indicating whether the streamline was successfully created.
    """
    # Call the stream tracer creation method in your ParaViewManager
    success, message, streamline, tube_name = pv_manager.create_stream_tracer(
        vector_field=vector_field,
        base_source=None,  # Use the active source
        point_center=None,  # Auto-calculate the center
        integration_direction=integration_direction,
        initial_step_length=initial_step,
        maximum_stream_length=maximum_step,
        number_of_streamlines=seed_point_number
    )
    
    if success:
        return f"{message} Tube registered as '{tube_name}'."
    else:
        return message

@mcp.tool(
    name="take_viewport_screenshot",
    description="Capture a screenshot of the current ParaView viewport and save it to the current working directory. The screenshot will be displayed in chat and saved as a timestamped PNG file for reference.",
)
async def get_screenshot_tool() -> str:
    """
    Capture a screenshot of the current view and save it to the current working directory.
    This avoids preview window issues by saving the file directly.
    
    Returns:
        Image data and file path information
    """
    import os
    logger.info("Capturing ParaView viewport screenshot")
    
    # Get current working directory for user reference
    current_dir = os.getcwd()
    logger.info(f"Screenshot will be saved to: {current_dir}")
    
    success, message, img_path = get_pv_manager().get_screenshot()    

    if not success:
        logger.error(f"Screenshot capture failed: {message}")
        return f"❌ Screenshot failed: {message}"
    else:
        # Extract just the filename for display
        filename = os.path.basename(img_path)
        logger.info(f"Screenshot saved successfully: {filename}")
        
        # Return text information only to avoid preview window issues
        return f"✅ {message}\n📁 Saved in: {current_dir}\n📄 Filename: {filename}\n\nScreenshot captured and saved successfully! You can view the file directly from your file system."

@mcp.tool(
    name="show_screenshot_preview", 
    description="Capture screenshot with improved inline preview. Uses temporary files and cleanup to avoid window closing issues."
)
async def show_screenshot_preview() -> str:
    """
    Screenshot tool with improved preview handling.
    Creates a temporary copy that gets cleaned up to avoid file locking issues.
    """
    import os
    import shutil
    import tempfile
    import threading
    import time
    
    logger.info("Capturing ParaView viewport screenshot with improved preview")
    
    current_dir = os.getcwd()
    success, message, img_path = get_pv_manager().get_screenshot()    

    if not success:
        logger.error(f"Screenshot capture failed: {message}")
        return f"❌ Screenshot failed: {message}"
    else:
        filename = os.path.basename(img_path)
        logger.info(f"Screenshot saved: {filename}")
        
        # Create a temporary copy for preview that gets cleaned up quickly
        # This might help with window closing issues
        temp_preview_path = None
        try:
            # Create temp file for preview only
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_preview_path = temp_file.name
            
            # Copy the screenshot to temp location for preview
            shutil.copy2(img_path, temp_preview_path)
            
            # Schedule immediate cleanup after a short delay (30 seconds)
            def cleanup_preview():
                time.sleep(30)
                try:
                    if temp_preview_path and os.path.exists(temp_preview_path):
                        os.remove(temp_preview_path)
                        logger.debug(f"Cleaned up preview temp file: {temp_preview_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup preview temp: {e}")
            
            # Start cleanup thread
            threading.Thread(target=cleanup_preview, daemon=True).start()
            
            # Return the image with metadata that should allow proper dismissal
            result = f"📸 Screenshot Preview\n✅ Saved as: {filename}\n📁 Location: {current_dir}\n\n"
            result += str(Image(
                path=temp_preview_path,
                alt_text=f"ParaView screenshot preview - Original saved as {filename}"
            ))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create preview: {e}")
            # Fall back to text-only response
            return f"✅ Screenshot saved: {filename}\n📁 Location: {current_dir}\n⚠️ Preview failed: {e}"
    
@mcp.tool()
def rotate_camera(azimuth: float = 30.0, elevation: float = 0.0) -> str:
    """
    Rotate the camera by specified angles.
    
    Args:
        azimuth: Rotation around vertical axis in degrees
        elevation: Rotation around horizontal axis in degrees
    
    Returns:
        Status message
    """
    success, message = pv_manager.rotate_camera(azimuth, elevation)
    return message

@mcp.tool()
def reset_camera() -> str:
    """
    Reset the camera to show all data.
    
    Returns:
        Status message
    """
    success, message = pv_manager.reset_camera()
    return message

# @mcp.tool()
# def plot_over_line(point1: list = None, point2: list = None, resolution: int = 100) -> str:
#     """
#     Create a 'Plot Over Line' filter to sample data along a line between two points.

#     Args:
#         point1 (list, optional): The [x, y, z] coordinates of the start point. If None, will use data bounds.
#         point2 (list, optional): The [x, y, z] coordinates of the end point. If None, will use data bounds.
#         resolution (int, optional): Number of sample points along the line (default: 100).

#     Returns:
#         Status message
#     """
#     success, message, plot_filter = pv_manager.plot_over_line(point1, point2, resolution)
#     return message

# Compatible with OpenAI tool using
@mcp.tool()
def plot_over_line(point1: list[float] = None, point2: list[float] = None, resolution: int = 100) -> str:
    """
    Create a 'Plot Over Line' filter to sample data along a line between two points.

    Args:
        point1 (list of float): The [x, y, z] coordinates of the start point. If None, will use data bounds.
        point2 (list of float): The [x, y, z] coordinates of the end point. If None, will use data bounds.
        resolution (int): Number of sample points along the line (default: 100).

    Returns:
        Status message
    """
    success, message, plot_filter = pv_manager.plot_over_line(point1, point2, resolution)
    return message

@mcp.tool()
def warp_by_vector(vector_field: str = None, scale_factor: float = 1.0) -> str:
    """
    Apply the 'Warp By Vector' filter to the active source.

    Args:
        vector_field (str, optional): The name of the vector field to use for warping. If None, the first available vector field will be used.
        scale_factor (float, optional): The scale factor for the warp (default: 1.0).

    Returns:
        Status message
    """
    success, message, warp_filter = pv_manager.warp_by_vector(vector_field, scale_factor)
    return message

@mcp.tool()
def list_commands() -> str:
    """
    List all available commands in this ParaView MCP server.
    
    Returns:
        List of available commands
    """
    commands = [
        "load_scientific_data: Load scientific datasets from various file formats (VTK, EXODUS, CSV, RAW, BP5, etc.)",
        "create_geometric_shape: Create geometric sources (Sphere, Cone, etc.)",
        "generate_isosurface: Create isosurface visualizations",
        "create_data_slice: Create slices through volume data",
        "configure_volume_display: Enable or disable volume rendering",
	    "toggle_visibility: Enable or disable visibility for the active source",
        "set_active_source: Set the active pipeline object by name",
        "get_active_source_names_by_type: Get a list of sources filtered by type",
        "apply_field_coloring: Color the visualization by a field",
        # "set_color_map_preset: Set the color map preset",
        "set_representation_type: Set the representation type (Surface, Wireframe, etc.)",
        "edit_volume_opacity: Edit the opacity transfer function",
        "get_pipeline: Get the current pipeline structure",
        "get_available_arrays: Get available data arrays",
        "generate_flow_streamlines: Create streamline visualizations",
        "compute_surface_area: Compute the surface area of the active surface",
        "save_contour_as_stl: Save the active surface as STL",
        "take_viewport_screenshot: Capture screenshot and save to current directory (recommended)",
        "show_screenshot_preview: Capture screenshot with improved inline preview (fixed closing issues)",
        "rotate_camera: Rotate the camera view",
        "reset_camera: Reset the camera to show all data",
        "plot_line: Plot a line through the data",
        "warp_by_vector: Warp the active source by a vector field",
    ]
    
    return "Available ParaView commands:\n\n" + "\n".join(commands)


def main():
    """
    Main entry point for the ParaView MCP server.
    Supports both stdio and SSE transports based on environment variables.
    """
    # Handle 'help' command (without dashes) by converting to --help
    if len(sys.argv) > 1 and sys.argv[1] == "help":
        sys.argv[1] = "--help"

    parser = argparse.ArgumentParser(
        description="ParaView MCP Server - Scientific visualization server with comprehensive ParaView capabilities",
        prog="paraview-mcp",
    )
    parser.add_argument("--version", action="version", version="ParaView MCP Server v1.0.0")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport type to use (default: stdio)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for SSE transport (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for SSE transport (default: 8000)"
    )
    parser.add_argument("--server", type=str, default="localhost", help="ParaView server hostname (default: localhost)")
    parser.add_argument("--pv-port", type=int, default=11111, help="ParaView server port (default: 11111)")
    parser.add_argument("--paraview_package_path", type=str, help="Path to the ParaView Python package", default=None)

    args = parser.parse_args()

    try:
        logger.info("Starting ParaView MCP Server")

        # Add the ParaView package path to sys.path
        if args.paraview_package_path:
            sys.path.append(args.paraview_package_path)
        
        # Set ParaView server connection parameters
        global server_host, server_port
        server_host = args.server
        server_port = args.pv_port
        logger.info(f"ParaView server configured: {server_host}:{server_port}")
        
        # Note: ParaView connection will be established when first tool is called

        # Use command-line args or environment variables
        transport = args.transport or os.getenv("MCP_TRANSPORT", "stdio").lower()

        if transport == "sse":
            # SSE transport for web-based clients
            host = args.host or os.getenv("MCP_SSE_HOST", "0.0.0.0")
            port = args.port or int(os.getenv("MCP_SSE_PORT", "8000"))
            logger.info(f"Starting SSE transport on {host}:{port}")
            print(
                json.dumps({"message": f"Starting SSE on {host}:{port}"}),
                file=sys.stderr,
            )
            mcp.run(transport="sse", host=host, port=port)
        else:
            # Default stdio transport
            logger.info("Starting stdio transport")
            print(json.dumps({"message": "Starting stdio transport"}), file=sys.stderr)
            mcp.run(transport="stdio")

    except Exception as e:
        logger.error(f"Server error: {e}")
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
