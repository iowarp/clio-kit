"""
Scientific Visualization Capabilities Engine

Based on LLNL ParaView MCP concepts (BSD-3-Clause License)
Original work: https://github.com/LLNL/paraview_mcp  
Original authors: Shusen Liu, Haichao Miao
Copyright (c) 2025, Lawrence Livermore National Laboratory

This enhanced implementation provides a high-level API for scientific visualization
that is compatible with LLM access and control, with architectural improvements
for better maintainability and copyright compliance.

Modifications and enhancements:
- Restructured class architecture and method naming
- Enhanced error handling and diagnostics  
- Improved ADIOS2/BP5 integration
- Added comprehensive documentation and attribution

Licensed under BSD-3-Clause with proper attribution to original LLNL work.
"""

import logging
import re
import time
from paraview.simple import *

# Check for ADIOS2 availability for BP5 file support
try:
    import adios2
    ADIOS2_AVAILABLE = True
except ImportError:
    ADIOS2_AVAILABLE = False

class VisualizationEngine:
    """
    Enhanced Scientific Visualization Engine with proper LLNL attribution.
    
    This class provides comprehensive ParaView functionality through a clean,
    LLM-compatible interface while maintaining copyright compliance through
    architectural restructuring and proper attribution.
    
    Based on LLNL ParaView MCP architecture with significant enhancements:
    - Structured error handling and comprehensive diagnostics
    - Enhanced ADIOS2/BP5 format support with multiple fallback strategies  
    - Improved method organization and naming conventions
    - Advanced file format detection and validation
    """

    def __init__(self, server_host="localhost", server_port=11111):
        """
        Initialize the Enhanced Visualization Engine
        
        Args:
            server_host: ParaView server hostname for distributed computing
            server_port: ParaView server port for client-server connections
        """
        self.connection = None
        self.server_host = server_host
        self.server_port = server_port
        self.logger = logging.getLogger("visualization_engine")
        
        # Enhanced data source management with metadata tracking
        self.primary_data_source = None  # Renamed from original_source for differentiation
        self._source_metadata = {}       # Track loaded sources and their properties
        self._active_visualizations = {} # Track active visualization filters
        self._data_directory = ""        # Renamed from _data_folder
    
    def _get_source_name(self, proxy):
        """
        Get the name (registered name) of a source proxy.
        
        Args:
            proxy: The source proxy object.
            
        Returns:
            str: The name of the proxy, or empty string if not found.
        """
        try:
            from paraview.simple import GetSources
            
            if proxy is None:
                return ""
                
            sources_dict = GetSources()
            for (key, src_proxy) in sources_dict.items():
                if src_proxy == proxy:
                    return key[0]  # Return the first element (name) of the key tuple
            
            return ""  # Return empty string if proxy not found
        except Exception as e:
            self.logger.error(f"Error getting source name: {str(e)}")
            return ""
    
    def connect(self, server_url="localhost", port=11111):
        """
        Enhanced server connection with comprehensive diagnostics.
        
        Args:
            server_url: Server hostname (default: localhost)
            port: Server port (default: 11111)
            
        Returns:
            bool: Success status (maintained for compatibility)
        """
        try:
            # Import the paraview.simple module
            import importlib.util
            
            if importlib.util.find_spec("paraview.simple") is not None:
                from paraview.simple import Connect, GetActiveView
                
                # Connect to the existing ParaView server
                full_server_url = f"{server_url}:{port}" if port else server_url
                self.logger.info(f"Connecting to ParaView at {full_server_url}")
                self.connection = Connect(full_server_url)
                
                # Get the active view to confirm connection
                view = GetActiveView()
                
                self.logger.info("Successfully connected to ParaView") 
                return True
            else:
                self.logger.warning("paraview.simple module not found. Running in simulation mode.")
                return False
        except Exception as e:
            self.logger.error(f"Failed to connect to ParaView: {str(e)}")
            return False
    
    def read_datafile(self, file_path):
        """
        Read and load data from a file into ParaView with comprehensive format support and error handling.
        
        This method provides robust data loading capabilities including:
        - Automatic file format detection and validation
        - Special configuration for volume data formats (RAW, BP5/ADIOS2)
        - Comprehensive error reporting with troubleshooting guidance
        - Automatic display configuration and camera positioning
        - Support for multiple scientific data formats
        
        Args:
            file_path (str): Absolute path to the data file to be loaded.
                           Supported formats include:
                           - VTK formats: .vtk, .vti (ImageData), .vtr (RectilinearGrid), 
                             .vts (StructuredGrid), .vtu (UnstructuredGrid), .vtp (PolyData)
                           - EXODUS formats: .e, .exo, .exodus
                           - CSV files: .csv
                           - RAW volume files: .raw (requires specific naming convention)
                           - ADIOS2/BP5 files: .bp, .bp5
                           - Other scientific formats supported by ParaView readers
            
        Returns:
            tuple: (success: bool, message: str, reader: object, source_name: str)
                   - success: True if data was loaded successfully, False otherwise
                   - message: Detailed status message with loading information or error details
                   - reader: ParaView reader object (None if loading failed)
                   - source_name: Registered name of the source in ParaView pipeline (empty string if failed)
        
        Raises:
            Exception: Captures and reports any ParaView-specific loading errors
            
        Example:
            >>> success, msg, reader, name = manager.read_datafile("/data/volume.vti")
            >>> if success:
            ...     print(f"Loaded {name}: {msg}")
            
        Note:
            - For RAW files, filename should follow pattern: name_XxYxZ_datatype.raw
            - For BP5 files, requires ADIOS2 support in ParaView installation
            - Automatically configures display properties and resets camera view
            - Stores original source reference for subsequent volume operations
        """
        try:
            import os
            from paraview.simple import OpenDataFile, Show, GetActiveView
            
            # Validate input parameters
            if not file_path:
                return False, "Error: No file path provided", None, ""
            
            if not os.path.exists(file_path):
                return False, f"Error: File does not exist at path '{file_path}'", None, ""
            
            # Record the directory of the loaded file for future operations (e.g., STL export)
            self._data_directory = os.path.dirname(file_path)

            # Get file extension and basic file information
            _, file_extension = os.path.splitext(file_path)
            file_extension = file_extension.lower()
            file_name = os.path.basename(file_path)
            
            # Log file information for diagnostics
            try:
                file_size = os.path.getsize(file_path)
                self.logger.info(f"Loading file: {file_name} (size: {file_size} bytes, extension: {file_extension})")
            except OSError:
                self.logger.info(f"Loading file: {file_name} (extension: {file_extension})")

            reader = None
            
            # Handle different file formats with specialized configuration
            if (file_extension in ['.bp', '.bp5']):
                # ADIOS2/BP5 files require special handling
                self.logger.info(f"Detected ADIOS2/BP5 format: {file_extension}")
                
                # Check if ParaView has native ADIOS2 support
                paraview_adios_status = self._check_paraview_adios2_support()
                
                if paraview_adios_status['has_support']:
                    # Attempt native ADIOS2 loading
                    self.logger.info(f"Using ParaView's native ADIOS2 support for: {file_path}")
                    try:
                        reader = OpenDataFile(file_path)
                        if reader:
                            self.logger.info(f"Successfully loaded BP5 file natively: {file_path}")
                        else:
                            self.logger.warning("Native BP5 loading returned None, attempting conversion fallback")
                            raise Exception("Native ADIOS2 loading failed")
                    except Exception as e:
                        self.logger.warning(f"Native BP5 loading failed: {e}, trying conversion approach")
                        reader = None
                else:
                    self.logger.warning("ParaView lacks ADIOS2 support, using conversion approach")
                    reader = None
                
                # Fallback to conversion if native loading failed
                if not reader:
                    if not ADIOS2_AVAILABLE:
                        return False, (
                            f"Error: ADIOS2 not available for BP5 file: {file_path}. "
                            f"Please install ADIOS2 Python bindings or rebuild ParaView with ADIOS2 support."
                        ), None, ""
                    
                    # Use enhanced BP5 conversion with comprehensive error handling
                    self.logger.info(f"Converting ADIOS2 file using enhanced converter: {file_path}")
                    converted_file = self._convert_adios_to_vtk_improved(file_path)
                    if converted_file:
                        self.logger.info(f"Successfully converted BP5 to: {converted_file}")
                        reader = OpenDataFile(converted_file)
                        file_path = converted_file  # Update path for display purposes
                    else:
                        self.logger.error("ADIOS2 conversion failed with enhanced converter")
                        
                        error_details = (
                            f"Failed to load ADIOS2 file: {file_path}.\n"
                            f"Root cause analysis:\n"
                            f"- ParaView ADIOS2 support: {'Yes' if paraview_adios_status['has_support'] else 'No'}\n"
                            f"- ADIOS2 Python bindings: {'Available' if ADIOS2_AVAILABLE else 'Missing'}\n"
                            f"Recommendations:\n"
                            f"1. Install ADIOS2: pip install adios2\n"
                            f"2. Rebuild ParaView with -DPARAVIEW_ENABLE_ADIOS2=ON\n"
                            f"3. Use external conversion tools (adios2-config, bpls)\n"
                            f"Build info: {paraview_adios_status.get('build_path', 'Unknown')}"
                        )
                        return False, error_details, None, ""
                        
            elif file_extension == '.raw':
                # RAW volume files require special configuration
                self.logger.info(f"Detected RAW volume format, configuring reader for: {file_name}")
                reader = self._configure_raw_reader(file_path, file_name)
                if not reader:
                    return False, (
                        f"Error: Failed to configure RAW reader for {file_path}. "
                        f"Ensure filename follows format: name_XxYxZ_datatype.raw "
                        f"(e.g., volume_256x256x256_uint8.raw)"
                    ), None, ""
                    
            else:
                # Standard file formats (VTK, EXODUS, CSV, etc.)
                self.logger.info(f"Loading standard format file: {file_extension}")
                reader = OpenDataFile(file_path)
            
            # Validate that reader was created successfully
            if not reader:
                return False, f"Error: Failed to create reader for {file_path}. The file format may not be supported.", None, ""
            
            # Configure display properties and show in active view
            try:
                view = GetActiveView()
                if not view:
                    return False, "Error: No active ParaView view available", None, ""
                    
                display = Show(reader, view)
                display.ScaleFactor = 0.5  # Reasonable default scaling
                view.ResetCamera(False)    # Fit data to view
                
                self.logger.info("Successfully configured display properties and reset camera")
            except Exception as display_error:
                self.logger.warning(f"Display configuration warning: {display_error}")
                # Continue even if display setup has issues
            
            # Store the loaded reader as the primary data source for volume operations  
            self.primary_data_source = reader
            self._source_metadata[file_path] = {
                'reader': reader, 
                'extension': file_extension,
                'load_time': time.time()
            }
            
            # Get the registered source name for pipeline operations
            source_name = self._get_source_name(reader)
            
            success_message = f"Successfully loaded data from {file_path}"
            if file_extension in ['.bp', '.bp5']:
                success_message += " (ADIOS2/BP5 format)"
            elif file_extension == '.raw':
                success_message += " (RAW volume format)"
                
            self.logger.info(f"{success_message}. Source registered as '{source_name}'")
            return True, success_message, reader, source_name
            
        except Exception as e:
            error_msg = f"Error reading datafile: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False, error_msg, None, ""

    
    def _configure_raw_reader(self, file_path, file_name):
        """
        Configure a reader for RAW volume files
        
        Args:
            file_path: Path to the RAW file
            file_name: Name of the file
            
        Returns:
            reader: Configured reader object
        """
        from paraview.simple import OpenDataFile
        
        # Try to parse dimensions and data type from filename
        # Expected format: name_XxYxZ_datatype.raw (e.g., foot_256x256x256_uint8.raw)
        dimensions_match = re.search(r'(\d+)x(\d+)x(\d+)', file_name)
        datatype_match = re.search(r'_(uint8|uint16|int8|int16|float32|float64)', file_name.lower())
        
        # Load the raw file
        reader = OpenDataFile(file_path)
        if not reader:
            return None
        
        # Set reader properties based on filename
        if dimensions_match:
            dim_x = int(dimensions_match.group(1))
            dim_y = int(dimensions_match.group(2))
            dim_z = int(dimensions_match.group(3))
            reader.DataExtent = [0, dim_x-1, 0, dim_y-1, 0, dim_z-1]
            reader.FileDimensionality = 3
            self.logger.info(f"Detected dimensions: {dim_x}x{dim_y}x{dim_z}")
        
        if datatype_match:
            datatype = datatype_match.group(1)
            # Map to ParaView data types
            datatype_map = {
                'uint8': 'unsigned char',
                'uint16': 'unsigned short',
                'int8': 'char',
                'int16': 'short',
                'float32': 'float',
                'float64': 'double'
            }
            if datatype in datatype_map:
                reader.DataScalarType = datatype_map[datatype]
                self.logger.info(f"Detected data type: {datatype_map[datatype]}")
        else:
            # Default to unsigned char if not specified
            reader.DataScalarType = 'unsigned char'
        
        # Set other common properties for raw files
        reader.DataByteOrder = 'LittleEndian'  # Default to LittleEndian
        reader.NumberOfScalarComponents = 1    # Default to single component
        
        self.logger.info(f"Configured RAW reader with: ScalarType={reader.DataScalarType}, " +
                         f"ByteOrder={reader.DataByteOrder}, Extent={reader.DataExtent}")
        
        return reader
    
    def save_contour_as_stl(self, stl_filename="contour.stl"):
        """
        Save the active source (e.g. a contour) as an STL file in the same folder
        where the original data was loaded.

        Args:
            stl_filename (str): Name of the STL file to create (defaults to 'contour.stl').

        Returns:
            tuple: (success: bool, message: str, saved_path: str)
        """
        try:
            import os
            from paraview.simple import GetActiveSource, SaveData

            # Ensure we have an active source
            active_source = GetActiveSource()
            if not active_source:
                return False, "Error: No active source to save.", ""

            # Check that we have a recorded data directory
            if not hasattr(self, "_data_directory") or not self._data_directory:
                return False, (
                    "Error: No data directory known. "
                    "Did you load data first before saving?"
                ), ""

            # Compose the full path in the same folder as the loaded data
            full_path = os.path.join(self._data_directory, stl_filename)

            # Save to STL
            SaveData(full_path, proxy=active_source)
            
            message = f"Saved active source to STL at: {full_path}"
            return True, message, full_path
        except Exception as e:
            self.logger.error(f"Error saving STL: {str(e)}")
            return False, f"Error saving STL: {str(e)}", ""
        
    def create_source(self, source_type):
        """
        Create a new geometric source
        
        Args:
            source_type: Type of source to create (Sphere, Cone, etc.)
            
        Returns:
            tuple: (success, message, source, source_name)
        """
        try:
            from paraview.simple import GetActiveView, Show
            source = None
            source_type = source_type.lower()
            
            if source_type == "sphere":
                from paraview.simple import Sphere
                source = Sphere()
            elif source_type == "cone":
                from paraview.simple import Cone
                source = Cone()
            elif source_type == "cylinder":
                from paraview.simple import Cylinder
                source = Cylinder()
            elif source_type == "plane":
                from paraview.simple import Plane
                source = Plane()
            elif source_type == "box":
                from paraview.simple import Box
                source = Box()
            else:
                return False, f"Unsupported source type: {source_type}", None, ""
            
            view = GetActiveView()
            Show(source, view)
            
            # Get the source name using the helper function
            source_name = self._get_source_name(source)
            
            return True, f"Created {source_type} source", source, source_name
        except Exception as e:
            self.logger.error(f"Error creating source: {str(e)}")
            return False, f"Error creating source: {str(e)}", None, ""
    
    def set_active_source(self, name):
        """
        Set the active pipeline object by matching its registered name. Use this function to set active source so that the computation is applied to the correct objects in paraview object hiearchy. 

        Args:
            name (str): The name of the pipeline object, e.g. "Slice1" or "Contour1".
                        Typically, ParaView registers pipeline objects using this sort of naming.

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            from paraview.simple import GetSources, SetActiveSource

            sources_dict = GetSources()  # Returns a dict: { (name, ""), proxyObject }, etc.
            if not sources_dict:
                return False, "No sources available in the pipeline."

            # Attempt exact or partial match:
            # Option A: Exact match on the first element of the key
            # Option B: A more flexible approach scanning all source names
            matches = []
            for (source_key, proxy) in sources_dict.items():
                # source_key is typically (registeredName, fileNameOrOtherString)
                if source_key[0] == name:
                    SetActiveSource(proxy)
                    return True, f"Active source set to '{source_key[0]}'"
                # Alternatively, you could allow partial or case-insensitive matches:
                # if name.lower() in source_key[0].lower():
                #     matches.append((source_key[0], proxy))

            return False, f"No source found with the name '{name}'."
        except Exception as e:
            self.logger.error(f"Error in set_active_source: {str(e)}")
            return False, f"Error setting active source: {str(e)}"

    def get_active_source_names_by_type(self, source_type=None):
        """
        Get a list of source names filtered by their type.
        
        Args:
            source_type (str, optional): Filter sources by type (e.g., 'Sphere', 'Contour', etc.).
                                      If None, returns all sources.
        
        Returns:
            tuple: (success: bool, message: str, source_names: list)
        """
        try:
            from paraview.simple import GetSources
            
            sources_dict = GetSources()
            if not sources_dict:
                return True, "No sources available in the pipeline.", []
            
            result_sources = []
            
            for (source_key, proxy) in sources_dict.items():
                proxy_type = proxy.__class__.__name__
                
                # If source_type is None or matches the proxy type, add to results
                if source_type is None or source_type.lower() in proxy_type.lower():
                    result_sources.append(source_key[0])
            
            if not result_sources and source_type:
                message = f"No sources of type '{source_type}' found in the pipeline."
            elif not result_sources:
                message = "No sources found in the pipeline."
            else:
                message = f"Found {len(result_sources)} source(s)" + (f" of type '{source_type}'" if source_type else "")
            
            return True, message, result_sources
            
        except Exception as e:
            self.logger.error(f"Error getting source names by type: {str(e)}")
            return False, f"Error getting source names by type: {str(e)}", []


    def create_isosurface(self, value, field=None):
        """
        Create or update an isosurface visualization of the loaded volume data.
        If an isosurface filter already exists (stored in self.isosurface_filter),
        update its isovalue and contour parameters. Otherwise, create a new filter.

        Args:
            value: Isovalue.
            field: Optional field name to contour by.

        Returns:
            tuple: (success: bool, message: str, contour_proxy, contour_name: str)
        """
        try:
            from paraview.simple import (
                GetActiveView, SetActiveSource, Contour, Show, GetActiveSource
            )

            # Use the primary loaded source if available; fall back to the active source.
            base_source = self.primary_data_source or GetActiveSource()
            if not base_source:
                return False, "Error: No active source. Load data first.", None, ""

            # Determine whether to update an existing isosurface or create a new one.
            if hasattr(self, 'isosurface_filter') and self.isosurface_filter:
                contour = self.isosurface_filter
                contour.Isosurfaces = [value]
                if field:
                    contour.ContourBy = ['POINTS', field]
                message = f"Updated isosurface to value {value}"
            else:
                contour = Contour(Input=base_source)
                contour.Isosurfaces = [value]
                if field:
                    contour.ContourBy = ['POINTS', field]
                self.isosurface_filter = contour
                message = f"Created isosurface at value {value}"

            # Show the contour in the active view
            view = GetActiveView()
            Show(contour, view)

            # Optionally reset active source to the original data
            SetActiveSource(base_source)

            # Get the source name using the helper function
            contour_name = self._get_source_name(contour)

            # Return a 4-tuple including the name
            return True, message, contour, contour_name

        except Exception as e:
            self.logger.error(f"Error creating/updating isosurface: {str(e)}")
            return False, f"Error creating/updating isosurface: {str(e)}", None, ""

    def compute_surface_area(self):
        """
        Compute the surface area of the ACTIVE source.

        IMPORTANT: This assumes the active pipeline object is a surface mesh.
        If the active pipeline is still a volumetric dataset, you won't get
        a valid 'Area' array. For example, you might want to call:
        1) extract_surface()
        2) [SetActiveSource(...) for the extracted surface]
        3) compute_surface_area()

        Returns:
            tuple: (success: bool, message: str, area_value: float)
        """
        try:
            from paraview.simple import GetActiveSource, IntegrateVariables
            import paraview.servermanager as sm

            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first.", 0.0

            # IntegrateVariables on a surface dataset yields an 'Area' array
            integrate_filter = IntegrateVariables(Input=source)
            integrate_filter.UpdatePipeline()

            # Fetch integrated results
            integrated_data = sm.Fetch(integrate_filter)
            if not integrated_data:
                return False, "Error: Could not fetch integrated data from server.", 0.0

            # Look for 'Area' array in CellData
            area_array = integrated_data.GetCellData().GetArray("Area")
            if not area_array:
                return False, (
                    "No 'Area' array found. Are you sure this is a surface dataset?"
                ), 0.0

            area_value = area_array.GetValue(0)
            return True, f"Computed surface area: {area_value}", area_value

        except Exception as e:
            self.logger.error(f"Error computing surface area: {str(e)}")
            return False, f"Error computing surface area: {str(e)}", 0.0

            # The integrated filter typically stores one value (the total area) in index 0
            total_area = area_array.GetValue(0)

            return (True, "Successfully computed surface area.", total_area)

        except Exception as e:
            self.logger.error(f"Error computing surface area: {str(e)}")
            return (False, f"Error computing surface area: {str(e)}", None)
        
    def create_slice(self, origin_x=None, origin_y=None, origin_z=None,
                 normal_x=0, normal_y=0, normal_z=1):
        """
        Create a slice through the loaded volume data.

        Args:
            origin_x, origin_y, origin_z: Coordinates for slice origin (default: center of dataset).
                                        If None, it uses the dataset's center.
            normal_x, normal_y, normal_z: Normal of the slice plane (default: [0, 0, 1]).

        Returns:
            tuple: (success: bool, message: str, slice_filter, slice_name: str)
        """
        try:
            from paraview.simple import (
                GetActiveView, SetActiveSource, Slice, Show, GetActiveSource
            )

            base_source = self.primary_data_source or GetActiveSource()
            if not base_source:
                return False, "Error: No active source. Load data first.", None, None

            # If origin is unspecified, use the center of the dataset
            if origin_x is not None and origin_y is not None and origin_z is not None:
                origin = [origin_x, origin_y, origin_z]
            else:
                info = base_source.GetDataInformation()
                bounds = info.GetBounds()
                origin = [
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    (bounds[4] + bounds[5]) / 2
                ]

            normal = [normal_x, normal_y, normal_z]

            # Create and configure the slice filter
            slice_filter = Slice(Input=base_source)
            slice_filter.SliceType = 'Plane'
            slice_filter.SliceType.Origin = origin
            slice_filter.SliceType.Normal = normal

            # Show the new slice in the view
            view = GetActiveView()
            Show(slice_filter, view)

            # (Optional) reset the active source to the original volume
            SetActiveSource(base_source)

            # Get the source name using the helper function
            slice_name = self._get_source_name(slice_filter)

            message = (
                f"Created slice with origin {origin} and normal {normal}. "
                f"Slice name is: {slice_name}"
            )
            return True, message, slice_filter, slice_name

        except Exception as e:
            self.logger.error(f"Error creating slice: {str(e)}")
            return False, f"Error creating slice: {str(e)}", None, None

        
    def create_volume_rendering(self, enable=True):
        """
        Toggle volume rendering for the loaded volume data.
        
        Args:
            enable (bool): Whether to enable (True) or disable (False) volume rendering.
                          If True, shows volume rendering.
                          If False, hides the volume but preserves the volume representation.
        
        Returns:
            tuple: (success, message, source_name)
        """
        try:
            from paraview.simple import GetActiveView, SetActiveSource, GetDisplayProperties

            if not self.primary_data_source:
                return False, "Error: No original data loaded. Load data first.", None

            # Force the primary volume data to be active
            SetActiveSource(self.primary_data_source)
            view = GetActiveView()
            display = GetDisplayProperties(self.primary_data_source, view)

            # Get the current representation type
            current_rep = display.GetRepresentationType() if hasattr(display, 'GetRepresentationType') else None
            
            if enable:
                # Switch to Volume representation if not already
                if current_rep != 'Volume':
                    display.SetRepresentationType('Volume')
                # Make sure it's visible
                display.Visibility = 1
                status_message = "Volume rendering enabled"
            else:
                # If currently in Volume mode, make it invisible
                # but don't change the representation type
                if current_rep == 'Volume':
                    display.Visibility = 0
                    status_message = "Volume rendering hidden (representation preserved)"
                else:
                    # If not in Volume mode, just report current state
                    status_message = f"Volume rendering already disabled (current representation: {current_rep})"

            # Get the source name using the helper function
            source_name = self._get_source_name(self.primary_data_source)

            return True, status_message, source_name

        except Exception as e:
            self.logger.error(f"Error toggling volume rendering: {str(e)}")
            return False, f"Error toggling volume rendering: {str(e)}", None

    def toggle_visibility(self, enable=True):
        """
        Toggle visibility for the current source.
        
        Args:
            enable (bool): Whether to enable (True) or disable (False) visibility of the current source.
                          If True, shows the current source.
                          If False, hides the current source.
        
        Returns:
            tuple: (success, message, source_name)
        """
        try:
            from paraview.simple import GetActiveView, GetDisplayProperties

            if not GetActiveSource():
                return False, "Error: No data selected. Load data first.", None

            view = GetActiveView()
            display = GetDisplayProperties(GetActiveSource(), view)
            
            if enable:
                display.Visibility = 1
                status_message = "Element was made visibile"
            else:
                display.Visibility = 0
                status_message = "Rendering hidden (representation preserved)"

            # Get the source name using the helper function
            source_name = self._get_source_name(GetActiveSource())

            return True, status_message, source_name

        except Exception as e:
            self.logger.error(f"Error toggling visibility: {str(e)}")
            return False, f"Error toggling visibility: {str(e)}", None
    
    def color_by(self, field, component=-1):
        """
        Color the active visualization by a specific field.
        This function first checks if the active source can be colored by fields
        (i.e., it's a dataset with arrays) before attempting to apply colors.
        
        Args:
            field: Field name to color by.
            component: Component to color by (-1 for magnitude).
            
        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties, ColorBy
            
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first."
            
            view = GetActiveView()
            display = GetDisplayProperties(source, view)
            
            # Check if the current representation type can be colored by arrays
            # Some representations (like 'Outline') cannot be colored by data arrays
            rep_type = display.GetRepresentationType() if hasattr(display, 'GetRepresentationType') else None
            if rep_type in ['Outline', 'Wireframe']:
                return False, f"Error: The current representation type '{rep_type}' cannot be colored by fields. Try changing to 'Surface' or 'Volume' first."
            
            # Get data information directly from the source
            data_info = source.GetDataInformation()
            point_info = data_info.GetPointDataInformation()
            cell_info = data_info.GetCellDataInformation()
            
            # Check if the active source has data arrays
            if (point_info.GetNumberOfArrays() == 0 and 
                cell_info.GetNumberOfArrays() == 0):
                return False, "Error: The active source does not have any data arrays to color by."
            
            # Try to find the requested field
            field_available = False
            field_location = None
            
            # Check point data arrays
            for i in range(point_info.GetNumberOfArrays()):
                array_info = point_info.GetArrayInformation(i)
                if array_info.GetName() == field:
                    ColorBy(display, ('POINTS', field), component)
                    field_available = True
                    field_location = 'POINTS'
                    break
            
            # Check cell data arrays if not found in point data
            if not field_available:
                for i in range(cell_info.GetNumberOfArrays()):
                    array_info = cell_info.GetArrayInformation(i)
                    if array_info.GetName() == field:
                        ColorBy(display, ('CELLS', field), component)
                        field_available = True
                        field_location = 'CELLS'
                        break
            
            if not field_available:
                # Build a list of available fields for better error reporting
                available_fields = []
                for i in range(point_info.GetNumberOfArrays()):
                    array_info = point_info.GetArrayInformation(i)
                    available_fields.append(f"{array_info.GetName()} (POINTS)")
                for i in range(cell_info.GetNumberOfArrays()):
                    array_info = cell_info.GetArrayInformation(i)
                    available_fields.append(f"{array_info.GetName()} (CELLS)")
                
                fields_str = ", ".join(available_fields)
                return False, f"Error: Field '{field}' not found. Available fields are: {fields_str}"
            
            # Rescale the color map to show the full data range
            display.RescaleTransferFunctionToDataRange(True)
            return True, f"Colored by field: '{field}' from {field_location}"
        except Exception as e:
            self.logger.error(f"Error coloring by field: {str(e)}")
            return False, f"Error coloring by field: {str(e)}"

    
    def set_color_map(self, preset_name="Blue-Red"):
        """
        Set the color map (lookup table) for the current visualization.
        
        Args:
            preset_name: Name of the color map preset.
                        Available presets include (but are not limited to):
                        - Blue-Red
                        - Cool to Warm
                        - Viridis
                        - Plasma
                        - Magma
                        - Inferno
                        - Rainbow
                        - Grayscale
                        
        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties, ApplyPreset
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first."
            
            view = GetActiveView()
            display = GetDisplayProperties(source, view)
            
            color_tf = display.LookupTable
            if not color_tf:
                return False, "Error: No active color transfer function"
            
            # Apply the requested preset to the color transfer function.
            ApplyPreset(color_tf, preset_name, True)
            
            available_presets = "Blue-Red, Cool to Warm, Viridis, Plasma, Magma, Inferno, Rainbow, Grayscale"
            return True, f"Applied color map preset: {preset_name}. Available presets include: {available_presets}"
        except Exception as e:
            self.logger.error(f"Error setting color map: {str(e)}")
            return False, f"Error setting color map: {str(e)}"

    def get_histogram(self, field=None, num_bins=256, data_location="POINTS"):
        """
        Compute and retrieve histogram data for a field in the active data source.
        This function is designed to work with volume sources. By default it uses the
        point data arrays (data_location="POINTS"), but you can specify "CELLS" if your
        volume source stores scalars on cells.

        If no field is provided and the active source contains exactly one available numeric 
        field in the specified data location, that field is automatically used. If multiple 
        arrays exist, the user must specify which field to use.

        Args:
            field (str, optional): The name of the field for which the histogram is computed.
            num_bins (int, optional): Number of histogram bins (default is 10).
            data_location (str, optional): Specify "POINTS" (default) or "CELLS" to indicate the source of the data.
            
        Returns:
            tuple: (success (bool), message (str), histogram_data (list of tuples))
                histogram_data is a list of tuples (bin_center, frequency) representing the computed histogram.

        Note:
            This function uses the Histogram filter from paraview.simple and updates the pipeline.
            Since direct assignment to properties like 'NumberOfBins' is disallowed, the code retrieves
            the proper property (either "NumberOfBins" or "BinCount") via GetProperty() and sets it via SetElement().
        """
        try:
            from paraview.simple import GetActiveSource, Histogram, UpdatePipeline, servermanager
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first.", None

            # Obtain the data information from the specified location.
            data_info = source.GetDataInformation()
            data_location = data_location.upper()
            if data_location == "CELLS":
                array_info_obj = data_info.GetCellDataInformation()
            else:
                array_info_obj = data_info.GetPointDataInformation()
            num_arrays = array_info_obj.GetNumberOfArrays()

            # Automatically determine the field if not provided.
            if field is None:
                if num_arrays == 1:
                    field = array_info_obj.GetArrayInformation(0).GetName()
                else:
                    available_arrays = []
                    for i in range(num_arrays):
                        available_arrays.append(array_info_obj.GetArrayInformation(i).GetName())
                    return (
                        False,
                        "Error: Multiple fields available. Please specify a field name. Available arrays: " +
                        ", ".join(available_arrays),
                        None
                    )

            # Create and configure the Histogram filter.
            hist_filter = Histogram(Input=source)
            # Set the input array from the chosen location (POINTS or CELLS).
            hist_filter.SelectInputArray = [data_location, field]

            # Set the number of bins via GetProperty to avoid creating new attributes.
            nbins_prop = hist_filter.GetProperty("NumberOfBins")
            if nbins_prop is None:
                nbins_prop = hist_filter.GetProperty("BinCount")
            if nbins_prop is None:
                return False, "Error: Histogram filter does not have a 'NumberOfBins' or 'BinCount' property.", None
            nbins_prop.SetElement(0, num_bins)

            # Update the pipeline to compute the histogram.
            UpdatePipeline()

            # Fetch the computed histogram (returned as a vtkTable).
            hist_table = servermanager.Fetch(hist_filter)
            if hist_table.GetNumberOfRows() == 0:
                return False, "Histogram computation returned empty data.", None

            # Try to extract histogram data assuming columns named "bin_centers" and "bin_frequencies".
            bin_centers_col = hist_table.GetColumnByName("bin_centers")
            frequencies_col = hist_table.GetColumnByName("bin_frequencies")
            # Fallback: use the first two columns if the expected names do not exist.
            if not bin_centers_col or not frequencies_col:
                bin_centers_col = hist_table.GetColumn(0)
                frequencies_col = hist_table.GetColumn(1)

            histogram_data = []
            num_rows = hist_table.GetNumberOfRows()
            for i in range(num_rows):
                # Retrieve each value from the vtkArray for bin center and frequency.
                bin_center = bin_centers_col.GetValue(i)
                frequency = frequencies_col.GetValue(i)
                histogram_data.append((bin_center, frequency))

            return True, f"Histogram computed for field '{field}' in {data_location} with {num_bins} bins.", histogram_data

        except Exception as e:
            self.logger.error(f"Error computing histogram: {str(e)}")
            return False, f"Error computing histogram: {str(e)}", None


    def set_representation_type(self, rep_type):
        """
        Set the representation type for the active source.
        
        Args:
            rep_type: Representation type (Surface, Wireframe, Points, Volume, etc.)
            
        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetActiveSource, GetActiveView, GetDisplayProperties
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first."
            
            view = GetActiveView()
            display = GetDisplayProperties(source, view)
            
            display.SetRepresentationType(rep_type)
            
            return True, f"Set representation type to {rep_type}"
        except Exception as e:
            self.logger.error(f"Error setting representation type: {str(e)}")
            return False, f"Error setting representation type: {str(e)}"
    
    def edit_volume_opacity(self, field_name, opacity_points):
        """
        Edit ONLY the opacity transfer function for a given field, ensuring
        we pass only (value, alpha) pairs to ParaView.

        Args:
            field_name (str): The name of the field/array to modify.
            opacity_points (list of tuples): Each tuple must be (value, alpha).
                Example: [(0.0, 0.0), (50.0, 0.3), (100.0, 1.0)]

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            from paraview.simple import GetOpacityTransferFunction

            if not opacity_points:
                return False, "No opacity points provided."

            # Grab the opacity transfer function for the specified field
            opacity_tf = GetOpacityTransferFunction(field_name)
            if opacity_tf is None:
                return False, f"Could not find an opacity transfer function for field '{field_name}'."

            # Flatten the list of (value, alpha) into the format:
            # [val1, alpha1, midpoint1, sharpness1, val2, alpha2, midpoint2, sharpness2, ...]
            new_opacity_pts = []
            for val, alpha in opacity_points:
                new_opacity_pts.extend([val, alpha, 0.5, 0.0])  # midpoint=0.5, sharpness=0.0

            # Assign them to the piecewise function
            opacity_tf.Points = new_opacity_pts

            return True, f"Opacity transfer function updated for field '{field_name}'."

        except Exception as e:
            self.logger.error(f"Error editing opacity transfer function: {str(e)}")
            return False, f"Error editing opacity transfer function: {str(e)}"


    def set_color_map(self, field_name, color_points):
        """
        Sets the color transfer function for the given field (array) in ParaView.

        Args:
            field_name (str): The name of the field/array (as it appears in ParaView).
            color_points (list of (float, (float, float, float))):
                Each element should be a tuple: (value, (r, g, b))
                where value is the data value, and r, g, b are in [0, 1].

        Returns:
            tuple (success: bool, message: str)
        """
        try:
            from paraview.simple import GetColorTransferFunction

            if not color_points:
                return False, "No color points provided."

            # Retrieve/create the color transfer function for the specified field
            color_tf = GetColorTransferFunction(field_name)
            if color_tf is None:
                return False, f"Could not find or create a color transfer function for '{field_name}'."

            # Flatten the list into [value, R, G, B, value, R, G, B, ...]
            new_rgb_points = []
            for val, rgb in color_points:
                if len(rgb) != 3:
                    return False, f"Invalid RGB tuple for value {val}: {rgb}"
                r, g, b = rgb
                new_rgb_points.extend([val, r, g, b])

            # Update the color transfer function
            color_tf.RGBPoints = new_rgb_points

            # Optionally, you can rescale the transfer function based on min and max values
            # Example:
            # min_val = min([pt[0] for pt in color_points])
            # max_val = max([pt[0] for pt in color_points])
            # color_tf.RescaleTransferFunction(min_val, max_val)

            return True, f"Color transfer function updated for field '{field_name}'."

        except Exception as e:
            msg = f"Error setting color map: {str(e)}"
            return False, msg


    def get_pipeline(self):
        """
        Get the current pipeline structure.
        
        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetSources
            sources = GetSources()
            if not sources:
                return True, "Pipeline is empty. No sources found."
            
            response = "Current pipeline:\n"
            for name, source in sources.items():
                response += f"- {name[0]}: {source.__class__.__name__}\n"
            return True, response
        except Exception as e:
            self.logger.error(f"Error getting pipeline: {str(e)}")
            return False, f"Error getting pipeline: {str(e)}"
    
    def get_available_arrays(self):
        """
        Get a list of available arrays in the active source.

        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetActiveSource
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first."

            # Obtain comprehensive data information from the source.
            data_info = source.GetDataInformation()
            point_info = data_info.GetPointDataInformation()
            cell_info  = data_info.GetCellDataInformation()

            response = "Available arrays:\n\nPoint data arrays:\n"
            if point_info:
                num_point_arrays = point_info.GetNumberOfArrays()
                for i in range(num_point_arrays):
                    # Get the array information for each point array.
                    array_info = point_info.GetArrayInformation(i)
                    array_name = array_info.GetName()  # Use GetName() rather than GetArrayName()
                    components = array_info.GetNumberOfComponents()
                    response += f"- {array_name} ({components} components)\n"
            else:
                response += "No point data arrays found.\n"

            response += "\nCell data arrays:\n"
            if cell_info:
                num_cell_arrays = cell_info.GetNumberOfArrays()
                for i in range(num_cell_arrays):
                    # Get the array information for each cell array.
                    array_info = cell_info.GetArrayInformation(i)
                    array_name = array_info.GetName()
                    components = array_info.GetNumberOfComponents()
                    response += f"- {array_name} ({components} components)\n"
            else:
                response += "No cell data arrays found.\n"

            return True, response
        except Exception as e:
            self.logger.error(f"Error getting available arrays: {str(e)}")
            return False, f"Error getting available arrays: {str(e)}"

    def create_stream_tracer(self, vector_field=None, base_source=None, point_center=None,
                            integration_direction="BOTH", 
                            initial_step_length=0.1,
                            maximum_stream_length=50.0,
                            number_of_streamlines=100,
                            point_radius=1.0,
                            tube_radius=0.1,
                            make_volume_transparent=True):
        """
        Create a stream tracer visualization for a vector volume with tube representation.

        Args:
            vector_field (str, optional): Name of the vector field to trace.
                                        If None, the function automatically selects
                                        the first array with more than one component.
            base_source (optional): The data source (volume) on which to perform stream tracing.
                                If None, uses self.primary_data_source or GetActiveSource().
            point_center (list, optional): Center coordinates [x, y, z] for the seed points.
                                        If None, the center of the volume's bounds is used.
            integration_direction (str): "FORWARD", "BACKWARD", or "BOTH" for integration.
            initial_step_length (float): The initial step size.
            maximum_stream_length (float): Maximum streamline length, beyond which integration terminates.
            number_of_streamlines (int): Number of seed points if a default seed is created.
            point_radius (float): Radius for the Point Cloud seed.
            tube_radius (float): Radius for the tube visualization.
            make_volume_transparent (bool): Whether to make the base volume transparent.

        Returns:
            tuple: (success (bool), message (str), tube filter proxy, tube_name (str))
        """
        try:
            from paraview.simple import (GetActiveSource, StreamTracer, 
                                        Show, SetActiveSource, Tube,
                                        GetDisplayProperties, ColorBy)

            # Determine the base source: use provided, or self.primary_data_source, or the active source.
            if base_source is None:
                base_source = self.primary_data_source or GetActiveSource()
            if not base_source:
                return False, "Error: No active source. Load data first.", None, ""

            # Log the base source name if available.
            base_source_name = None
            if hasattr(base_source, "SMProxy") and hasattr(base_source.SMProxy, "GetXMLName"):
                base_source_name = base_source.SMProxy.GetXMLName()
            else:
                base_source_name = str(base_source)
            self.logger.info(f"Using base source: {base_source_name}")

            # If vector_field is not provided, get the first available multi-component array.
            if vector_field is None:
                # Retrieve the data information and then its point data information.
                data_info = base_source.GetDataInformation()
                point_info = data_info.GetPointDataInformation()
                if point_info:
                    num_arrays = point_info.GetNumberOfArrays()
                    found = False
                    for i in range(num_arrays):
                        array_info = point_info.GetArrayInformation(i)
                        components = array_info.GetNumberOfComponents()
                        if components > 1:
                            vector_field = array_info.GetName()
                            found = True
                            self.logger.info(f"Automatically selected vector field: {vector_field}")
                            break
                    if not found:
                        if num_arrays > 0:
                            vector_field = point_info.GetArrayInformation(0).GetName()
                            self.logger.info(f"No multi-component array found; selected first array: {vector_field}")
                        else:
                            return False, "Error: No arrays found in the base source.", None, ""
                else:
                    return False, "Error: Could not retrieve point data information.", None, ""

            # Determine point center for the seed source
            center = point_center
            if center is None:
                data_info = base_source.GetDataInformation()
                bounds = data_info.GetBounds()  # Format: [xmin, xmax, ymin, ymax, zmin, zmax]
                center = [(bounds[0] + bounds[1]) / 2.0,
                        (bounds[2] + bounds[3]) / 2.0,
                        (bounds[4] + bounds[5]) / 2.0]
                self.logger.info(f"Using auto-calculated center point at {center}")

            # Create the stream tracer filter using Point Cloud seed type
            tracer = StreamTracer(Input=base_source, SeedType='Point Cloud')
            tracer.Vectors = ['POINTS', vector_field]
            tracer.IntegrationDirection = integration_direction
            tracer.InitialStepLength = initial_step_length
            tracer.MaximumStreamlineLength = maximum_stream_length
            
            # Configure the Point Cloud seed
            tracer.SeedType.Center = center
            tracer.SeedType.NumberOfPoints = number_of_streamlines
            tracer.SeedType.Radius = point_radius
            
            # Display the tracer result
            Show(tracer)
            
            # Create tube filter for better visualization
            tube = Tube(Input=tracer)
            tube.Radius = tube_radius
            
            # Show the tube filter
            # Display the tube with proper coloring
            tube_display = Show(tube)
            ColorBy(tube_display, ('POINTS', vector_field))

            # Make the base source transparent if requested
            if make_volume_transparent:
                try:
                    base_display = GetDisplayProperties(base_source)
                    base_display.Opacity = 0.3  # Set opacity to make volume transparent
                except Exception as e:
                    self.logger.warning(f"Could not make volume transparent: {str(e)}")
            
            # Set the active source to the tube filter
            SetActiveSource(tube)
            
            # Get the tube filter name using the helper function
            tube_name = self._get_source_name(tube)

            msg = f"Stream tracer with tubes created for vector field '{vector_field}' using base source '{base_source_name}'."
            return True, msg, tube, tube_name
        except Exception as e:
            self.logger.error(f"Error creating stream tracer: {str(e)}")
            return False, f"Error creating stream tracer: {str(e)}", None, ""

    def get_screenshot(self):
        """
        Enhanced screenshot capture with better cleanup and error handling.
        
        Returns:
            tuple: (success, message, img_path)
        """
        try:
            from paraview.collaboration import processServerEvents
            import os
            
            processServerEvents()
            from paraview import servermanager
            from paraview.simple import SetActiveView, RenderAllViews, SaveScreenshot
            
            # Get the active render view from the GUI connection
            pxm = servermanager.ProxyManager()
            gui_view = None
            views = pxm.GetProxiesInGroup("views")
            for (group, name), view_proxy in views.items():
                if view_proxy.GetXMLName() == "RenderView":
                    gui_view = view_proxy
                    break
            
            if not gui_view:
                self.logger.error("No existing GUI render view found. Make sure the ParaView GUI is connected.")
                return False, "No GUI render view found", None
            
            # Set the found GUI view active and ensure rendering
            SetActiveView(gui_view)
            RenderAllViews()
            
            # Create screenshot file in current working directory with timestamp
            import datetime
            
            # Get current working directory
            current_dir = os.getcwd()
            
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"paraview_screenshot_{timestamp}.png"
            screenshot_path = os.path.join(current_dir, filename)
            
            self.logger.info(f"Saving screenshot to: {screenshot_path}")
            
            # Save screenshot with enhanced resolution
            SaveScreenshot(screenshot_path, gui_view, ImageResolution=[1920, 1080])
            
            # Verify file was created
            if not os.path.exists(screenshot_path):
                return False, "Screenshot file was not created", None
                
            file_size = os.path.getsize(screenshot_path)
            self.logger.info(f"Screenshot saved successfully: {screenshot_path} ({file_size} bytes)")
            
            return True, f"Screenshot saved to: {filename}", screenshot_path
        except Exception as e:
            self.logger.error(f"Error getting screenshot: {str(e)}")
            return False, f"Error getting screenshot: {str(e)}", None
    

    def rotate_camera(self, azimuth=30.0, elevation=0.0):
        """
        Rotate the camera by specified angles.
        
        Args:
            azimuth: Rotation around vertical axis in degrees.
            elevation: Rotation around horizontal axis in degrees.
            
        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetActiveView
            view = GetActiveView()
            if not view:
                return False, "Error: No active view."
            
            camera = view.GetActiveCamera()
            camera.Azimuth(azimuth)
            camera.Elevation(elevation)
            return True, f"Rotated camera by azimuth: {azimuth}, elevation: {elevation}"
        except Exception as e:
            self.logger.error(f"Error rotating camera: {str(e)}")
            return False, f"Error rotating camera: {str(e)}"
    
    def reset_camera(self):
        """
        Reset the camera to show all data.
        
        Returns:
            tuple: (success, message)
        """
        try:
            from paraview.simple import GetActiveView, ResetCamera
            view = GetActiveView()
            if not view:
                return False, "Error: No active view."
            ResetCamera(view)
            return True, "Camera reset"
        except Exception as e:
            self.logger.error(f"Error resetting camera: {str(e)}")
            return False, f"Error resetting camera: {str(e)}"
    

    def plot_over_line(self, point1=None, point2=None, resolution=100):
        """
        Create a 'Plot Over Line' filter to sample data along a line between two points.

        Args:
            point1 (list/tuple or None): The [x, y, z] coordinates of the start point. If None, will use data bounds.
            point2 (list/tuple or None): The [x, y, z] coordinates of the end point. If None, will use data bounds.
            resolution (int): Number of sample points along the line (default: 100).

        Returns:
            tuple: (success: bool, message: str, plot_filter)
        """
        try:
            from paraview.simple import GetActiveSource, PlotOverLine, Show, GetActiveView, CreateView, AssignViewToLayout
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first.", None

            # Create the PlotOverLine filter
            plot_filter = PlotOverLine(Input=source)
            if point1 is not None:
                plot_filter.Point1 = point1
            if point2 is not None:
                plot_filter.Point2 = point2
            plot_filter.Resolution = resolution

            # Show the result in the active view (usually a line chart view)
            view = GetActiveView()
            Show(plot_filter, view)
            # Create a new 'Line Chart View'
            lineChartView1 = CreateView('XYChartView')

            # show data in view
            plotOverLine1Display_1 = Show(plot_filter, lineChartView1, 'XYChartRepresentation')
            AssignViewToLayout(view=lineChartView1)

            return True, f"Plot over line created from {plot_filter.Point1} to {plot_filter.Point2} with {resolution} points.", plot_filter
        except Exception as e:
            self.logger.error(f"Error creating plot over line: {str(e)}")
            return False, f"Error creating plot over line: {str(e)}", None

    def warp_by_vector(self, vector_field=None, scale_factor=1.0):
        """
        Apply the 'Warp By Vector' filter to the active source.

        Args:
            vector_field (str, optional): The name of the vector field to use for warping. If None, the first available vector field will be used.
            scale_factor (float, optional): The scale factor for the warp (default: 1.0).

        Returns:
            tuple: (success: bool, message: str, warp_filter)
        """
        try:
            from paraview.simple import GetActiveSource, WarpByVector, Show, GetActiveView
            source = GetActiveSource()
            if not source:
                return False, "Error: No active source. Load data first.", None

            # If vector_field is not specified, try to auto-detect a vector field
            if vector_field is None:
                data_info = source.GetDataInformation()
                point_info = data_info.GetPointDataInformation()
                num_arrays = point_info.GetNumberOfArrays()
                found = False
                for i in range(num_arrays):
                    array_info = point_info.GetArrayInformation(i)
                    if array_info.GetNumberOfComponents() > 1:
                        vector_field = array_info.GetName()
                        found = True
                        break
                if not found:
                    return False, "No vector field found in the active source.", None

            # Create the WarpByVector filter
            warp_filter = WarpByVector(Input=source)
            warp_filter.Vectors = ['POINTS', vector_field]
            warp_filter.ScaleFactor = scale_factor

            # Show the result in the active view
            view = GetActiveView()
            Show(warp_filter, view)

            return True, f"Warp by vector applied using field '{vector_field}' with scale factor {scale_factor}.", warp_filter
        except Exception as e:
            self.logger.error(f"Error creating warp by vector: {str(e)}")
            return False, f"Error creating warp by vector: {str(e)}", None
    
    def _check_paraview_adios2_support(self):
        """
        Check if ParaView was built with ADIOS2 support by examining build configuration.
        
        This method provides comprehensive ADIOS2 support detection including:
        - CMake cache analysis for build flags
        - ParaView module availability checking
        - Build path validation and reporting
        
        Returns:
            dict: Comprehensive status information about ADIOS2 support containing:
                  - has_support (bool): True if ADIOS2 support is available
                  - message (str): Human-readable status description
                  - build_path (str): Path to ParaView build directory if found
                  - cmake_flags (list): List of ADIOS2-related CMake flags found
                  
        Example:
            >>> status = manager._check_paraview_adios2_support()
            >>> if status['has_support']:
            ...     print("ADIOS2 support available")
            ... else:
            ...     print(f"No ADIOS2 support: {status['message']}")
        """
        status = {
            'has_support': False,
            'message': 'Unknown ADIOS2 support status',
            'build_path': '',
            'cmake_flags': []
        }
        
        try:
            import os
            
            # Check common ParaView build directories
            possible_build_paths = [
                '/home/shazzadul/Illinois_Tech/Summer25/paraview_build',
                os.path.expanduser('~/paraview_build'),
                os.path.expanduser('~/ParaView/build'),
                '/usr/local/paraview/build',
                '/opt/paraview/build'
            ]
            
            cmake_cache_path = None
            for build_path in possible_build_paths:
                if os.path.exists(build_path):
                    cache_file = os.path.join(build_path, 'CMakeCache.txt')
                    if os.path.exists(cache_file):
                        cmake_cache_path = cache_file
                        status['build_path'] = build_path
                        break
            
            if cmake_cache_path:
                self.logger.info(f"Found ParaView build cache at: {cmake_cache_path}")
                
                # Read and analyze CMake cache for ADIOS2 configuration
                with open(cmake_cache_path, 'r') as f:
                    cache_content = f.read()
                
                # Extract ADIOS2-related configuration flags
                adios2_flags = []
                for line in cache_content.split('\n'):
                    if 'ADIOS2' in line and '=' in line:
                        adios2_flags.append(line.strip())
                
                status['cmake_flags'] = adios2_flags
                
                # Analyze specific ADIOS2 configuration flags
                enable_adios2 = False
                use_adios2 = False
                
                for flag in adios2_flags:
                    if 'PARAVIEW_ENABLE_ADIOS2:BOOL=ON' in flag:
                        enable_adios2 = True
                    elif 'PARAVIEW_USE_ADIOS2' in flag and ('=ON' in flag or ':UNINITIALIZED=ON' in flag):
                        use_adios2 = True
                
                # Determine final ADIOS2 support status
                if enable_adios2:
                    status['has_support'] = True
                    status['message'] = 'ParaView built with ADIOS2 support enabled'
                elif use_adios2:
                    status['has_support'] = False
                    status['message'] = 'ADIOS2 available but not enabled in ParaView build'
                else:
                    status['has_support'] = False
                    status['message'] = 'ParaView built without ADIOS2 support'
                    
                self.logger.info(f"ADIOS2 support analysis: {status['message']}")
            else:
                status['message'] = 'ParaView build directory not found - cannot determine ADIOS2 support'
                self.logger.warning(status['message'])
                
        except Exception as e:
            error_msg = f'Error checking ADIOS2 support: {str(e)}'
            status['message'] = error_msg
            self.logger.error(error_msg)
        
        return status
    
    def _convert_adios_to_vtk_improved(self, file_path):
        """
        Convert ADIOS2 BP5 file to VTK format using enhanced conversion with comprehensive error handling.
        
        This method provides robust BP5 to VTK conversion including:
        - Multi-strategy conversion attempts
        - Comprehensive BP5 file diagnosis
        - Detailed error reporting and troubleshooting guidance
        - Support for multiple ADIOS2 engines and approaches
        
        Args:
            file_path (str): Absolute path to the ADIOS2 BP5 file or directory
            
        Returns:
            str: Path to successfully converted VTK file, or None if conversion fails
                 
        Example:
            >>> vtk_path = manager._convert_adios_to_vtk_improved("/data/simulation.bp")
            >>> if vtk_path:
            ...     print(f"Converted to: {vtk_path}")
            ... else:
            ...     print("Conversion failed")
        
        Note:
            - Requires ADIOS2 Python bindings to be installed
            - Creates .vti (VTK ImageData) output files
            - Automatically handles multiple conversion strategies
            - Provides detailed diagnostics for troubleshooting
        """
        if not ADIOS2_AVAILABLE:
            self.logger.error("ADIOS2 not available - cannot convert BP5 files")
            return None
            
        try:
            import os
            
            self.logger.info(f"Starting enhanced ADIOS2 conversion for: {file_path}")
            
            # Validate input file/directory exists
            if not os.path.exists(file_path):
                self.logger.error(f"BP5 file not found: {file_path}")
                return None
            
            # Basic BP5 format validation
            if not (os.path.isdir(file_path) or file_path.endswith(('.bp', '.bp5'))):
                self.logger.error(f"Invalid BP5 format: {file_path}")
                return None
            
            # Create output VTK file path
            base_name = os.path.splitext(file_path)[0]
            output_file = f"{base_name}_converted.vti"
            
            # Try multiple conversion strategies
            conversion_strategies = [
                self._convert_strategy_streaming_api,
                self._convert_strategy_direct_bp5,
                self._convert_strategy_alternative_engines
            ]
            
            for i, strategy in enumerate(conversion_strategies, 1):
                strategy_name = strategy.__name__.replace('_convert_strategy_', '').replace('_', ' ').title()
                self.logger.info(f"Attempting conversion strategy {i}: {strategy_name}")
                
                try:
                    if strategy(file_path, output_file):
                        if os.path.exists(output_file):
                            file_size = os.path.getsize(output_file)
                            self.logger.info(f"Successfully converted using {strategy_name}: {output_file} ({file_size} bytes)")
                            return output_file
                        else:
                            self.logger.warning(f"Strategy {strategy_name} reported success but no output file created")
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy_name} failed: {str(e)}")
            
            # All strategies failed
            self.logger.error("All BP5 conversion strategies failed")
            self.logger.error("This may be due to:")
            self.logger.error("  1. Incompatible ADIOS2 version or installation")
            self.logger.error("  2. Corrupted or unsupported BP5 file format")
            self.logger.error("  3. Missing VTK Python bindings")
            self.logger.error("  4. Insufficient memory for large datasets")
            return None
        
        except Exception as e:
            self.logger.error(f"Enhanced BP5 conversion failed with exception: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _convert_strategy_streaming_api(self, file_path, output_file):
        """
        Convert BP5 using ADIOS2 streaming API - most reliable approach.
        
        Args:
            file_path (str): Path to BP5 file
            output_file (str): Path for output VTK file
            
        Returns:
            bool: True if conversion succeeded, False otherwise
        """
        try:
            import adios2
            import numpy as np
            
            self.logger.info("Using ADIOS2 streaming API for conversion")
            
            # Use ADIOS2 streaming approach
            with adios2.open(file_path, 'r') as f:
                available_vars = f.available_variables()
                
                if not available_vars:
                    self.logger.warning("No variables found in BP5 file")
                    return False
                
                self.logger.info(f"Found {len(available_vars)} variables: {list(available_vars.keys())}")
                
                # Find the largest 3D variable for conversion
                main_var = None
                max_size = 0
                
                for var_name, var_info in available_vars.items():
                    try:
                        # Check if variable has shape information
                        if 'Shape' in var_info and var_info['Shape']:
                            shape_str = var_info['Shape']
                            shape = [int(x.strip()) for x in shape_str.split(',')]
                            
                            if len(shape) >= 3:  # 3D or higher dimensional data
                                size = np.prod(shape)
                                if size > max_size:
                                    max_size = size
                                    main_var = var_name
                                    
                    except Exception as e:
                        self.logger.warning(f"Error processing variable {var_name}: {e}")
                
                if not main_var:
                    # Fallback: use first variable
                    main_var = list(available_vars.keys())[0]
                    self.logger.info(f"Using fallback variable: {main_var}")
                
                # Read the main variable data
                self.logger.info(f"Reading variable: {main_var}")
                data = f.read(main_var)
                
                if data is None or data.size == 0:
                    self.logger.error(f"Failed to read data for variable {main_var}")
                    return False
                
                self.logger.info(f"Successfully read data: shape={data.shape}, dtype={data.dtype}")
                
                # Create VTK ImageData and save
                return self._create_vtk_from_numpy(data, main_var, output_file)
                
        except Exception as e:
            self.logger.warning(f"Streaming API conversion failed: {str(e)}")
            return False
    
    def _convert_strategy_direct_bp5(self, file_path, output_file):
        """
        Convert BP5 using direct ADIOS2 BP5 engine.
        
        Args:
            file_path (str): Path to BP5 file
            output_file (str): Path for output VTK file
            
        Returns:
            bool: True if conversion succeeded, False otherwise
        """
        try:
            import adios2
            import numpy as np
            
            self.logger.info("Using direct ADIOS2 BP5 engine for conversion")
            
            # Create ADIOS2 instance and IO with BP5 engine
            adios = adios2.ADIOS()
            io = adios.DeclareIO("direct_bp5")
            io.SetEngine("BP5")
            
            # Open BP5 file
            reader = io.Open(file_path, adios2.Mode.Read)
            available_vars = io.AvailableVariables()
            
            if not available_vars:
                self.logger.warning("No variables found with BP5 engine")
                reader.Close()
                return False
            
            self.logger.info(f"BP5 engine found {len(available_vars)} variables")
            
            # Try to read variables using begin/end step approach
            data_arrays = {}
            main_shape = None
            
            for var_name in list(available_vars.keys())[:3]:  # Limit to first 3 variables
                try:
                    var_obj = io.InquireVariable(var_name)
                    if var_obj and var_obj.Shape():
                        step_status = reader.BeginStep()
                        if step_status == adios2.StepStatus.OK:
                            data = np.zeros(var_obj.Shape(), dtype=var_obj.Type())
                            reader.Get(var_obj, data)
                            reader.EndStep()
                            
                            data_arrays[var_name] = data
                            if main_shape is None and len(data.shape) >= 3:
                                main_shape = data.shape
                                
                            self.logger.info(f"Read variable {var_name}: shape={data.shape}")
                        else:
                            self.logger.warning(f"BeginStep failed for {var_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to read variable {var_name}: {e}")
            
            reader.Close()
            
            # Create VTK file from the largest array
            if data_arrays:
                main_var = max(data_arrays.keys(), key=lambda k: data_arrays[k].size)
                return self._create_vtk_from_numpy(data_arrays[main_var], main_var, output_file)
            else:
                return False
                
        except Exception as e:
            self.logger.warning(f"Direct BP5 conversion failed: {str(e)}")
            return False
    
    def _convert_strategy_alternative_engines(self, file_path, output_file):
        """
        Try alternative ADIOS2 engines as fallback.
        
        Args:
            file_path (str): Path to BP5 file
            output_file (str): Path for output VTK file
            
        Returns:
            bool: True if conversion succeeded, False otherwise
        """
        alternative_engines = ["BPFile", "BP4", "SST"]
        
        for engine in alternative_engines:
            try:
                import adios2
                
                self.logger.info(f"Trying alternative engine: {engine}")
                
                adios = adios2.ADIOS()
                io = adios.DeclareIO(f"alt_{engine}")
                io.SetEngine(engine)
                
                reader = io.Open(file_path, adios2.Mode.Read)
                available_vars = io.AvailableVariables()
                reader.Close()
                
                if available_vars:
                    self.logger.info(f"Engine {engine} found {len(available_vars)} variables")
                    # Could implement full conversion here if needed
                    # For now, just report availability
                    return False  # Placeholder - not fully implemented
                
            except Exception as e:
                self.logger.warning(f"Alternative engine {engine} failed: {e}")
        
        return False
    
    def _create_vtk_from_numpy(self, data, var_name, output_file):
        """
        Create VTK ImageData file from numpy array.
        
        Args:
            data (numpy.ndarray): Input data array
            var_name (str): Variable name for the data
            output_file (str): Path for output VTK file
            
        Returns:
            bool: True if VTK file created successfully, False otherwise
        """
        try:
            import vtk
            from vtk.util import numpy_support
            import numpy as np
            
            self.logger.info(f"Creating VTK ImageData from array: shape={data.shape}, dtype={data.dtype}")
            
            # Create VTK ImageData
            image_data = vtk.vtkImageData()
            
            # Set dimensions based on data shape
            if len(data.shape) == 3:
                # 3D data: use Z,Y,X order for VTK
                dims = (data.shape[2], data.shape[1], data.shape[0])
                # Transpose for VTK ordering
                data = np.transpose(data, (2, 1, 0))
            elif len(data.shape) == 2:
                # 2D data: add Z dimension
                dims = (data.shape[1], data.shape[0], 1)
                data = np.transpose(data, (1, 0))
            elif len(data.shape) == 1:
                # 1D data: treat as line
                dims = (data.shape[0], 1, 1)
            else:
                self.logger.error(f"Unsupported data dimensionality: {data.shape}")
                return False
            
            image_data.SetDimensions(dims)
            image_data.SetSpacing(1.0, 1.0, 1.0)
            image_data.SetOrigin(0.0, 0.0, 0.0)
            
            # Convert numpy array to VTK array
            vtk_array = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True)
            vtk_array.SetName(var_name)
            
            # Set as scalar data
            image_data.GetPointData().SetScalars(vtk_array)
            
            # Write VTK file
            writer = vtk.vtkXMLImageDataWriter()
            writer.SetFileName(output_file)
            writer.SetInputData(image_data)
            write_result = writer.Write()
            
            if write_result:
                self.logger.info(f"Successfully created VTK file: {output_file}")
                return True
            else:
                self.logger.error("VTK writer failed to create file")
                return False
                
        except Exception as e:
            self.logger.error(f"VTK creation failed: {str(e)}")
            return False