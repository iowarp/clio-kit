# mcp_handlers.py
import json
from typing import Any, Dict, Optional
from implementation import bp5_list, bp5_inspect_variables, bp5_attributes, bp5_read_variable_at_step, bp5_inspect_variables_at_step, bp5_read_variable_chunk

class UnknownToolError(Exception):
    """Raised when an unsupported tool_name is requested."""
    pass

async def list_bp5_files(directory: str = "data") -> Dict[str, Any]:
    """
    List BP5 files in a directory.
    
    Args:
        directory: Path to the directory containing BP5 files
        
    Returns:
        Dict containing list of files and metadata
    """
    try:
        files = bp5_list.list_bp5(directory)
        return {"files": files}
    except Exception as e:
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "list_bp5", "error": type(e).__name__},
            "isError": True
        }
    
async def inspect_variables_handler(filename: str, variable_name: str = None) -> Dict[str, Any]:
    """
    Async handler for 'inspect_variables' tool.
    
    Args:
        filename: Path to the BP5 file
        variable_name: Optional name of specific variable to inspect
        
    Returns:
        Dict containing either metadata for all variables or data for a specific variable
    """
    try:
        if variable_name:
            # If variable name is provided, use read_variable_at_step to get its data
            # We'll get data from step 0 as a default
            return bp5_inspect_variables.inspect_variables(filename, variable_name)
        else:
            # If no variable name, return metadata for all variables
            return bp5_inspect_variables.inspect_variables(filename)
    except Exception as e:
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "inspect_variables", "error": type(e).__name__},
            "isError": True
        }

async def inspect_variables_at_step_handler(
    filename: str, variable_name: str, step: int
) -> Dict[str, Any]:
    """
    Async handler for 'inspect_variables_at_step' tool.
    
    Args:
        filename: Path to the BP5 file
        variable_name: Name of the variable to inspect
        step: Step number to inspect
        
    Returns:
        Dict containing variable metadata or error information
    """
    try:
        result = bp5_inspect_variables_at_step.inspect_variables_at_step(filename, variable_name, step)
        return result
    except Exception as e:
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "inspect_variables_at_step", "error": type(e).__name__},
            "isError": True
        }

async def inspect_attributes_handler(
    filename: str, variable_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Async handler for 'inspect_attributes' tool.
    """
    try:
        return bp5_attributes.inspect_attributes(filename, variable_name)
    except Exception as e:
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "inspect_attributes", "error": type(e).__name__},
            "isError": True
        }
    
async def read_variable_at_step_handler(
    filename: str, variable_name: str, target_step: int
) -> Dict[str, Any]:
    try:
        value = bp5_read_variable_at_step.read_variable_at_step(filename, variable_name, target_step)
        return {"value": value}
    except Exception as e:
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "read_variable_at_step", "error": type(e).__name__},
            "isError": True
        }

async def read_variable_chunk_handler(
    filename: str, variable_name: str, target_step: int, start_index: int = 0
) -> Dict[str, Any]:
    try:
        # First inspect the variable to calculate appropriate chunk size
        variable_info = bp5_inspect_variables.inspect_variables(filename, variable_name)
        
        # Extract shape information from the variable info
        if variable_name in variable_info:
            shape_str = variable_info[variable_name].get("Shape", "")
            if shape_str == "":
                # Scalar variable
                total_elements = 1
            else:
                # Parse shape string like "7813, 5" into actual dimensions
                try:
                    shape_parts = [int(x.strip()) for x in shape_str.split(",")]
                    total_elements = 1
                    for dim in shape_parts:
                        total_elements *= dim
                except:
                    total_elements = 1  # fallback for unparseable shapes
        else:
            total_elements = 1  # fallback if variable not found in info
        
        # Calculate chunk size (roughly total/100, but between 100-500)
        chunk_size = max(100, min(500, total_elements // 100))
        
        result = bp5_read_variable_chunk.read_variable_chunked(
            filename, variable_name, target_step, start_index, chunk_size
        )
        return {
            "value": result["data"],
            "chunk_info": result["chunk_info"]
        }
    except Exception as e:
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "read_variable_chunk", "error": type(e).__name__},
            "isError": True
        }
    
