import numpy as np
import adios2

def read_variable_chunked(
    filename: str, variable_name: str, target_step: int, start_index: int = 0, chunk_size: int = 1000
):
    """
    Read a single chunk of a large variable to avoid memory issues.
    
    Args:
        filename: Path to the .bp directory (basename.bp)
        variable_name: Name of the variable to read
        target_step: The step index to read from
        start_index: Starting index along the first dimension
        chunk_size: Number of elements to read per chunk along first dimension
        
    Returns:
        Dict containing:
        - data: The chunk data as a flattened list
        - chunk_info: Metadata about this chunk and remaining data
        
    Raises:
        ValueError: if the step or variable is not found
    """
    # Initialize ADIOS
    adios = adios2.Adios()
    io = adios.declare_io("chunk_reader")
    engine = io.open(filename, adios2.bindings.adios2_bindings.Mode.Read)
    
    try:
        # Check if target step is available
        available_steps = engine.steps()
        if target_step >= available_steps:
            raise ValueError(f"Step {target_step} not found. Max step: {available_steps-1}")
        
        # Begin step
        engine.begin_step()
        
        # Get variable
        var = io.inquire_variable(variable_name)
        if not var:
            raise ValueError(f"Variable '{variable_name}' not found")
        
        # Get variable shape
        shape = var.shape()
        
        if len(shape) == 0:
            # Scalar variable
            data = np.zeros((), dtype=np.float64)
            engine.get(var, data)
            engine.end_step()
            return {
                "data": data.item(),
                "chunk_info": {
                    "is_complete": True,
                    "total_elements": 1,
                    "current_chunk": 1,
                    "total_chunks": 1
                }
            }
        
        # For arrays, read single chunk along the first dimension
        total_elements_first_dim = int(shape[0])
        
        # Validate start_index
        if start_index >= total_elements_first_dim:
            raise ValueError(f"start_index {start_index} >= total_elements {total_elements_first_dim}")
        
        # Calculate chunk boundaries
        end_index = min(start_index + chunk_size, total_elements_first_dim)
        actual_chunk_size = end_index - start_index
        
        # Set selection for this chunk
        if len(shape) == 1:
            # 1D array
            var.set_selection([[start_index], [actual_chunk_size]])
            chunk_shape = (actual_chunk_size,)
        else:
            # Multi-dimensional array - chunk along first dimension only
            selection_start = [start_index] + [0] * (len(shape) - 1)
            selection_count = [actual_chunk_size] + list(shape[1:])
            var.set_selection([selection_start, selection_count])
            chunk_shape = tuple(selection_count)
        
        # Read this chunk
        chunk_data = np.zeros(chunk_shape, dtype=np.float64)
        engine.get(var, chunk_data)
        
        engine.end_step()
        
        # Convert to list
        if chunk_data.ndim == 1:
            data = chunk_data.tolist()
        else:
            data = chunk_data.reshape(-1).tolist()
        
        # Calculate chunk metadata
        total_chunks = (total_elements_first_dim + chunk_size - 1) // chunk_size
        current_chunk = (start_index // chunk_size) + 1
        is_complete = end_index >= total_elements_first_dim
        next_start_index = end_index if not is_complete else None
        
        return {
            "data": data,
            "chunk_info": {
                "is_complete": is_complete,
                "current_chunk": current_chunk,
                "total_chunks": total_chunks,
                "start_index": start_index,
                "end_index": end_index,
                "next_start_index": next_start_index,
                "chunk_size": actual_chunk_size,
                "total_elements_first_dim": total_elements_first_dim,
                "variable_shape": list(shape)
            }
        }
        
    finally:
        engine.close()