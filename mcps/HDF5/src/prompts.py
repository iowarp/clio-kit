"""
Prompt management for HDF5 MCP server.

This module provides functionality for managing and generating prompts
for common HDF5 operations and analysis tasks.
"""

#!/usr/bin/env python3
# /// script
# dependencies = [
#   "mcp>=1.4.0",
#   "h5py>=3.9.0",
#   "numpy>=1.24.0,<2.0.0",
#   "jinja2>=3.1.0"
# ]
# requires-python = ">=3.10"
# ///

import logging
import json
import functools
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar, Callable, cast
from threading import Lock

import h5py
import numpy as np
from mcp.types import (
    Prompt,
    PromptMessage,
    PromptArgument,
    GetPromptResult,
    TextContent
)
from jinja2 import Template

from .utils import HDF5Manager

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')

# Cache for previously generated prompt content
prompt_cache: Dict[str, Dict[str, GetPromptResult]] = {
    "file_summary": {},
    "group_summary": {},
    "dataset_summary": {}
}

# Cache TTL in seconds
CACHE_TTL = 300  # 5 minutes

# =========================================================================
# Decorators for Performance and Logging
# =========================================================================

def log_prompt_generation(func: Callable) -> Callable:
    """Decorator for standardized logging of prompt generation."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        prompt_type = func.__name__
        logger.info(f"Generating prompt: {prompt_type} with args: {kwargs}")
        result = await func(*args, **kwargs)
        logger.info(f"Generated prompt: {prompt_type}")
        return result
    return wrapper

def measure_prompt_performance(func: Callable) -> Callable:
    """Decorator for measuring prompt generation performance."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.debug(f"Performance: {func.__name__} took {execution_time:.4f} seconds")
        return result
    return wrapper

def cache_prompt_result(cache_key_fn: Callable) -> Callable:
    """
    Decorator for caching prompt generation results.
    
    Args:
        cache_key_fn: Function that takes the same args/kwargs as the wrapped function
                     and returns a string cache key.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Determine the cache category from the function name
            cache_category = func.__name__
            if cache_category not in prompt_cache:
                prompt_cache[cache_category] = {}
            
            # Generate cache key
            cache_key = cache_key_fn(*args, **kwargs)
            cache_entry = prompt_cache[cache_category].get(cache_key)
            
            # Check for valid cached entry
            now = time.time()
            if cache_entry and cache_entry.get("timestamp", 0) + CACHE_TTL > now:
                logger.debug(f"Cache hit for {cache_category}:{cache_key}")
                return cache_entry["result"]
            
            # Generate new result
            result = await func(*args, **kwargs)
            
            # Cache the result
            prompt_cache[cache_category][cache_key] = {
                "result": result,
                "timestamp": now
            }
            
            return result
        return wrapper
    return decorator

# =========================================================================
# Prompt Registry System
# =========================================================================

class PromptRegistry:
    """Registry for HDF5 prompts with automatic documentation."""
    
    # Class-level storage for registered prompts
    _prompts: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, description: Optional[str] = None):
        """
        Decorator to register a method as a prompt.
        
        Args:
            description: Override the docstring for the prompt description

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            prompt_name = func.__name__
            
            # Extract argument info from the function signature
            import inspect
            sig = inspect.signature(func)
            prompt_args = []
            
            for param_name, param in sig.parameters.items():
                if param_name == 'cls' or param_name == 'self':
                    continue
                
                # Determine if parameter is required
                required = param.default == inspect.Parameter.empty
                
                # Determine parameter type
                param_type = "string"  # Default type
                annotation = param.annotation
                if annotation == bool:
                    param_type = "boolean"
                elif annotation == int:
                    param_type = "integer"
                elif annotation == float:
                    param_type = "number"
                elif getattr(annotation, "__origin__", None) in (list, List):
                    param_type = "array"
                
                prompt_args.append(
                    PromptArgument(
                        name=param_name,
                        description=f"The {param_name} parameter",
                        type=param_type,
                        required=required
                    )
                )
            
            # Generate or use provided description
            prompt_description = description or func.__doc__ or f"Generate {prompt_name} prompt"
            # Clean up the description
            prompt_description = inspect.cleandoc(prompt_description).split("\n")[0]
            
            # Create prompt
            prompt = Prompt(
                name=prompt_name,
                description=prompt_description,
                arguments=prompt_args
            )
            
            # Register the prompt
            cls._prompts[prompt_name] = {
                "prompt": prompt,
                "function": func
            }
            
            return func
        
        return decorator
    
    @classmethod
    def get_prompts(cls) -> List[Prompt]:
        """Get all registered prompts."""
        return [info["prompt"] for info in cls._prompts.values()]
    
    @classmethod
    def get_prompt_function(cls, name: str) -> Optional[Callable]:
        """Get the function for a registered prompt by name."""
        prompt_info = cls._prompts.get(name)
        if prompt_info:
            return prompt_info["function"]
        return None

# =========================================================================
# Prompt Implementation
# =========================================================================

class PromptGenerator:
    """Generator for HDF5-related prompts."""
    
    @staticmethod
    @PromptRegistry.register(
        description="Generate a comprehensive summary of an HDF5 file structure and contents"
    )
    @log_prompt_generation
    @measure_prompt_performance
    @cache_prompt_result(lambda file_path, include_attributes=False: f"{file_path}:{include_attributes}")
    async def summarize_file(file_path: str, include_attributes: bool = False) -> GetPromptResult:
        """
        Generate a summary of an HDF5 file.
        
        Args:
            file_path: Path to the HDF5 file
            include_attributes: Whether to include attributes
            
        Returns:
            Prompt result with file summary
        """
        try:
            with HDF5Manager(file_path) as h5m:
                root_group = h5m.file["/"]
                
                # Get basic file info
                file_info = {
                    "filename": h5m.file.filename,
                    "mode": h5m.file.mode,
                    "userblock_size": h5m.file.userblock_size
                }
                
                # Get top-level groups and datasets
                top_level_items = {}
                for name, obj in root_group.items():
                    if isinstance(obj, h5py.Group):
                        top_level_items[name] = "Group"
                    elif isinstance(obj, h5py.Dataset):
                        top_level_items[name] = f"Dataset: {obj.shape}, {obj.dtype}"
                    else:
                        top_level_items[name] = type(obj).__name__
                
                # Include attributes if requested
                attrs = {}
                if include_attributes:
                    attrs = dict(root_group.attrs)
                    # Convert numpy values to Python native types
                    for key, value in attrs.items():
                        if hasattr(value, "tolist"):
                            attrs[key] = value.tolist()
                        else:
                            attrs[key] = str(value)
                
                # Build the summary prompt
                combined_content = (
                    "System: You are an HDF5 file analysis assistant. Analyze the provided HDF5 file "
                    "summary and provide insights about its structure and contents.\n\n"
                    f"User: Please analyze this HDF5 file: {file_path}\n\n"
                    f"File Information:\n{json.dumps(file_info, indent=2)}\n\n"
                    f"Top-level Contents:\n{json.dumps(top_level_items, indent=2)}\n\n"
                    + (f"Root Attributes:\n{json.dumps(attrs, indent=2)}" if include_attributes else "")
                )
                
                return GetPromptResult(
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(type="text", text=combined_content)
                        )
                    ]
                )
                
        except Exception as e:
            logger.error(f"Error generating file summary: {e}")
            return GetPromptResult(
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text", 
                            text=f"Error generating HDF5 file summary: An error occurred while processing the file {file_path}: {str(e)}"
                        )
                    )
                ]
            )
    
    @staticmethod
    @PromptRegistry.register(
        description="Generate a detailed summary of an HDF5 group with its contents and structure"
    )
    @log_prompt_generation
    @measure_prompt_performance
    @cache_prompt_result(lambda file_path, group_path, recursive=False: f"{file_path}:{group_path}:{recursive}")
    async def summarize_group(
        file_path: str, 
        group_path: str, 
        recursive: bool = False
    ) -> GetPromptResult:
        """
        Generate a summary of an HDF5 group.
        
        Args:
            file_path: Path to the HDF5 file
            group_path: Path to the group within the file
            recursive: Whether to recursively summarize subgroups
            
        Returns:
            Prompt result with group summary
        """
        try:
            with HDF5Manager(file_path) as h5m:
                # Access the group
                group = h5m.file[group_path]
                
                if not isinstance(group, h5py.Group):
                    raise ValueError(f"Path {group_path} does not point to a group")
                
                # Get group info
                group_info = h5m.get_object_info(group)
                
                # Handle recursive exploration
                if recursive:
                    items = {}
                    
                    def visit_item(name, obj):
                        if isinstance(obj, h5py.Group):
                            items[name] = "Group"
                        elif isinstance(obj, h5py.Dataset):
                            items[name] = f"Dataset: {obj.shape}, {obj.dtype}"
                        else:
                            items[name] = type(obj).__name__
                    
                    group.visititems(visit_item)
                    group_info["contents"] = items
                
                # Build the summary prompt
                combined_content = (
                    "System: You are an HDF5 group analysis assistant. Analyze the provided HDF5 group "
                    "summary and provide insights about its structure and contents.\n\n"
                    f"User: Please analyze this HDF5 group: {group_path} in file {file_path}\n\n"
                    f"Group Information:\n{json.dumps(group_info, indent=2)}\n\n"
                )
                
                return GetPromptResult(
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(type="text", text=combined_content)
                        )
                    ]
                )
                
        except Exception as e:
            logger.error(f"Error generating group summary: {e}")
            return GetPromptResult(
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text="Error generating HDF5 group summary")
                    ),
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=f"Error: An error occurred while processing the group {group_path} in file {file_path}: {str(e)}")
                    )
                ]
            )
    
    @staticmethod
    @PromptRegistry.register(
        description="Generate a comprehensive analysis of an HDF5 dataset with optional statistical information"
    )
    @log_prompt_generation
    @measure_prompt_performance
    @cache_prompt_result(lambda file_path, dataset_path, include_statistics=False: f"{file_path}:{dataset_path}:{include_statistics}")
    async def summarize_dataset(
        file_path: str, 
        dataset_path: str, 
        include_statistics: bool = False
    ) -> GetPromptResult:
        """
        Generate a summary of an HDF5 dataset.
        
        Args:
            file_path: Path to the HDF5 file
            dataset_path: Path to the dataset within the file
            include_statistics: Whether to include basic statistics in the summary
            
        Returns:
            Prompt result with dataset summary
        """
        try:
            with HDF5Manager(file_path) as h5m:
                # Access the dataset
                dataset = h5m.file[dataset_path]
                
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError(f"Path {dataset_path} does not point to a dataset")
                
                # Get dataset info
                dataset_info = {
                    "shape": dataset.shape,
                    "dtype": str(dataset.dtype),
                    "size": dataset.size,
                    "nbytes": dataset.nbytes,
                    "chunks": dataset.chunks,
                    "compression": dataset.compression,
                    "compression_opts": dataset.compression_opts,
                    "scaleoffset": dataset.scaleoffset,
                    "shuffle": dataset.shuffle,
                    "fletcher32": dataset.fletcher32,
                    "fillvalue": str(dataset.fillvalue) if dataset.fillvalue is not None else None,
                }
                
                # Convert any numpy values to Python native types
                for key, value in dataset_info.items():
                    if hasattr(value, "tolist"):
                        dataset_info[key] = value.tolist()
                
                # Include attributes
                attrs = {}
                for key, value in dataset.attrs.items():
                    if hasattr(value, "tolist"):
                        attrs[key] = value.tolist()
                    else:
                        attrs[key] = str(value)
                
                dataset_info["attributes"] = attrs
                
                # Include statistics if requested
                if include_statistics and dataset.dtype.kind in "iuf":  # Integer, unsigned integer, or float
                    # Compute statistics safely on potentially large datasets
                    try:
                        # Use dask or direct access for small datasets
                        if dataset.size < 1_000_000:  # 1M elements
                            data = dataset[()]
                            stats = {
                                "min": np.min(data).item(),
                                "max": np.max(data).item(),
                                "mean": np.mean(data).item(),
                                "std": np.std(data).item()
                            }
                        else:
                            # Sample the dataset for large arrays
                            import random
                            
                            # Get total dimensions
                            dims = len(dataset.shape)
                            sample_size = min(1000, dataset.size)
                            
                            # Create random slices for sampling
                            samples = []
                            for _ in range(sample_size):
                                idx = tuple(random.randint(0, s-1) for s in dataset.shape)
                                samples.append(dataset[idx])
                            
                            data = np.array(samples)
                            stats = {
                                "min": np.min(data).item(),
                                "max": np.max(data).item(),
                                "mean": np.mean(data).item(),
                                "std": np.std(data).item(),
                                "note": "Statistics based on random sampling due to large dataset size"
                            }
                        
                        dataset_info["statistics"] = stats
                    except Exception as stats_e:
                        logger.warning(f"Failed to compute statistics: {stats_e}")
                        dataset_info["statistics"] = {"error": str(stats_e)}
                
                # Build the summary prompt
                combined_content = (
                    "System: You are an HDF5 dataset analysis assistant. Analyze the provided HDF5 dataset "
                    "summary and provide insights about its structure, contents, and statistical properties.\n\n"
                    f"User: Please analyze this HDF5 dataset: {dataset_path} in file {file_path}\n\n"
                    f"Dataset Information:\n{json.dumps(dataset_info, indent=2)}\n\n"
                )
                
                return GetPromptResult(
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(type="text", text=combined_content)
                        )
                    ]
                )
                
        except Exception as e:
            logger.error(f"Error generating dataset summary: {e}")
            return GetPromptResult(
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text="Error generating HDF5 dataset summary")
                    ),
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=f"Error: An error occurred while processing the dataset {dataset_path} in file {file_path}: {str(e)}")
                    )
                ]
            )
    
    @staticmethod
    @PromptRegistry.register(
        description="Generate a prompt for comprehensive multi-file HDF5 analysis and performance optimization"
    )
    @log_prompt_generation
    @measure_prompt_performance
    @cache_prompt_result(lambda directory, analysis_type="overview": f"{directory}:{analysis_type}")
    async def analyze_hdf5_collection(
        directory: str,
        analysis_type: str = "overview"
    ) -> GetPromptResult:
        """
        Generate a prompt for analyzing a collection of HDF5 files.
        
        Args:
            directory: Directory containing HDF5 files
            analysis_type: Type of analysis (overview, performance, structure, statistical)
            
        Returns:
            Prompt result for HDF5 collection analysis
        """
        try:
            # Build analysis prompt based on type
            system_content = "You are an advanced HDF5 data analysis assistant specializing in multi-file operations and performance optimization."
            
            if analysis_type == "overview":
                user_content = f"""I need to analyze a collection of HDF5 files in directory: {directory}

Please help me understand:
1. The overall structure and organization of these files
2. Key datasets and their characteristics
3. Potential relationships between files
4. Performance considerations for processing this collection

Suggested approach:
- Use hdf5_parallel_scan('{directory}') for fast multi-file scanning
- Use analyze_dataset_structure() for detailed structure analysis
- Use find_similar_datasets() to identify patterns across files"""
                
            elif analysis_type == "performance":
                user_content = f"""I need to optimize the performance of working with HDF5 files in directory: {directory}

Please help me:
1. Identify potential I/O bottlenecks
2. Recommend optimal access patterns
3. Suggest parallel processing strategies
4. Optimize memory usage for large datasets

Suggested tools to use:
- identify_io_bottlenecks() to find performance issues
- optimize_access_pattern() for access optimization
- hdf5_stream_data() for large dataset processing
- hdf5_batch_read() for multiple dataset operations"""
                
            elif analysis_type == "structure":
                user_content = f"""I need to understand the structural organization of HDF5 files in directory: {directory}

Please help me:
1. Map the hierarchical structure of the data
2. Identify common patterns and conventions
3. Find related datasets across files
4. Understand data organization principles

Recommended analysis approach:
- Start with hdf5_parallel_scan('{directory}') for overview
- Use analyze_dataset_structure() for each file
- Use find_similar_datasets() to identify patterns
- Use suggest_next_exploration() for guided discovery"""
                
            elif analysis_type == "statistical":
                user_content = f"""I need to perform statistical analysis across HDF5 files in directory: {directory}

Please help me:
1. Compute comprehensive statistics across datasets
2. Identify statistical patterns and outliers
3. Compare datasets statistically
4. Generate summary statistics for the collection

Suggested statistical workflow:
- Use hdf5_parallel_scan('{directory}') to identify datasets
- Use hdf5_aggregate_stats() for comprehensive statistics
- Use hdf5_batch_read() for multi-dataset comparisons
- Use hdf5_stream_data() for large dataset statistics"""
                
            else:
                user_content = f"""I need to analyze HDF5 files in directory: {directory}

Please provide general guidance on:
1. How to efficiently explore this data collection
2. What analysis tools would be most appropriate
3. How to optimize performance for my use case
4. Best practices for working with this data structure"""
            
            return GetPromptResult(
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=f"System: {system_content}\n\nUser: {user_content}")
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Error generating collection analysis prompt: {e}")
            return GetPromptResult(
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text="Error generating HDF5 collection analysis prompt")
                    ),
                    PromptMessage(
                        role="user",
                        content=f"An error occurred while generating analysis prompt for directory {directory}: {str(e)}"
                    )
                ]
            )
    
    @staticmethod
    @PromptRegistry.register(
        description="Generate a prompt for optimizing HDF5 data access patterns and performance"
    )
    @log_prompt_generation
    @measure_prompt_performance
    @cache_prompt_result(lambda file_path, use_case, dataset_paths=None: f"{file_path}:{use_case}:{str(dataset_paths)}")
    async def optimize_hdf5_workflow(
        file_path: str,
        use_case: str,
        dataset_paths: List[str] = None
    ) -> GetPromptResult:
        """
        Generate a prompt for optimizing HDF5 workflows.
        
        Args:
            file_path: Path to the HDF5 file
            use_case: The intended use case (streaming, random_access, batch_processing, statistics)
            dataset_paths: Optional list of specific dataset paths to focus on
            
        Returns:
            Prompt result for workflow optimization
        """
        try:
            system_content = "You are an HDF5 performance optimization specialist. Help users choose the most efficient tools and strategies for their specific workflow needs."
            
            datasets_info = ""
            if dataset_paths:
                datasets_info = f"\n\nFocus on these specific datasets: {', '.join(dataset_paths)}"
            
            if use_case == "streaming":
                user_content = f"""I need to process large HDF5 datasets in file: {file_path} using streaming approaches.{datasets_info}

Please help me optimize for:
1. Memory-efficient processing of large datasets
2. Chunk-based processing strategies
3. Real-time data processing capabilities
4. Progress monitoring for long operations

Recommended tools and approach:
- Use hdf5_stream_data() for memory-efficient processing
- Use optimize_access_pattern(dataset_path, 'sequential') for guidance
- Consider chunked processing with appropriate chunk sizes
- Monitor performance with built-in measurement tools"""
                
            elif use_case == "random_access":
                user_content = f"""I need efficient random access to HDF5 data in file: {file_path}.{datasets_info}

Please help me optimize for:
1. Fast random element access
2. Efficient partial dataset reading
3. Optimal chunking strategies
4. Cache-friendly access patterns

Recommended optimization strategy:
- Use optimize_access_pattern(dataset_path, 'random') for guidance
- Use read_partial_dataset() with specific slices
- Consider identify_io_bottlenecks() to find access issues
- Optimize chunking configuration for your access patterns"""
                
            elif use_case == "batch_processing":
                user_content = f"""I need to process multiple datasets in batch from file: {file_path}.{datasets_info}

Please help me optimize for:
1. Parallel processing of multiple datasets
2. Efficient batch operations
3. Resource utilization optimization
4. Throughput maximization

Recommended batch processing approach:
- Use hdf5_batch_read() for parallel dataset processing
- Use hdf5_aggregate_stats() for batch statistical operations
- Consider hdf5_parallel_scan() for multi-file batch processing
- Use optimize_access_pattern(dataset_path, 'batch') for guidance"""
                
            elif use_case == "statistics":
                user_content = f"""I need to perform comprehensive statistical analysis on file: {file_path}.{datasets_info}

Please help me optimize for:
1. Efficient statistical computation
2. Cross-dataset statistical analysis
3. Large dataset statistical sampling
4. Performance-optimized statistics

Recommended statistical workflow:
- Use hdf5_aggregate_stats() for comprehensive statistics
- Use hdf5_stream_data() for large dataset statistics
- Consider sampling strategies for very large datasets
- Use batch operations for multi-dataset statistics"""
                
            else:
                user_content = f"""I need to optimize my workflow for HDF5 file: {file_path}.{datasets_info}

My use case is: {use_case}

Please provide guidance on:
1. The most efficient tools for my specific needs
2. Performance optimization strategies
3. Best practices for my workflow
4. Potential bottlenecks to avoid

General optimization recommendations:
- Start with analyze_dataset_structure() to understand your data
- Use identify_io_bottlenecks() to find performance issues
- Choose appropriate tools based on your access patterns
- Consider memory and performance trade-offs"""
            
            return GetPromptResult(
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=f"System: {system_content}\n\nUser: {user_content}")
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Error generating workflow optimization prompt: {e}")
            return GetPromptResult(
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text="Error generating HDF5 workflow optimization prompt")
                    ),
                    PromptMessage(
                        role="user",
                        content=f"An error occurred while generating optimization prompt for {file_path}: {str(e)}"
                    )
                ]
            )
    
    @staticmethod
    @PromptRegistry.register(
        description="Generate a prompt for guided HDF5 data exploration and discovery"
    )
    @log_prompt_generation
    @measure_prompt_performance
    async def explore_hdf5_data(
        file_path: str,
        exploration_goal: str = "general"
    ) -> GetPromptResult:
        """
        Generate a prompt for guided HDF5 data exploration.
        
        Args:
            file_path: Path to the HDF5 file to explore
            exploration_goal: The exploration goal (general, patterns, relationships, structure)
            
        Returns:
            Prompt result for guided data exploration
        """
        try:
            system_content = "You are a data exploration guide specializing in HDF5 files. Help users discover insights and understand their data through systematic exploration."
            
            if exploration_goal == "patterns":
                user_content = f"""I want to discover patterns and relationships in HDF5 file: {file_path}

Please guide me through:
1. Identifying similar datasets and structures
2. Finding data patterns across different groups
3. Understanding data relationships and dependencies
4. Discovering hidden structures in the data

Suggested exploration workflow:
- Start with analyze_dataset_structure() for overall understanding
- Use find_similar_datasets() to identify patterns
- Use suggest_next_exploration() for guided discovery
- Analyze interesting findings with appropriate tools"""
                
            elif exploration_goal == "relationships":
                user_content = f"""I want to understand the relationships between datasets in HDF5 file: {file_path}

Please help me explore:
1. How different datasets relate to each other
2. Dependencies between data elements
3. Hierarchical relationships in the structure
4. Cross-references and data connections

Recommended relationship analysis:
- Use analyze_dataset_structure() to map the hierarchy
- Use find_similar_datasets() to identify related data
- Compare dataset characteristics and patterns
- Look for naming conventions and organizational principles"""
                
            elif exploration_goal == "structure":
                user_content = f"""I want to understand the organizational structure of HDF5 file: {file_path}

Please guide me through:
1. Understanding the hierarchical organization
2. Identifying the purpose of different groups and datasets
3. Understanding the data architecture principles
4. Mapping the overall data layout

Structural exploration approach:
- Begin with analyze_dataset_structure() for overview
- Explore each major group systematically
- Use suggest_next_exploration() for guided navigation
- Document the organizational principles you discover"""
                
            else:
                user_content = f"""I want to explore and understand HDF5 file: {file_path}

Please guide me through systematic exploration:
1. Getting an overview of the data structure
2. Identifying the most interesting datasets
3. Understanding the data organization
4. Finding insights and patterns

Recommended exploration strategy:
- Start with analyze_dataset_structure() for overview
- Use suggest_next_exploration() for guided discovery
- Investigate interesting datasets with appropriate tools
- Look for patterns with find_similar_datasets()"""
            
            return GetPromptResult(
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=f"System: {system_content}\n\nUser: {user_content}")
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Error generating exploration prompt: {e}")
            return GetPromptResult(
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text="Error generating HDF5 exploration prompt")
                    ),
                    PromptMessage(
                        role="user",
                        content=f"An error occurred while generating exploration prompt for {file_path}: {str(e)}"
                    )
                ]
            )

# Function to get all available prompts
def get_available_prompts() -> List[Prompt]:
    """Get all available prompts."""
    return PromptRegistry.get_prompts()

# Function to clear the prompt cache
def clear_prompt_cache() -> None:
    """Clear the prompt cache."""
    for category in prompt_cache:
        prompt_cache[category].clear()
    logger.info("Prompt cache cleared")

# Function to generate a prompt by name
async def generate_prompt(prompt_name: str, **kwargs) -> GetPromptResult:
    """
    Generate a prompt by name with the given arguments.
    
    Args:
        prompt_name: Name of the prompt to generate
        **kwargs: Arguments for the prompt
        
    Returns:
        GetPromptResult with the generated prompt
    """
    func = PromptRegistry.get_prompt_function(prompt_name)
    if func:
        return await func(**kwargs)
    
    logger.error(f"Unknown prompt: {prompt_name}")
    return GetPromptResult(
        messages=[
            PromptMessage(
                role="assistant",
                content=TextContent(type="text", text="Error generating prompt")
            ),
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=f"Unknown prompt: {prompt_name}")
            )
        ]
    )
