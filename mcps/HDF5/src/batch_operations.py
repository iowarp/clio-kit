"""
Batch operations handler for HDF5 files.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
import h5py
import numpy as np
from pathlib import Path

from src.hdf5_mcp_server.protocol import (
    BatchOperation,
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    ErrorCodes,
    ErrorSeverity
)
from src.hdf5_mcp_server.task_queue import TaskPriority

logger = logging.getLogger(__name__)

class BatchOperationError(Exception):
    """Exception raised for batch operation errors."""
    def __init__(self, message: str, operation_id: str, code: str = ErrorCodes.BATCH_FAILED):
        self.message = message
        self.operation_id = operation_id
        self.code = code
        super().__init__(message)

class BatchTransaction:
    """Manages a transaction-like batch of HDF5 operations."""
    
    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self.operations: List[BatchOperation] = []
        self.completed_ops: Set[str] = set()
        self.failed_ops: Set[str] = set()
        self.start_time = datetime.now()
        self.results: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._snapshots: Dict[str, Dict[str, Any]] = {}
    
    async def add_operation(self, operation: BatchOperation):
        """Add an operation to the batch."""
        self.operations.append(operation)
        
        # Create locks for any new files referenced in the operation
        files = self._extract_file_paths(operation.parameters)
        for file_path in files:
            if file_path not in self._locks:
                self._locks[file_path] = asyncio.Lock()
    
    def _extract_file_paths(self, parameters: Dict[str, Any]) -> Set[str]:
        """Extract HDF5 file paths from operation parameters."""
        files = set()
        if isinstance(parameters, dict):
            for key, value in parameters.items():
                if key in ('file_path', 'path', 'source', 'target') and isinstance(value, (str, Path)):
                    path = str(value)
                    if path.endswith('.h5'):
                        files.add(path)
                elif isinstance(value, (dict, list)):
                    files.update(self._extract_file_paths(value))
        elif isinstance(parameters, list):
            for item in parameters:
                if isinstance(item, (dict, list)):
                    files.update(self._extract_file_paths(item))
        return files
    
    async def _create_snapshot(self, file_path: str) -> Dict[str, Any]:
        """Create a snapshot of HDF5 file state for rollback."""
        snapshot = {}
        try:
            with h5py.File(file_path, 'r') as f:
                def capture_attrs(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        snapshot[name] = {
                            'data': obj[()],
                            'attrs': dict(obj.attrs)
                        }
                    elif isinstance(obj, h5py.Group):
                        snapshot[name] = {
                            'attrs': dict(obj.attrs)
                        }
                f.visititems(capture_attrs)
            return snapshot
        except Exception as e:
            logger.error(f"Failed to create snapshot for {file_path}: {e}")
            raise BatchOperationError(
                f"Failed to create snapshot: {str(e)}",
                "snapshot_creation"
            )
    
    async def _restore_snapshot(self, file_path: str, snapshot: Dict[str, Any]):
        """Restore HDF5 file state from snapshot."""
        try:
            with h5py.File(file_path, 'w') as f:
                for name, content in snapshot.items():
                    if 'data' in content:  # Dataset
                        dataset = f.create_dataset(name, data=content['data'])
                        for key, value in content['attrs'].items():
                            dataset.attrs[key] = value
                    else:  # Group
                        if name != '/':
                            group = f.create_group(name)
                            for key, value in content['attrs'].items():
                                group.attrs[key] = value
        except Exception as e:
            logger.error(f"Failed to restore snapshot for {file_path}: {e}")
            raise BatchOperationError(
                f"Failed to restore snapshot: {str(e)}",
                "snapshot_restoration"
            )
    
    async def prepare(self):
        """Prepare the batch transaction."""
        # Create snapshots for all files
        for file_path in self._locks.keys():
            self._snapshots[file_path] = await self._create_snapshot(file_path)
    
    async def execute(self, executor) -> BatchResponse:
        """Execute the batch of operations."""
        execution_start = datetime.now()
        rollback_performed = False
        
        try:
            # Sort operations by dependencies
            sorted_ops = self._sort_operations()
            
            # Execute operations
            for operation in sorted_ops:
                try:
                    # Acquire locks for files used in this operation
                    files = self._extract_file_paths(operation.parameters)
                    locks = [self._locks[f] for f in files]
                    
                    async with asyncio.AsyncExitStack() as stack:
                        for lock in locks:
                            await stack.enter_async_context(lock)
                        
                        # Execute operation
                        result = await executor.execute_operation(
                            operation,
                            priority=TaskPriority.BATCH
                        )
                        
                        self.results[operation.operation_id] = result
                        self.completed_ops.add(operation.operation_id)
                        
                except Exception as e:
                    logger.error(f"Operation {operation.operation_id} failed: {e}")
                    self.failed_ops.add(operation.operation_id)
                    
                    if operation.rollback_on_error:
                        await self.rollback()
                        rollback_performed = True
                        break
        
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            await self.rollback()
            rollback_performed = True
        
        execution_time = (datetime.now() - execution_start).total_seconds()
        
        return BatchResponse(
            batch_id=self.batch_id,
            results=self.results,
            failed_operations=list(self.failed_ops),
            execution_time=execution_time,
            rollback_performed=rollback_performed
        )
    
    async def rollback(self):
        """Rollback changes made by the batch."""
        logger.info(f"Rolling back batch {self.batch_id}")
        
        for file_path, snapshot in self._snapshots.items():
            try:
                async with self._locks[file_path]:
                    await self._restore_snapshot(file_path, snapshot)
            except Exception as e:
                logger.error(f"Failed to rollback {file_path}: {e}")
                # Continue with other rollbacks even if one fails
    
    def _sort_operations(self) -> List[BatchOperation]:
        """Sort operations based on dependencies."""
        # Build dependency graph
        graph: Dict[str, Set[str]] = {op.operation_id: set(op.dependencies) for op in self.operations}
        sorted_ops: List[str] = []
        temp_marks: Set[str] = set()
        perm_marks: Set[str] = set()
        
        def visit(op_id: str):
            if op_id in temp_marks:
                raise BatchOperationError(
                    "Circular dependency detected",
                    op_id,
                    ErrorCodes.INVALID_PARAMETERS
                )
            if op_id not in perm_marks:
                temp_marks.add(op_id)
                for dep in graph[op_id]:
                    visit(dep)
                temp_marks.remove(op_id)
                perm_marks.add(op_id)
                sorted_ops.insert(0, op_id)
        
        # Perform topological sort
        for op_id in graph:
            if op_id not in perm_marks:
                visit(op_id)
        
        # Map sorted operation IDs back to operations
        op_map = {op.operation_id: op for op in self.operations}
        return [op_map[op_id] for op_id in sorted_ops]

class BatchOperationHandler:
    """Handles execution of batch operations."""
    
    def __init__(self):
        self.active_batches: Dict[str, BatchTransaction] = {}
    
    async def execute_batch(self, request: BatchRequest, executor) -> BatchResponse:
        """Execute a batch request."""
        # Create and prepare transaction
        transaction = BatchTransaction(request.batch_id)
        self.active_batches[request.batch_id] = transaction
        
        try:
            # Add operations to transaction
            for operation in request.operations:
                await transaction.add_operation(operation)
            
            # Prepare transaction (create snapshots)
            await transaction.prepare()
            
            # Execute batch
            response = await transaction.execute(executor)
            
            return response
            
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            raise
        finally:
            # Clean up
            del self.active_batches[request.batch_id]
    
    def get_batch_status(self, batch_id: str) -> Tuple[int, int, List[str]]:
        """Get the status of a batch operation."""
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        batch = self.active_batches[batch_id]
        return (
            len(batch.completed_ops),
            len(batch.operations),
            list(batch.failed_ops)
        )
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch operation."""
        if batch_id not in self.active_batches:
            return False
        
        batch = self.active_batches[batch_id]
        await batch.rollback()
        del self.active_batches[batch_id]
        return True 