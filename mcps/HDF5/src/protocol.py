"""
Protocol definitions for HDF5 MCP server-client communication.
Enhanced with batch operations and comprehensive error handling.
"""
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from datetime import datetime

class ServerState(str, Enum):
    """Server state enumeration."""
    INITIALIZING = "initializing"
    READY = "ready"
    SCANNING = "scanning"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"

class ServerStatus(BaseModel):
    """Server status information."""
    state: ServerState
    session_id: str
    uptime: float
    active_connections: int
    files_monitored: int
    error_message: Optional[str] = None
    active_operations: int = Field(default=0, description="Number of active operations")
    batch_operations: int = Field(default=0, description="Number of active batch operations")
    resource_usage: Dict[str, float] = Field(
        default_factory=dict,
        description="Resource usage metrics (CPU, memory, etc.)"
    )

class MessageType(str, Enum):
    """Types of messages that can be exchanged."""
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"
    COMMAND = "command"
    COMMAND_RESPONSE = "command_response"
    BATCH_REQUEST = "batch_request"
    BATCH_RESPONSE = "batch_response"
    ERROR = "error"
    NOTIFICATION = "notification"
    CANCEL_REQUEST = "cancel_request"
    CANCEL_RESPONSE = "cancel_response"

class ErrorSeverity(str, Enum):
    """Error severity levels."""
    FATAL = "fatal"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ErrorResponse(BaseModel):
    """Enhanced error response format."""
    code: str
    message: str
    severity: ErrorSeverity
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    recoverable: bool = True
    suggestion: Optional[str] = None
    error_id: str = Field(default_factory=lambda: f"err_{datetime.now().timestamp()}")

class BatchOperation(BaseModel):
    """Batch operation specification."""
    operation_id: str
    command: str
    parameters: Dict[str, Any]
    timeout: Optional[float] = None
    dependencies: List[str] = Field(default_factory=list)
    rollback_on_error: bool = True

class BatchRequest(BaseModel):
    """Batch request format."""
    batch_id: str
    operations: List[BatchOperation]
    parallel: bool = False
    timeout: Optional[float] = None
    abort_on_error: bool = True

class BatchResponse(BaseModel):
    """Batch response format."""
    batch_id: str
    results: Dict[str, Any]
    failed_operations: List[str] = Field(default_factory=list)
    execution_time: float
    rollback_performed: bool = False

class CommandRequest(BaseModel):
    """Enhanced command request format."""
    command: str
    parameters: Dict[str, Any]
    timeout: Optional[float] = None
    priority: Optional[int] = None
    idempotency_key: Optional[str] = None
    retry_policy: Optional[Dict[str, Any]] = None

class CommandResponse(BaseModel):
    """Enhanced command response format."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[ErrorResponse] = None
    execution_time: float = Field(default=0.0)
    resource_usage: Optional[Dict[str, float]] = None

class CancellationRequest(BaseModel):
    """Request to cancel an operation."""
    operation_id: str
    force: bool = False
    timeout: Optional[float] = None

class CancellationResponse(BaseModel):
    """Response to a cancellation request."""
    operation_id: str
    success: bool
    error: Optional[ErrorResponse] = None
    state: str

class NotificationType(str, Enum):
    """Enhanced notification types."""
    FILE_CHANGE = "file_change"
    SERVER_STATE_CHANGE = "server_state_change"
    PERFORMANCE_ALERT = "performance_alert"
    RESOURCE_UPDATE = "resource_update"
    BATCH_PROGRESS = "batch_progress"
    OPERATION_TIMEOUT = "operation_timeout"
    RESOURCE_PRESSURE = "resource_pressure"
    SECURITY_ALERT = "security_alert"

class NotificationSeverity(str, Enum):
    """Notification severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class Notification(BaseModel):
    """Enhanced notification message format."""
    type: NotificationType
    severity: NotificationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str
    requires_action: bool = False
    action_deadline: Optional[datetime] = None

class Message(BaseModel):
    """Enhanced base message format."""
    type: MessageType
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str
    correlation_id: Optional[str] = None
    payload: Union[
        Dict[str, Any],
        CommandRequest,
        CommandResponse,
        BatchRequest,
        BatchResponse,
        ErrorResponse,
        Notification,
        CancellationRequest,
        CancellationResponse
    ]
    metadata: Dict[str, Any] = Field(default_factory=dict)

# API Endpoints
API_ENDPOINTS = {
    "status": "/api/status",
    "command": "/api/command",
    "batch": "/api/batch",
    "resources": "/api/resources",
    "tools": "/api/tools",
    "prompts": "/api/prompts",
    "cancel": "/api/cancel"
}

# Standard error codes with descriptions
class ErrorCodes:
    """Comprehensive error codes with descriptions."""
    
    # Request errors
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_PARAMETERS = "INVALID_PARAMETERS"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    MALFORMED_REQUEST = "MALFORMED_REQUEST"
    
    # Authentication/Authorization errors
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    
    # Resource errors
    NOT_FOUND = "NOT_FOUND"
    RESOURCE_BUSY = "RESOURCE_BUSY"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    RESOURCE_LOCKED = "RESOURCE_LOCKED"
    
    # Operational errors
    TIMEOUT = "TIMEOUT"
    OPERATION_CANCELLED = "OPERATION_CANCELLED"
    BATCH_FAILED = "BATCH_FAILED"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    
    # System errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    MAINTENANCE_MODE = "MAINTENANCE_MODE"
    
    # Data errors
    DATA_CORRUPTION = "DATA_CORRUPTION"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    
    # Network errors
    NETWORK_ERROR = "NETWORK_ERROR"
    CONNECTION_LOST = "CONNECTION_LOST"
    
    # Security errors
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    @staticmethod
    def get_description(code: str) -> str:
        """Get a human-readable description for an error code."""
        descriptions = {
            ErrorCodes.INVALID_REQUEST: "The request was invalid or malformed.",
            ErrorCodes.INVALID_PARAMETERS: "One or more parameters were invalid.",
            ErrorCodes.UNSUPPORTED_OPERATION: "The requested operation is not supported.",
            ErrorCodes.UNAUTHORIZED: "Authentication is required for this operation.",
            ErrorCodes.FORBIDDEN: "The authenticated user lacks permission for this operation.",
            ErrorCodes.NOT_FOUND: "The requested resource was not found.",
            ErrorCodes.RESOURCE_BUSY: "The requested resource is currently busy.",
            ErrorCodes.TIMEOUT: "The operation timed out.",
            ErrorCodes.INTERNAL_ERROR: "An internal server error occurred.",
            # Add more descriptions as needed
        }
        return descriptions.get(code, "Unknown error") 