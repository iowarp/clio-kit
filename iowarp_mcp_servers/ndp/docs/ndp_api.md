# API Search Documentation

This document provides comprehensive information about the search functionality in the API, specifically covering the `/organizations` and `/search` endpoints.

## Organizations Endpoint

### GET /organization - List Organizations

**Purpose**: Retrieve a list of all organizations with optional filtering.

**Method**: `GET`

**Parameters**:
- `name` (query, optional): Filter organizations by name
- `server` (query, optional): Specify server to list organizations from
  - Options: `local`, `global`, `pre_ckan`
  - Default: `global`

**Response**:
- **200 OK**: Returns an array of organization names
- **400 Bad Request**: Error message explaining the bad request
- **422 Validation Error**: Validation error details

**Example Request**:
```
GET /organization?name=example&server=global
```

**Example Response**:
```json
["org1", "org2", "org3"]
```

### POST /organization - Create Organization

**Purpose**: Create a new organization.

**Method**: `POST`

**Parameters**:
- `server` (query, optional): Specify server (`local` or `pre_ckan`, defaults to `local`)

**Request Body**:
```json
{
  "name": "example_org_name",           // Required: Unique organization name
  "title": "Example Organization Title", // Required: Organization title
  "description": "This is an example organization." // Optional: Description
}
```

**Response**:
- **201 Created**: Organization created successfully
- **400 Bad Request**: Organization name already exists
- **422 Validation Error**: Validation error details

**Example Response**:
```json
{
  "id": "305284e6-6338-4e13-b39b-e6efe9f1c45a",
  "message": "Organization created successfully"
}
```

## Search Endpoint

### GET /search - Search Datasets by Terms

**Purpose**: Search CKAN datasets using a list of search terms.

**Method**: `GET`

**Parameters**:
- `terms` (query, required): Array of search terms
- `keys` (query, optional): Array of keys corresponding to each term (use `null` for global search)
- `server` (query, optional): Server to search on
  - Options: `local`, `global`
  - Default: `global`
  - Note: If 'local' CKAN is disabled, it cannot be used

**Response**:
- **200 OK**: Returns array of matching datasets
- **400 Bad Request**: Error message explaining the bad request
- **422 Unprocessable Entity**: Validation error details

**Example Request**:
```
GET /search?terms=climate&terms=data&server=global
```

### POST /search - Advanced Dataset Search

**Purpose**: Search datasets using various parameters with more granular control.

**Method**: `POST`

**Request Body Schema**:
All fields are optional and can be used in combination:

#### Common Registration-Matching Parameters:
- `dataset_name`: The name of the dataset
- `dataset_title`: The title of the dataset
- `owner_org`: The name of the organization
- `resource_url`: The URL of the dataset resource
- `resource_name`: The name of the dataset resource
- `dataset_description`: The description of the dataset
- `resource_description`: The description of the resource
- `resource_format`: The format of the dataset resource

#### User-Defined Search Parameters:
- `search_term`: Comma-separated list of terms to search across all fields
- `filter_list`: Array of field filters in the form `key:value`
- `timestamp`: Filter on the timestamp field of results
- `server`: Server selection (`local`, `global`, or `pre_ckan`, defaults to `global`)

**Example Request Body**:
```json
{
  "dataset_name": "climate_data",
  "resource_format": "CSV",
  "search_term": "temperature,weather",
  "filter_list": ["type:sensor", "location:europe"],
  "server": "global"
}
```

**Response**:
- **200 OK**: Returns array of matching datasets
- **400 Bad Request**: Error occurred during search
- **422 Validation Error**: Request validation failed

## Response Schema

Both search endpoints return an array of `DataSourceResponse` objects:

```json
[
  {
    "id": "12345678-abcd-efgh-ijkl-1234567890ab",
    "name": "example_dataset_name",
    "title": "Example Dataset Title",
    "owner_org": "example_org_name",
    "notes": "This is an example dataset.",
    "resources": [
      {
        "id": "abcd1234-efgh5678-ijkl9012",
        "url": "http://example.com/resource",
        "name": "Example Resource Name",
        "description": "This is an example resource.",
        "format": "CSV"
      }
    ],
    "extras": {
      "key1": "value1",
      "key2": "value2",
      "mapping": {
        "field1": "qeadw2",
        "field2": "gw4aw34",
        "time": "gw4aw34"
      },
      "processing": {
        "data_key": "",
        "info_key": "key_with_info"
      }
    }
  }
]
```

### DataSource Response Fields:
- `id` (string, required): Unique dataset identifier
- `name` (string, required): Unique dataset name
- `title` (string, required): Dataset title
- `owner_org` (string, optional): Organization ID that owns the dataset
- `notes` (string, optional): Dataset description
- `resources` (array, required): List of associated resources
- `extras` (object, optional): Additional metadata

### Resource Object Fields:
- `id` (string, required): Unique resource identifier
- `url` (string, optional): Resource URL
- `name` (string, required): Resource name
- `description` (string, optional): Resource description
- `format` (string, optional): Resource format (e.g., CSV, JSON, etc.)

## Search Examples

### 1. Simple term search:
```bash
curl -X GET "http://155.101.6.191:8003/search?terms=climate&server=global" \
  -H "Authorization: Bearer <token>"
```

### 2. Advanced search with multiple criteria:
```bash
curl -X POST "http://155.101.6.191:8003/search" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "dataset_name": "sensor_data",
    "resource_format": "CSV",
    "search_term": "temperature,humidity",
    "server": "global"
  }'
```

### 3. List organizations with filtering:
```bash
curl -X GET "http://155.101.6.191:8003/organization?name=research&server=global" \
  -H "Authorization: Bearer <token>"
```

## Server Options

The API supports multiple server configurations:
- **local**: Search in local CKAN instance (may be disabled)
- **global**: Search in global CKAN instance (default for most endpoints)
- **pre_ckan**: Search in pre-production CKAN instance (available for some endpoints)

Choose the appropriate server based on your data access requirements and instance availability.
