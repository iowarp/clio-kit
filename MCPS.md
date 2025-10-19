# IoWarp MCPs - Complete Reference

Comprehensive reference for all 15 MCP servers in the IoWarp Scientific Computing suite.

## Quick Summary

| Metric | Value |
|--------|-------|
| **Total MCPs** | 15 |
| **v2.0+ MCPs** | 1 (HDF5) |
| **v1.0 MCPs** | 14 |
| **MCP SDK MCPs** | 14 |
| **FastMCP MCPs** | 1 (NDP) |
| **Data I/O MCPs** | 4 |
| **HPC MCPs** | 2 |
| **Util MCPs** | 8 |

---

## MCP Status Matrix

### Legend
- 🟢 Production Ready
- 🟡 Stable
- ⭐ Recommended / Upgraded
- 🔧 Needs Modernization

### All MCPs Inventory

| # | Package | Ver | Status | SDK | Tools | Category | Entry Point | Key Feature |
|---|---------|-----|--------|-----|-------|----------|-------------|-------------|
| 1️⃣ | **HDF5** | v2.0 | 🟢⭐ | MCP | 25+ | Data I/O | `hdf5-mcp` | HPC-optimized with caching/streaming |
| 2️⃣ | **Adios** | v1.0 | 🟢 | MCP | 8+ | Data I/O | `adios-mcp` | ADIOS2 engine data access |
| 3️⃣ | **Parquet** | v1.0 | 🟢 | MCP | 5+ | Data I/O | `parquet-mcp` | Parquet file operations |
| 4️⃣ | **NDP** | v1.0 | 🟢 | FastMCP | 3+ | Data Protocol | `ndp-mcp` | CKAN dataset discovery |
| 5️⃣ | **Pandas** | v1.0 | 🟢 | MCP | 6+ | Data Analysis | `pandas-mcp` | CSV loading/filtering |
| 6️⃣ | **Slurm** | v1.0 | 🟢 | MCP | 4+ | HPC | `slurm-mcp` | Job submission/monitoring |
| 7️⃣ | **Jarvis** | v1.0 | 🟢 | MCP | 5+ | Workflow | `jarvis-mcp` | Pipeline management |
| 8️⃣ | **Plot** | v1.0 | 🟢 | MCP | 4+ | Visualization | `plot-mcp` | Matplotlib/Seaborn viz |
| 9️⃣ | **Arxiv** | v1.0 | 🟢 | MCP | 3+ | Research | `arxiv-mcp` | Research paper retrieval |
| 🔟 | **Darshan** | v1.0 | 🟢 | MCP | 4+ | Performance | `darshan-mcp` | I/O trace analysis |
| 1️⃣1️⃣ | **Chronolog** | v1.0 | 🟢 | MCP | 4+ | Logging | `chronolog-mcp` | Event logging system |
| 1️⃣2️⃣ | **Compression** | v1.0 | 🟢 | MCP | 3+ | Utilities | `compression-mcp` | Gzip compression |
| 1️⃣3️⃣ | **Parallel_Sort** | v1.0 | 🟢 | MCP | 2+ | Computing | `parallel-sort-mcp` | Large file sorting |
| 1️⃣4️⃣ | **lmod** | v1.0 | 🟢 | MCP | 3+ | Environment | `lmod-mcp` | Module environment mgmt |
| 1️⃣5️⃣ | **Node_Hardware** | v1.0 | 🟢 | MCP | 3+ | System | `node-hardware-mcp` | Hardware information |

---

## Detailed MCP Profiles

### 1️⃣ HDF5 MCP - HPC-Optimized Scientific Data Access

**Status**: 🟢⭐ Production Ready - Upgraded to v2.0
**SDK**: MCP 1.4+
**Tools**: 25+ comprehensive operations
**Entry Point**: `hdf5-mcp`

**Key Features**:
- 🚀 **Advanced Performance**: 100-1000x speedup via LRU caching
- 📊 **Parallel Processing**: 4-8x faster batch operations
- 🌊 **Streaming**: Handle 100GB+ files memory-efficiently
- 🔍 **Smart Discovery**: Find similar datasets, suggest exploration paths
- ⚙️ **Optimization**: Detect I/O bottlenecks, recommend patterns
- 🔗 **Multi-Transport**: Stdio + SSE/HTTP + Memory support

**Tool Categories**:
- **File**: open, close, get_filename, get_mode, get_by_path, list_keys, visit
- **Dataset**: read_full, read_partial, get_shape, get_dtype, get_size, get_chunks
- **Attribute**: read_attribute, list_attributes
- **Performance**: parallel_scan, batch_read, stream_data, aggregate_stats
- **Discovery**: analyze_structure, find_similar, suggest_exploration, identify_bottlenecks

**Performance Characteristics**:
```
Repeated Queries:    100-1000x faster (LRU cache)
Batch Operations:    4-8x faster (parallel)
Directory Scans:     3-5x faster (multi-threaded)
Large Files:         Unlimited (streaming)
```

**Installation**: `uvx iowarp-mcps hdf5`

**Documentation**:
- TOOLS.md - Complete tool reference
- ARCHITECTURE.md - System design
- EXAMPLES.md - Usage workflows
- TRANSPORTS.md - Transport configuration

**Next Steps for HDF5**: Monitor production usage, gather feedback on new features in v2.0

---

### 2️⃣ ADIOS MCP - ADIOS2 Engine Data Access

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 8+
**Entry Point**: `adios-mcp`

**Key Features**:
- BP5 format support
- Remote file access
- Metadata inspection
- Variable reading

**Installation**: `uvx iowarp-mcps adios`

---

### 3️⃣ Parquet MCP - Parquet File Operations

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 5+
**Entry Point**: `parquet-mcp`

**Key Features**:
- Column reading
- Schema inspection
- Metadata extraction
- Row filtering

**Installation**: `uvx iowarp-mcps parquet`

---

### 4️⃣ NDP MCP - National Data Platform Discovery

**Status**: 🟢 Production Ready
**SDK**: FastMCP 0.2+
**Tools**: 3
**Entry Point**: `ndp-mcp`

**Key Features**:
- 🌐 **Multi-Server Support**: Local, global, pre-production CKAN instances
- 🔍 **Organization Discovery**: List and filter available data sources
- 📦 **Comprehensive Search**: Term-based and field-specific searches
- 📋 **Dataset Details**: Complete metadata and resource information
- 🔄 **Robust Error Handling**: Retry logic, network resilience
- ⚙️ **Result Limiting**: Prevent context overflow with configurable limits

**Tools**:
1. `list_organizations` - Discover available organizations
2. `search_datasets` - Find datasets with advanced filtering
3. `get_dataset_details` - Retrieve complete dataset metadata

**SDK Note**: NDP intentionally uses **FastMCP** instead of standard MCP SDK. This is a proven, production-ready pattern that provides significant developer experience benefits.

**Installation**: `uvx iowarp-mcps ndp`

---

### 5️⃣ Pandas MCP - CSV Data Analysis

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 6+
**Entry Point**: `pandas-mcp`

**Key Features**:
- CSV loading
- Data filtering
- Statistical analysis
- Column operations

**Installation**: `uvx iowarp-mcps pandas`

---

### 6️⃣ Slurm MCP - HPC Job Management

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 4+
**Entry Point**: `slurm-mcp`

**Key Features**:
- Job submission
- Job monitoring
- Queue status
- Resource information

**Installation**: `uvx iowarp-mcps slurm`

---

### 7️⃣ Jarvis MCP - Data Pipeline Management

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 5+
**Entry Point**: `jarvis-mcp`

**Key Features**:
- Pipeline creation
- Workflow execution
- Status tracking
- Result retrieval

**Installation**: `uvx iowarp-mcps jarvis`

---

### 8️⃣ Plot MCP - Scientific Visualization

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 4+
**Entry Point**: `plot-mcp`

**Key Features**:
- Matplotlib integration
- Seaborn support
- Plot generation
- Image output

**Installation**: `uvx iowarp-mcps plot`

---

### 9️⃣ Arxiv MCP - Research Paper Discovery

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 3+
**Entry Point**: `arxiv-mcp`

**Key Features**:
- Paper search
- Metadata retrieval
- Abstract access
- Citation information

**Installation**: `uvx iowarp-mcps arxiv`

---

### 🔟 Darshan MCP - I/O Performance Analysis

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 4+
**Entry Point**: `darshan-mcp`

**Key Features**:
- I/O trace analysis
- Performance metrics
- Bottleneck detection
- Pattern identification

**Installation**: `uvx iowarp-mcps darshan`

---

### 1️⃣1️⃣ Chronolog MCP - Event Logging

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 4+
**Entry Point**: `chronolog-mcp`

**Key Features**:
- Event logging
- Timestamp tracking
- Query capabilities
- Data persistence

**Installation**: `uvx iowarp-mcps chronolog`

---

### 1️⃣2️⃣ Compression MCP - File Compression

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 3+
**Entry Point**: `compression-mcp`

**Key Features**:
- Gzip compression
- File archiving
- Compression statistics
- Decompression support

**Installation**: `uvx iowarp-mcps compression`

---

### 1️⃣3️⃣ Parallel Sort MCP - Large File Sorting

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 2+
**Entry Point**: `parallel-sort-mcp`

**Key Features**:
- Parallel sorting
- Large file handling
- Memory-efficient operations
- Performance metrics

**Installation**: `uvx iowarp-mcps parallel-sort`

---

### 1️⃣4️⃣ lmod MCP - Module Environment Management

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 3+
**Entry Point**: `lmod-mcp`

**Key Features**:
- Module loading
- Environment management
- Module search
- Configuration control

**Installation**: `uvx iowarp-mcps lmod`

---

### 1️⃣5️⃣ Node Hardware MCP - System Information

**Status**: 🟢 Production Ready
**SDK**: MCP 1.4+
**Tools**: 3+
**Entry Point**: `node-hardware-mcp`

**Key Features**:
- Hardware detection
- System metrics
- Resource information
- Device enumeration

**Installation**: `uvx iowarp-mcps node-hardware`

---

## SDK Strategy & Migration Path

### Current Status

```
Standard MCP SDK:  14/15 MCPs (93%)
├── HDF5 v2.0     ⭐ Enhanced, production-ready
├── Adios         Stable
├── Parquet       Stable
├── Pandas        Stable
├── Slurm         Stable
├── Jarvis        Stable
├── Plot          Stable
├── Arxiv         Stable
├── Darshan       Stable
├── Chronolog     Stable
├── Compression   Stable
├── Parallel_Sort Stable
├── lmod          Stable
└── Node_Hardware Stable

FastMCP:           1/15 MCPs (7%)
└── NDP           ✅ Intentionally using FastMCP
```

### Why NDP Uses FastMCP

NDP deliberately uses FastMCP because:
- **60-90% less boilerplate** than standard MCP SDK
- **Faster development** - 3-5x quicker tool creation
- **Production-proven** - FastMCP 2.0 is based on official SDK's FastMCP 1.0
- **Better DX** - Decorator-based API similar to FastAPI
- **Enterprise features** - Built-in auth, composition, proxying

### FastMCP 2.0 Advantages

| Feature | Standard SDK | FastMCP 2.0 |
|---------|--------------|-------------|
| Boilerplate | High | Minimal |
| Tool Definition | Classes + Handlers | Simple @decorator |
| Schema Generation | Manual | Automatic |
| Type Safety | Partial | Full (Pydantic) |
| Testing | Complex | Simple |
| Multiple Transports | Basic | Advanced (SSE, FastAPI, Memory) |
| Authentication | Manual | Built-in (OAuth, OIDC) |
| Server Composition | Limited | Full support |
| Development Speed | Hours | Minutes |

### Recommended Migration Strategy

#### Phase 1: Validation (Oct 2025)
- **Goal**: Prove FastMCP works for IoWarp MCPs
- **MCPs**: Start with 2-3 small MCPs (Compression, Parallel_Sort, Node_Hardware)
- **Effort**: 8-12 hours per MCP
- **Success Metric**: All tests pass, equivalent functionality

#### Phase 2: Quick Wins (Q4 2025)
- **Goal**: Migrate all small-to-medium MCPs
- **MCPs**: Arxiv, Chronolog, lmod, Plot, Parquet
- **Effort**: 4-6 hours per MCP
- **Success Metric**: All tools functional, tests passing

#### Phase 3: Strategic Migration (Q1 2026)
- **Goal**: Migrate medium-to-large MCPs
- **MCPs**: Pandas, Jarvis, Darshan, Adios
- **Effort**: 6-10 hours per MCP
- **Success Metric**: Performance parity, all features preserved

#### Phase 4: Enterprise Standardization (Q2 2026)
- **Goal**: Modernize HDF5 to FastMCP (already advanced, optional)
- **MCPs**: HDF5 (optional - already production-ready)
- **Effort**: 8-12 hours (would simplify codebase)
- **Success Metric**: 10% codebase reduction, same features

### Migration Effort Estimates

| MCP | Size | Current SDK | Est. Effort | Priority | Timeline |
|-----|------|-------------|------------|----------|----------|
| Compression | Small | MCP | 4h | High | Week 1 |
| Parallel_Sort | Small | MCP | 4h | High | Week 1 |
| Node_Hardware | Small | MCP | 4h | High | Week 1 |
| Arxiv | Small | MCP | 4h | Medium | Week 2 |
| Chronolog | Small | MCP | 5h | Medium | Week 2 |
| lmod | Small | MCP | 5h | Medium | Week 2 |
| Plot | Small | MCP | 5h | Medium | Week 3 |
| Parquet | Small | MCP | 5h | Medium | Week 3 |
| Pandas | Medium | MCP | 8h | Medium | Week 4 |
| Jarvis | Medium | MCP | 8h | Low | Week 5 |
| Darshan | Medium | MCP | 8h | Low | Week 5 |
| Adios | Medium | MCP | 8h | Low | Week 6 |
| HDF5 | Large | MCP | 12h | Optional | Later |
| **TOTAL** | | | **111 hours** | | 6 weeks |

**Key Insight**: Full migration to FastMCP for all 12 standard SDK MCPs (excluding HDF5 and NDP already using advanced patterns) would take ~6 weeks with concurrent teams.

### Risk Assessment

**Low Risk**:
- Small MCPs (< 5 tools) with simple I/O
- No custom state management
- No complex authentication

**Medium Risk**:
- Medium MCPs (5-10 tools)
- Some async I/O patterns
- Moderate state management

**Higher Caution**:
- Large MCPs (10+ tools like HDF5)
- Complex resource management
- Advanced transport handling

**Mitigation**:
- Start with small MCPs to validate process
- Use parallel testing during migration
- Maintain backward compatibility during rollout
- Comprehensive test coverage before deployment

### Benefits of Full FastMCP Ecosystem

After complete migration to FastMCP:

| Aspect | Benefit |
|--------|---------|
| **Codebase** | 50-70% less code |
| **Development** | 3-5x faster MCP creation |
| **Maintenance** | 50% less maintenance burden |
| **Testing** | 40% simpler test setup |
| **Consistency** | 100% standardized patterns |
| **DX** | Unified decorator approach |
| **Features** | Access to FastMCP advanced capabilities |

---

## Installation & Usage

### Run Any MCP

```bash
# Start any MCP server
uvx iowarp-mcps <mcp-name>

# Examples
uvx iowarp-mcps hdf5
uvx iowarp-mcps ndp
uvx iowarp-mcps pandas
```

### List All MCPs

```bash
# See all available MCPs
uvx iowarp-mcps
```

### Development Installation

```bash
# Clone and develop
git clone https://github.com/iowarp/iowarp-mcps
cd iowarp-mcps

# Install all MCPs with dev dependencies
uv sync --all-extras --dev

# Run specific MCP
uv run python mcps/HDF5/src/hdf5_mcp/server.py
```

---

## Documentation Structure

Each MCP includes:
- `README.md` - Quick start and overview
- `pyproject.toml` - Dependencies and configuration
- `src/` - Source code
- `tests/` - Test suite
- Optional `docs/` - Extended documentation
- Optional `data/` - Sample data

Root documentation:
- `README.md` - Main project overview
- `MCPS.md` - This file (detailed MCP reference)
- `pyproject.toml` - Root project configuration

---

## Contributing New MCPs

### FastMCP (Recommended for New MCPs)

```python
from fastmcp import FastMCP

mcp = FastMCP("NewMCPName")

@mcp.tool()
def my_tool(param: str) -> dict:
    """Tool description."""
    return {"result": "data"}

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
```

### Migration Checklist

- [ ] Create `mcps/NewMCP/` directory
- [ ] Add `pyproject.toml` with dependencies
- [ ] Create `src/newmcp_name/server.py`
- [ ] Implement tools using @mcp.tool() decorator
- [ ] Add `tests/` directory with test suite
- [ ] Create `README.md` with usage examples
- [ ] Test with `uvx iowarp-mcps newmcp-name`
- [ ] Add to main README.md table
- [ ] Submit pull request

---

## Support & Resources

- **GitHub**: https://github.com/iowarp/iowarp-mcps
- **Documentation**: https://iowarp.github.io/iowarp-mcps/
- **Zulip Chat**: [IoWarp Community](https://grc.zulipchat.com/#narrow/channel/518574-iowarp-mcps)
- **Issues**: https://github.com/iowarp/iowarp-mcps/issues

---

## Version History

### v0.4.0 (Oct 18, 2025)
- ✅ Comprehensive README modernization
- ✅ Added NDP (v1.0) to public-facing documentation
- ✅ HDF5 upgraded to v2.0 with HPC-optimized capabilities
- ✅ Added version column to MCP matrix
- ✅ Created detailed MCP reference (MCPS.md)
- ✅ Strategic FastMCP migration planning
- 🎯 15 MCPs now fully documented and discoverable

### v0.3.9 (Earlier)
- 14 MCPs with standard MCP SDK
- Basic documentation

---

**Part of [IoWarp Scientific Computing MCPs](https://github.com/iowarp/iowarp-mcps)**
