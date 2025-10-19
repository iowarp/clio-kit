#!/bin/bash
set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     HDF5 MCP Test Environment Setup                      ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo

# Variables
TEST_DIR="$HOME/testing"
IOWARP_DIR="/home/akougkas/projects/iowarp-mcps"
HDF5_MCP_DIR="$IOWARP_DIR/mcps/HDF5"

# Step 1: Create test directory
echo "[1/4] Creating test directory at $TEST_DIR..."
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"
echo "✓ Created $TEST_DIR"
echo

# Step 2: Generate synthetic HDF5 test files
echo "[2/4] Generating synthetic HDF5 test files..."
cat > generate_data.py << 'PYEOF'
import h5py
import numpy as np

# File 1: Simple structure for basic testing
print("Creating test1.h5 (simple)...")
with h5py.File('test1.h5', 'w') as f:
    f.create_dataset('data', data=np.arange(1000))
    f.create_dataset('matrix', data=np.random.rand(100, 100))
    f.attrs['experiment'] = 'test_experiment_1'
    f.attrs['version'] = '1.0'

    grp = f.create_group('results')
    grp.create_dataset('temperature', data=np.random.rand(500) * 100)
    grp.create_dataset('pressure', data=np.random.rand(500) * 10)

# File 2: Larger file for performance testing
print("Creating test2.h5 (large)...")
with h5py.File('test2.h5', 'w') as f:
    f.create_dataset('large_data', data=np.random.rand(10000, 100))

    for i in range(5):
        grp = f.create_group(f'experiment_{i}')
        grp.create_dataset('measurements', data=np.random.rand(200))
        grp.attrs['id'] = i

# File 3: Complex for discovery testing
print("Creating test3.h5 (complex)...")
with h5py.File('test3.h5', 'w') as f:
    f.create_dataset('tensor_3d', data=np.random.rand(50, 50, 50))
    f.create_dataset('integers', data=np.arange(1000, dtype=np.int32))
    f.create_dataset('chunked',
                     data=np.random.rand(1000, 1000),
                     chunks=(100, 100),
                     compression='gzip')

print("\n✓ Created 3 test HDF5 files")
PYEOF

uv run --with h5py --with numpy python3 generate_data.py
rm generate_data.py
echo "✓ Generated test1.h5, test2.h5, test3.h5"
echo

# Step 3: Create test instructions
echo "[3/4] Creating test instructions..."
cat > README.md << 'EOF'
# HDF5 MCP Test Environment

## Setup Complete ✅

Test files ready:
- `test1.h5` - Simple structure (1.7MB)
- `test2.h5` - Large datasets (7.6MB)
- `test3.h5` - Complex structure (380MB)

## Install to Claude Code

```bash
claude mcp add hdf5-dev -- uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp
```

Then restart Claude Code.

## Test Commands

Try these in Claude Code:

### Basic
```
List all available HDF5 MCP tools
```

### File Operations
```
Open test1.h5 and analyze its structure
```

### Discovery
```
Suggest what I should explore in this file
```

### Read Data
```
Read the dataset at /data and show me the first 10 values
```

### Performance
```
Find similar datasets to /results/temperature
```

### Advanced
```
Stream the large_data dataset from test2.h5 in chunks
```

## Report Issues

If something breaks:
1. Copy the error message
2. Check logs: `tail -50 ~/.config/Claude/logs/mcp-*.log`
3. Report back with: Tool name + Error + What you tried

## Quick Validation

```bash
# Test server starts
echo | uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp

# Should see "INFO:hdf5_mcp.server:Starting stdio server"
```

---

**All ready! Start testing in Claude Code.**
EOF

echo "✓ Created README.md with instructions"
echo

# Step 4: Validate setup
echo "[4/4] Validating setup..."

# Check files exist
if [ -f "test1.h5" ] && [ -f "test2.h5" ] && [ -f "test3.h5" ]; then
    echo "✓ All test files present"
    ls -lh *.h5
else
    echo "✗ Missing test files"
    exit 1
fi

echo
echo "═══════════════════════════════════════════════════════════"
echo "✅ TEST ENVIRONMENT READY"
echo "═══════════════════════════════════════════════════════════"
echo
echo "Location: $TEST_DIR"
echo "Files: test1.h5, test2.h5, test3.h5"
echo
echo "NEXT STEPS:"
echo "  1. cd ~/testing"
echo "  2. cat README.md"
echo "  3. claude mcp add hdf5-dev -- uvx --from $HDF5_MCP_DIR hdf5-mcp"
echo "  4. Restart Claude Code"
echo "  5. Start testing!"
echo
