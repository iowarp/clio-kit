# Manual Testing Guide for HDF5 MCP v2.0

## Overview

This guide helps you test the new HDF5 MCP implementation in Claude Code and report any issues.

---

## Phase 1: Setup Test Environment

### Step 1: Create Test Directory
```bash
# Open new terminal
cd ~
mkdir -p hdf5_mcp_test
cd hdf5_mcp_test
```

### Step 2: Create Test HDF5 Files
```bash
# Create test script
cat > create_test_data.py << 'EOF'
import h5py
import numpy as np

# Test File 1: Simple structure
print("Creating test1.h5...")
with h5py.File('test1.h5', 'w') as f:
    # Simple dataset
    f.create_dataset('data', data=np.arange(1000))
    f.create_dataset('matrix', data=np.random.rand(100, 100))
    f.attrs['experiment'] = 'test_1'
    f.attrs['date'] = '2025-10-18'

    # Nested group
    grp = f.create_group('results')
    grp.create_dataset('temperature', data=np.random.rand(500) * 100)
    grp.create_dataset('pressure', data=np.random.rand(500) * 10)
    grp.attrs['units'] = 'celsius,pascal'

# Test File 2: Larger structure
print("Creating test2.h5...")
with h5py.File('test2.h5', 'w') as f:
    # Large dataset (for streaming test)
    f.create_dataset('large_data', data=np.random.rand(10000, 100))

    # Multiple groups
    for i in range(5):
        grp = f.create_group(f'experiment_{i}')
        grp.create_dataset('measurements', data=np.random.rand(200))
        grp.create_dataset('timestamps', data=np.arange(200))
        grp.attrs['id'] = i

# Test File 3: Complex structure
print("Creating test3.h5...")
with h5py.File('test3.h5', 'w') as f:
    # Multi-dimensional
    f.create_dataset('tensor_3d', data=np.random.rand(50, 50, 50))
    f.create_dataset('tensor_4d', data=np.random.rand(10, 10, 10, 10))

    # Different dtypes
    f.create_dataset('integers', data=np.arange(1000, dtype=np.int32))
    f.create_dataset('floats', data=np.random.rand(1000).astype(np.float32))
    f.create_dataset('complex', data=np.random.rand(100) + 1j * np.random.rand(100))

    # Chunked dataset
    f.create_dataset('chunked',
                     data=np.random.rand(1000, 1000),
                     chunks=(100, 100),
                     compression='gzip')

print("✓ Created test1.h5, test2.h5, test3.h5")
EOF

# Run it
python3 create_test_data.py
```

### Step 3: Verify Files Created
```bash
ls -lh *.h5
# Should see: test1.h5, test2.h5, test3.h5
```

---

## Phase 2: Install to Claude Code

### Step 4: Add HDF5 MCP to Claude Code
```bash
# Install from local development version
claude mcp add hdf5-dev -- uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp

# Verify it was added
claude mcp list | grep hdf5
```

### Step 5: Restart Claude Code
```bash
# Restart Claude Code to pick up new MCP
# (Close and reopen Claude Code application)
```

---

## Phase 3: Test Basic Functionality

### Step 6: Test in Claude Code Chat

Open Claude Code and try these prompts in order:

#### Test 1: List Tools
```
Can you list all available HDF5 tools?
```

**Expected**: Should see 25 tools listed

**If fails**: Copy error message

---

#### Test 2: Open File
```
Open the file test1.h5 from my current directory
```

**Expected**: Success message "Successfully opened test1.h5 in r mode"

**If fails**: Note error details

---

#### Test 3: Analyze Structure
```
Analyze the structure of the current HDF5 file
```

**Expected**: Shows groups, datasets, hierarchy

**If fails**: Copy error output

---

#### Test 4: List Attributes
```
List all attributes at path "/"
```

**Expected**: Shows root-level attributes (experiment, date)

**If fails**: Note what happened

---

#### Test 5: Read Dataset
```
Read the dataset at path "/data" and tell me about it
```

**Expected**: Successfully reads and describes the data

**If fails**: Copy error

---

#### Test 6: Suggest Exploration
```
Suggest what I should explore next in this file
```

**Expected**: Ranked suggestions based on dataset characteristics

**If fails**: Note issue

---

#### Test 7: Find Similar Datasets
```
Find datasets similar to "/results/temperature"
```

**Expected**: List of similar datasets (pressure, etc.)

**If fails**: Document error

---

#### Test 8: Stream Large Dataset
```
Stream the dataset at "/large_data" from test2.h5 (open it first)
```

**Expected**: Chunked processing with statistics

**If fails**: Copy error details

---

## Phase 4: Capture Debugging Information

### Step 9: Check MCP Server Logs

```bash
# Claude Code MCP logs location (may vary by OS):
# Linux: ~/.config/Claude/logs/
# macOS: ~/Library/Logs/Claude/

# Find recent HDF5 MCP logs
find ~/.config/Claude/logs -name "*hdf5*" -o -name "*mcp*" 2>/dev/null | head -5

# Or check stderr output
tail -100 ~/.config/Claude/logs/mcp-hdf5-dev.log
```

### Step 10: Test Direct CLI (Without Claude Code)

```bash
cd ~/hdf5_mcp_test

# Test 1: Start server manually
echo '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | \
uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp

# Expected: JSON response with serverInfo

# Test 2: List tools
cat > test_tools.json << 'EOF'
{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","method":"tools/list","id":2}
EOF

cat test_tools.json | uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp 2>&1 | tee test_output.log

# Check output
grep -o '"name":"[^"]*"' test_output.log | wc -l
# Expected: 25
```

---

## Phase 5: Report Issues

### Create Issue Report

Copy this template and fill in details:

```markdown
## HDF5 MCP v2.0 Test Report

**Date**: [DATE]
**Tester**: [YOUR NAME]
**Environment**:
- OS: [Linux/macOS/Windows]
- Python: [VERSION]
- Claude Code: [VERSION]

### Test Results

#### Test 1: Installation
- [ ] PASS / [ ] FAIL
- Details:

#### Test 2: Server Starts
- [ ] PASS / [ ] FAIL
- Details:

#### Test 3: Tools List (25 expected)
- [ ] PASS / [ ] FAIL
- Found: [NUMBER] tools
- Details:

#### Test 4: Open File
- [ ] PASS / [ ] FAIL
- Error (if any):

#### Test 5: Analyze Structure
- [ ] PASS / [ ] FAIL
- Output:

#### Test 6: Read Dataset
- [ ] PASS / [ ] FAIL
- Error (if any):

#### Test 7: Discovery Tools
- [ ] PASS / [ ] FAIL
- Details:

#### Test 8: Streaming
- [ ] PASS / [ ] FAIL
- Details:

### Issues Found

1. **Issue**: [Brief description]
   - **Tool**: [tool name]
   - **Error**: [error message]
   - **Steps to reproduce**:
     ```
     [exact steps]
     ```
   - **Expected**: [what should happen]
   - **Actual**: [what actually happened]

2. **Issue**: [description]
   ...

### Logs

Attach relevant logs:
```
[paste logs here]
```

### Additional Notes

[Any other observations]
```

---

## Phase 6: Performance Testing (Optional)

### Test Caching
```bash
# In Claude Code, run same query twice:

# First time:
"Read dataset /data from test1.h5"
# Note response time

# Second time:
"Read dataset /data from test1.h5 again"
# Should be much faster (cache hit)
```

### Test Parallel Operations
```bash
# Ask Claude Code:
"Use hdf5_batch_read to read all datasets from test2.h5 experiment groups in parallel"

# Expected: 4-8x speedup mentioned
```

### Test Large File Streaming
```bash
# Ask Claude Code:
"Stream the large_data dataset from test2.h5"

# Expected: Chunk-by-chunk processing, memory efficient
```

---

## Quick Reference Commands

### Check MCP is Running
```bash
ps aux | grep hdf5-mcp
```

### View Live Logs
```bash
tail -f ~/.config/Claude/logs/mcp-hdf5-dev.log
```

### Reinstall After Code Changes
```bash
# Remove old MCP
claude mcp remove hdf5-dev

# Re-add with latest code
claude mcp add hdf5-dev -- uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp

# Restart Claude Code
```

### Test Specific Tool Directly
```bash
cat > test_tool.json << 'EOF'
{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","method":"tools/call","id":2,"params":{"name":"analyze_dataset_structure","arguments":{"path":"/"}}}
EOF

cat test_tool.json | uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp
```

---

## Common Issues & Solutions

### Issue: "Module not found"
**Solution**:
```bash
# Reinstall dependencies
cd /home/akougkas/projects/iowarp-mcps/mcps/HDF5
uv sync
```

### Issue: "No file currently open"
**Solution**: Always call `open_file` before dataset operations
```
First: "Open test1.h5"
Then: "Read dataset /data"
```

### Issue: Tools not showing up
**Check**:
```bash
# Verify tools registered
uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp << 'EOF'
{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","method":"tools/list","id":2}
EOF
```

### Issue: SSE mode won't start
**Try**:
```bash
# Start SSE server manually
uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp --transport sse --port 8765

# In another terminal, test health endpoint
curl http://localhost:8765/health
```

---

## Debugging Checklist

When reporting issues, include:

- [ ] Exact command or prompt used
- [ ] Full error message
- [ ] Server logs (stderr output)
- [ ] HDF5 file that caused issue
- [ ] Expected vs actual behavior
- [ ] Python/system environment details

---

## Success Criteria

After testing, you should be able to:

✅ Open HDF5 files
✅ Navigate file structure
✅ Read datasets (full and partial)
✅ Access attributes
✅ Get exploration suggestions
✅ Find similar datasets
✅ Identify bottlenecks
✅ Stream large data
✅ See performance improvements (caching)

---

## Feedback Template

Use this to give me feedback:

```markdown
## Test Session: [DATE]

### What Worked ✅
- Tool X worked perfectly
- Feature Y exceeded expectations
- Performance was [describe]

### What Didn't Work ❌
1. **Tool**: [name]
   **Issue**: [description]
   **Error**: [paste error]
   **Logs**: [relevant logs]

2. **Tool**: [name]
   ...

### Observations
- [anything notable]

### Suggestions
- [improvements]

### Overall Rating
[1-10]: [your rating]

### Ready for Production?
[ ] Yes, merge it
[ ] No, fix issues first
```

---

## Quick Test Script

Run this for automated testing:

```bash
cd ~/hdf5_mcp_test

cat > quick_test.sh << 'EOF'
#!/bin/bash
echo "=== HDF5 MCP Quick Test ==="

# Test 1: Server starts
echo -e "\n[Test 1] Server starts..."
timeout 2 bash -c 'echo | uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp 2>&1' > /dev/null
if [ $? -eq 124 ]; then
    echo "✓ Server started (timed out waiting for input - expected)"
else
    echo "✗ Server failed to start"
fi

# Test 2: Tools registered
echo -e "\n[Test 2] Tools registration..."
TOOL_COUNT=$(cat << 'JSON' | uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp 2>/dev/null | grep '"name":' | wc -l
{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","method":"tools/list","id":2}
JSON
)

if [ "$TOOL_COUNT" -eq 25 ]; then
    echo "✓ All 25 tools registered"
else
    echo "✗ Expected 25 tools, found $TOOL_COUNT"
fi

# Test 3: Version check
echo -e "\n[Test 3] Version..."
VERSION=$(cat << 'JSON' | uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp 2>/dev/null | grep -o '"version":"[^"]*"' | head -1
{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
JSON
)

if [[ "$VERSION" == *"2.0.0"* ]]; then
    echo "✓ Version 2.0.0 confirmed"
else
    echo "✗ Version mismatch: $VERSION"
fi

echo -e "\n=== Test Complete ==="
EOF

chmod +x quick_test.sh
./quick_test.sh
```

---

## Advanced Testing

### Test SSE Mode
```bash
# Terminal 1: Start SSE server
uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp --transport sse --port 8765

# Terminal 2: Test endpoints
curl http://localhost:8765/health
curl http://localhost:8765/stats

# Terminal 3: Test SSE stream
curl -N -H "Accept: text/event-stream" \
     -H "MCP-Protocol-Version: 2025-06-18" \
     http://localhost:8765/mcp
```

### Test Specific Tools

Create test scripts for each tool category:

```bash
cat > test_file_ops.json << 'EOF'
{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}
{"jsonrpc":"2.0","method":"tools/call","id":2,"params":{"name":"open_file","arguments":{"path":"test1.h5"}}}
{"jsonrpc":"2.0","method":"tools/call","id":3,"params":{"name":"get_filename","arguments":{}}}
{"jsonrpc":"2.0","method":"tools/call","id":4,"params":{"name":"close_file","arguments":{}}}
EOF

cat test_file_ops.json | uvx --from /home/akougkas/projects/iowarp-mcps/mcps/HDF5 hdf5-mcp 2>&1 | tee file_ops_test.log
```

---

## Reporting Back to Me

### Format 1: Quick Status
```
TESTED: stdio mode
RESULT: ✅ Works / ❌ Broken
TOOLS TESTED: [list]
ISSUES: [count]
```

### Format 2: Detailed Issue
```
ISSUE: [Tool name] fails with [error]

REPRODUCE:
1. Open test1.h5
2. Call tool X with args Y
3. See error Z

ERROR MESSAGE:
[paste full error]

LOGS:
[paste relevant logs]

EXPECTED:
[what should happen]

ACTUAL:
[what happened]
```

### Format 3: Performance Report
```
PERFORMANCE TEST:

Tool: read_full_dataset
File: test1.h5 (7.8MB)
First call: 150ms
Second call: 2ms
Speedup: 75x ✅

Tool: hdf5_batch_read
Files: 5 datasets
Sequential estimate: 500ms
Parallel actual: 120ms
Speedup: 4.2x ✅
```

---

## Red Flags to Watch For

🚩 Server crashes
🚩 Import errors
🚩 Tools not available
🚩 Incorrect results
🚩 Memory leaks (use `top` to monitor)
🚩 Slow performance
🚩 Security warnings

---

## After Testing

Send me one message with:

1. **Overall verdict**: Ready / Needs fixes
2. **Tests passed**: X/Y
3. **Critical issues**: [list]
4. **Minor issues**: [list]
5. **Performance**: Good / Bad / Not tested
6. **Recommendation**: Merge / Fix issues first

I'll prioritize fixes based on your feedback.

---

## Example Session

```
YOU: "I tested in Claude Code. Here's what happened:"

Test 1 ✅: Listed 25 tools
Test 2 ✅: Opened test1.h5 successfully
Test 3 ✅: Structure analysis worked
Test 4 ❌: read_dataset failed with "AttributeError: 'NoneType'"

Issue Details:
- Tool: read_full_dataset
- File: test1.h5
- Path: "/data"
- Error: AttributeError: 'NoneType' object has no attribute 'shape'
- Logs: [attached]

Overall: 3/4 tests passed. One critical bug. Fix before merge.

ME: [I'll analyze the error and fix it immediately]
```

---

That's it! Test when you're ready and send me feedback.
