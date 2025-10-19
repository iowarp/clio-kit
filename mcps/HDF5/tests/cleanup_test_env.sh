#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     HDF5 MCP Test Environment Cleanup                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo

TEST_DIR="$HOME/testing"

if [ -d "$TEST_DIR" ]; then
    echo "Removing test directory: $TEST_DIR"
    rm -rf "$TEST_DIR"
    echo "✓ Removed $TEST_DIR"
else
    echo "✓ Test directory doesn't exist (already clean)"
fi

echo
echo "═══════════════════════════════════════════════════════════"
echo "✅ CLEANUP COMPLETE"
echo "═══════════════════════════════════════════════════════════"
