#!/bin/bash
# Fixed Safemode that uses the standalone repair system

echo "========================================"
echo "🛡️ OZ SAFEMODE FIXED"
echo "========================================"

echo ""
echo "This safemode uses the standalone repair system"
echo "that doesn't modify OzUnifiedHypervisor.py"
echo ""

# Run the standalone safemode
python3 oz_safemode.py
