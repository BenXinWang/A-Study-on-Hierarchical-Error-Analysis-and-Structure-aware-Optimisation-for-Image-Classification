#!/usr/bin/env python3
"""
Standalone error analysis script for subclass predictions.
This script can be run independently to generate error tables and analysis.
"""

import sys
import os
from pathlib import Path

# Add scripts to path
sys.path.append('scripts')

def main():
    """Main function to run error analysis."""
    from evaluation.generate_error_table import main as generate_error_table_main
    
    print("🚀 Starting standalone error analysis...")
    print("="*60)
    
    # Run the error table generation
    generate_error_table_main()
    
    print("\n" + "="*60)
    print("✅ Standalone error analysis completed!")
    print("="*60)

if __name__ == '__main__':
    main()
