#!/usr/bin/env python3

"""
Test local installation of the zero_cost_search package

This script verifies that the package can be imported and used after installation.

Usage:
    python test_local_install.py
"""

import sys


def main():
    print("Testing local installation of zero_cost_search...")
    
    try:
        import zero_cost_search
        print(f"Successfully imported zero_cost_search version {zero_cost_search.__version__}")
        
        # Check that main classes are available
        from zero_cost_search import ZeroCostNAS, ZeroCostMetrics, MLP
        print("Successfully imported main classes")
        
        # Create a simple instance
        nas = ZeroCostNAS(input_dim=10, output_dim=2)
        print("Successfully created ZeroCostNAS instance")
        
        print("\nAll tests passed! The package is installed correctly.")
        return 0
    except ImportError as e:
        print(f"Error importing package: {e}")
        print("\nThe package may not be installed correctly.")
        print("Try reinstalling with: pip install -e .")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())