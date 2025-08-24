#!/usr/bin/env python3
"""
Standalone script to run workflow evaluation pipeline
"""
import os
import sys

# Add the parent directory to the Python path so we can import evoagentx
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from evaluation_pipeline.main_evaluator import main

if __name__ == '__main__':
    sys.exit(main())
