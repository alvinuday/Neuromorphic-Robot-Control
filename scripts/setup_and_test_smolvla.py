#!/usr/bin/env python3
"""
Helper script to extract SmolVLA server URL from Jupyter notebook.

Usage:
    python3 scripts/get_smolvla_url.py <notebook_path>

This script:
1. Connects to a Jupyter kernel running the SmolVLA notebook
2. Finds the ngrok URL from the notebook's output
3. Sets SMOLVLA_SERVER_URL environment variable
4. Runs integration tests with real server

Prerequisite:
- Start the notebook in Jupyter: jupyter lab vla/smolvla_server.ipynb
- Run all cells up to the ngrok tunnel creation
- Copy the ngrok token into the notebook
"""

import json
import subprocess
import sys
import os
import re
from pathlib import Path


def extract_url_from_notebook(notebook_path: str) -> str:
    """
    Extract the smolvla_url variable from a Jupyter notebook.
    
    Looks for the most recent assignment to smolvla_url in the notebook.
    """
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
        
        # Look through all cells for the smolvla_url variable
        for cell in reversed(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                # Look for smolvla_url assignment
                if 'smolvla_url' in source and '=' in source:
                    # Try to find URL in output
                    if 'outputs' in cell:
                        for output in cell['outputs']:
                            if output.get('output_type') == 'stream':
                                text = ''.join(output.get('text', []))
                                # Look for ngrok URL
                                match = re.search(r'https://[a-zA-Z0-9_-]+-ngrok(?:-free)?\.dev', text)
                                if match:
                                    return match.group(0)
        
        return None
    
    except Exception as e:
        print(f"Error reading notebook: {e}")
        return None


def run_with_server(notebook_path: str, test_file: str = "tests/test_integration_real_smolvla.py"):
    """
    Extract URL from notebook and run tests with it.
    
    Args:
        notebook_path: Path to SmolVLA server notebook
        test_file: Test file to run
    """
    print(f"📓 Reading SmolVLA notebook: {notebook_path}")
    
    # Try kernel variable access (if notebook is running in Jupyter)
    print("\n🔍 Attempting to extract URL from notebook...")
    url = extract_url_from_notebook(notebook_path)
    
    if not url:
        print("\n❌ Could not extract URL from notebook.")
        print("\nManual setup instructions:")
        print("1. Start Jupyter: jupyter lab")
        print("2. Open: vla/smolvla_server.ipynb")
        print("3. Run all cells up to the ngrok tunnel")
        print("4. Copy the ngrok URL output")
        print("5. Set environment: export SMOLVLA_SERVER_URL='https://xxxx-ngrok-free.dev'")
        print("6. Run tests: pytest tests/test_integration_real_smolvla.py -v -s")
        return False
    
    print(f"\n✓ Found URL: {url}")
    
    # Set environment variable
    os.environ['SMOLVLA_SERVER_URL'] = url
    
    # Run tests
    print(f"\n🧪 Running integration tests with real server...")
    print(f"   URL: {url}\n")
    
    ret = subprocess.run(
        ['python3', '-m', 'pytest', test_file, '-v', '-s'],
        cwd=Path(test_file).parent.parent  # Project root
    )
    
    return ret.returncode == 0


if __name__ == "__main__":
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else "vla/smolvla_server.ipynb"
    
    success = run_with_server(notebook_path)
    sys.exit(0 if success else 1)
