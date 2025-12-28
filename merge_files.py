#!/usr/bin/env python3
"""
Merge multiple Python files into one standalone file.
Properly handles multi-line imports.
"""

import re
import os

# Files to merge in dependency order
FILES = [
    "tcg/cards.py",
    "tcg/state.py", 
    "tcg/actions.py",
    "tcg/effects.py",
    "tcg/env.py",
    "tcg/mcts.py",
    "tcg/scripted_agent.py",
    "alpha_rank.py",
    "train_advanced.py",
]

# Patterns for local imports
LOCAL_IMPORT_STARTS = [
    'from tcg.',
    'import tcg.',
    'from alpha_rank',
    'from record_replay',
]

# Standard library imports to collect
STD_IMPORTS = set()

def extract_and_clean(filepath):
    """Extract code from file, removing local imports including multi-line."""
    print(f"Processing {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    cleaned = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip shebang
        if stripped.startswith('#!/'):
            i += 1
            continue
        
        # Skip TYPE_CHECKING blocks
        if 'if TYPE_CHECKING:' in line:
            i += 1
            # Skip until we hit unindented line
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith(' ') or lines[i].startswith('\t')):
                i += 1
            continue
        
        # Check for local import
        is_local_import = any(stripped.startswith(p) for p in LOCAL_IMPORT_STARTS)
        
        if is_local_import:
            # Check if it's multi-line (ends with '(' or contains '(')
            if '(' in stripped and ')' not in stripped:
                # Skip until we find closing paren
                while i < len(lines) and ')' not in lines[i]:
                    i += 1
            i += 1
            continue
        
        # Standard library imports - collect them
        if stripped.startswith('from ') or stripped.startswith('import '):
            if not any(p.replace('from ', '').replace('import ', '') in stripped for p in LOCAL_IMPORT_STARTS):
                # Multi-line import?
                if '(' in stripped and ')' not in stripped:
                    import_lines = [stripped]
                    i += 1
                    while i < len(lines) and ')' not in lines[i]:
                        import_lines.append(lines[i].strip())
                        i += 1
                    if i < len(lines):
                        import_lines.append(lines[i].strip())
                    # Join and add
                    full_import = ' '.join(import_lines)
                    STD_IMPORTS.add(full_import)
                else:
                    STD_IMPORTS.add(stripped)
                i += 1
                continue
        
        cleaned.append(line)
        i += 1
    
    return '\n'.join(cleaned)

def main():
    sections = []
    
    for filepath in FILES:
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping")
            continue
        
        section_name = os.path.basename(filepath).replace('.py', '').upper()
        code = extract_and_clean(filepath)
        
        # Clean up excessive blank lines
        while '\n\n\n\n' in code:
            code = code.replace('\n\n\n\n', '\n\n\n')
        
        # Add section header
        header = f"""
# {'=' * 77}
# {section_name}
# Source: {filepath}
# {'=' * 77}

"""
        sections.append(header + code.strip())
    
    # Build final file
    output = '''#!/usr/bin/env python3
"""
STANDALONE Pokemon TCG AlphaZero Training Script

This is a complete, self-contained training script combining all source files.
Generated automatically from the multi-file Pokemon TCG RL project.

Usage:
    python standalone_full.py --episodes 5000 --resume checkpoint.pt
"""

from __future__ import annotations
import argparse
import copy
import csv
import gymnasium as gym
import json
import math
import multiprocessing as mp
import numpy as np
import os
import random
import torch
import torch.multiprocessing as tmp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, Counter
from dataclasses import dataclass, field
from gymnasium import spaces
from multiprocessing import Pool, Queue
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Callable, Any

'''
    
    # Add all sections
    for section in sections:
        output += section
        output += '\n\n'
    
    # Write output
    with open('standalone_full.py', 'w') as f:
        f.write(output)
    
    # Verify syntax
    print("\nVerifying syntax...")
    import ast
    try:
        with open('standalone_full.py', 'r') as f:
            ast.parse(f.read())
        print("✅ Syntax OK!")
    except SyntaxError as e:
        print(f"❌ Syntax error at line {e.lineno}: {e.msg}")
        # Show the problematic line
        with open('standalone_full.py', 'r') as f:
            lines = f.readlines()
            if e.lineno and e.lineno <= len(lines):
                print(f"   Line {e.lineno}: {lines[e.lineno-1].rstrip()[:80]}")
                if e.lineno > 1:
                    print(f"   Line {e.lineno-1}: {lines[e.lineno-2].rstrip()[:80]}")
        return
    
    # Report stats
    with open('standalone_full.py', 'r') as f:
        lines = len(f.readlines())
    size = os.path.getsize('standalone_full.py')
    print(f"\nGenerated standalone_full.py:")
    print(f"  Lines: {lines}")
    print(f"  Size: {size:,} bytes ({size/1024:.1f} KB)")

if __name__ == '__main__':
    main()
