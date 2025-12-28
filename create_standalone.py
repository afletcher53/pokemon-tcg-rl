#!/usr/bin/env python3
"""
Script to create a standalone training file by combining all source files.
"""

import re

# Files to combine in order (dependency order)
files = [
    ("tcg/cards.py", "CARDS"),
    ("tcg/state.py", "STATE"),
    ("tcg/actions.py", "ACTIONS"),
    ("tcg/effects.py", "EFFECTS"),
    ("tcg/env.py", "ENVIRONMENT"),
    ("tcg/mcts.py", "MCTS"),
    ("tcg/scripted_agent.py", "SCRIPTED_AGENTS"),
    ("alpha_rank.py", "ALPHA_RANK"),
    ("train_advanced.py", "TRAINING"),
]

# Local imports to remove (they'll be inline)
local_imports = [
    r"from tcg\.cards import.*",
    r"from tcg\.state import.*",
    r"from tcg\.actions import.*",
    r"from tcg\.effects import.*",
    r"from tcg\.env import.*",
    r"from tcg\.mcts import.*",
    r"from tcg\.scripted_agent import.*",
    r"from alpha_rank import.*",
    r"from record_replay import.*",
    r"import tcg\..*",
]

# Type checking imports to remove
type_check_block = re.compile(r"if TYPE_CHECKING:.*?(?=\n[^\s]|\Z)", re.DOTALL)

output = []

output.append('#!/usr/bin/env python3')
output.append('"""')
output.append('STANDALONE Pokemon TCG AlphaZero Training Script')
output.append('')
output.append('This is a complete, self-contained training script combining all source files.')
output.append('Generated automatically from the multi-file Pokemon TCG RL project.')
output.append('')
output.append('Usage:')
output.append('    python standalone_full.py --episodes 5000 --resume checkpoint.pt')
output.append('"""')
output.append('')

# Standard library imports (deduplicated)
std_imports = set()
std_imports.add("from __future__ import annotations")
std_imports.add("import torch")
std_imports.add("import torch.nn as nn")
std_imports.add("import torch.nn.functional as F")
std_imports.add("import torch.optim as optim")
std_imports.add("import numpy as np")
std_imports.add("import random")
std_imports.add("import os")
std_imports.add("import argparse")
std_imports.add("import copy")
std_imports.add("import math")
std_imports.add("import json")
std_imports.add("from collections import deque")
std_imports.add("from dataclasses import dataclass, field")
std_imports.add("from typing import List, Tuple, Dict, Optional, Callable, Any, TYPE_CHECKING")
std_imports.add("from tqdm import tqdm")
std_imports.add("import multiprocessing as mp")
std_imports.add("from multiprocessing import Pool, Queue")
std_imports.add("import torch.multiprocessing as tmp")
std_imports.add("import gymnasium as gym")
std_imports.add("from gymnasium import spaces")

for imp in sorted(std_imports):
    output.append(imp)

output.append("")
output.append("")

for filepath, section_name in files:
    output.append(f"# {'=' * 77}")
    output.append(f"# {section_name}")
    output.append(f"# Source: {filepath}")
    output.append(f"# {'=' * 77}")
    output.append("")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Remove local imports
        for pattern in local_imports:
            content = re.sub(pattern + r"\n?", "", content)
        
        # Remove TYPE_CHECKING blocks
        content = type_check_block.sub("", content)
        
        # Remove standard imports (already added at top)
        content = re.sub(r"from __future__ import annotations\n?", "", content)
        content = re.sub(r"import torch\n?", "", content)
        content = re.sub(r"import torch\.nn as nn\n?", "", content)
        content = re.sub(r"import torch\.nn\.functional as F\n?", "", content)
        content = re.sub(r"import torch\.optim as optim\n?", "", content)
        content = re.sub(r"import numpy as np\n?", "", content)
        content = re.sub(r"import random\n?", "", content)
        content = re.sub(r"import os\n?", "", content)
        content = re.sub(r"import argparse\n?", "", content)
        content = re.sub(r"import copy\n?", "", content)
        content = re.sub(r"import math\n?", "", content)
        content = re.sub(r"import json\n?", "", content)
        content = re.sub(r"from collections import deque\n?", "", content)
        content = re.sub(r"from dataclasses import.*\n?", "", content)
        content = re.sub(r"from typing import.*\n?", "", content)
        content = re.sub(r"from tqdm import tqdm\n?", "", content)
        content = re.sub(r"import multiprocessing as mp\n?", "", content)
        content = re.sub(r"from multiprocessing import.*\n?", "", content)
        content = re.sub(r"import torch\.multiprocessing as tmp\n?", "", content)
        content = re.sub(r"import gymnasium as gym\n?", "", content)
        content = re.sub(r"from gymnasium import spaces\n?", "", content)
        
        # Remove shebang if present (except for main file)
        content = re.sub(r"#/usr/bin/env /bin/python3 /home/aaron/.cursor/extensions/ms-python.debugpy-2025.14.1-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher 33587 -- /home/aaron/Desktop/SR_Stopping/beta.py  python3\n?", "", content)
        
        # Remove empty lines at start
        content = content.lstrip('\n')
        
        output.append(content)
        output.append("")
        output.append("")
    except FileNotFoundError:
        output.append(f"# ERROR: File not found: {filepath}")
        output.append("")

# Write output
with open("standalone_full.py", "w") as f:
    f.write("\n".join(output))

print(f"Created standalone_full.py")
import os
print(f"File size: {os.path.getsize('standalone_full.py')} bytes")
