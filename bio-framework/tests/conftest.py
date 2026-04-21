"""Pytest configuration: add project root to sys.path for applications/ and analysis/."""

import sys
from pathlib import Path

# Add project root so that 'applications' and 'analysis' are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
