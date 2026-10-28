import sys
from pathlib import Path

# Make `src` and `scripts` importable when pytest is run from anywhere.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
