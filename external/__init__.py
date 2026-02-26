import sys
from pathlib import Path

# Add external CatVTON to path
project_root = Path(__file__).resolve().parent
catvton_path = project_root / "catvton"
if str(catvton_path) not in sys.path:
    sys.path.insert(0, str(catvton_path))

oot_path = project_root / "ootdiffusion"
if str(oot_path) not in sys.path:
    sys.path.insert(0, str(oot_path))