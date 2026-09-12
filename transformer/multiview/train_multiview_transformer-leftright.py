"""
Multiview transformer, left vs right imagined
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "transformer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The whole variant: the parent script, run with a different task spec. Every
# flag, model, split and training loop is the parent's, so a left/right number
# and a 5-class number from these two files differ in the labels and nothing
# else. See transformer/leftright.py for what the spec changes.
from leftright import LEFT_RIGHT
import train_multiview_transformer as base

if __name__ == "__main__":
    base.main(task_spec=LEFT_RIGHT)
