# Adds the project's module directories to sys.path.
import os, sys
_root = os.path.dirname(os.path.abspath(__file__))
for _d in ('theory', 'experiments', 'experiments/constructed'):
    _p = os.path.join(_root, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
