'''Python package for fast, stable root finding.

'''

from numalgsolve.polyroots import solve
from numalgsolve.subdivision import solve as subdivide_solve

__all__ = ['solve', 'subdivide_solve']
