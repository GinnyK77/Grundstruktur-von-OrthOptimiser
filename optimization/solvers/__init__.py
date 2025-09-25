"""
Optimierungssolver-Module.

Dieses Paket enthält verschiedene Optimierungssolver wie
Augmented Lagrangian, Particle Swarm Optimization, GEKKO und SciPy.
"""

# Direkte Importe, damit Nutzer schreiben können:
# from optimization.solvers import AugmentedLagrangianSolver
from .base_solver import BaseSolver
from .augmented_lagrangian import AugmentedLagrangianSolver
from .particle_swarm import ParticleSwarmSolver

# Bedingte Importe für optionale Abhängigkeiten
try:
    from .gekko_solver import GekkoSolver
    GEKKO_AVAILABLE = True
except ImportError:
    GEKKO_AVAILABLE = False

try:
    from .scipy_solver import ScipySolver
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Öffentliche API definieren
__all__ = ['BaseSolver', 'AugmentedLagrangianSolver', 'ParticleSwarmSolver']

if GEKKO_AVAILABLE:
    __all__.append('GekkoSolver')

if SCIPY_AVAILABLE:
    __all__.append('ScipySolver')