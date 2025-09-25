"""
Strategiemodule für Optimierungsstartwerte.

Dieses Paket enthält Funktionen zur Generierung von Startwerten
für Optimierungsalgorithmen.
"""

# Direkte Importe für häufig genutzte Funktionen
from .starting_points import get_starting_points, generate_fixed_points, generate_random_points

# Öffentliche API definieren
__all__ = ['get_starting_points', 'generate_fixed_points', 'generate_random_points']