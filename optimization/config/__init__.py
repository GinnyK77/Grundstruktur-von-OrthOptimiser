"""
Konfigurationsmodule für die Optimierung.

Dieses Paket enthält Definitionen von Zielfunktionen,
Nebenbedingungen und Implantatparametern.
"""

# Direkte Importe für häufig genutzte Funktionen
from .parameters import get_all_implant_params
from .functions import create_total_function, create_total_function_with_t
from .constraints import create_horizontal_constraint_with_t

# Öffentliche API definieren
__all__ = [
    'get_all_implant_params',
    'create_total_function',
    'create_total_function_with_t',
    'create_horizontal_constraint_with_t'
]