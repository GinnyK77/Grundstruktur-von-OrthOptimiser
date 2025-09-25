"""Konfigurationsdatei für Nebenbedingungen."""
import numpy as np

# Wörterbuch zum Speichern aller registrierten Nebenbedingungen
_REGISTERED_CONSTRAINTS = {}

# Dekorator zum Registrieren von Nebenbedingungen
def register_constraint(name=None):
    """Registriert eine Nebenbedingung, um sie für die Optimierung verfügbar zu machen."""
    def decorator(func):
        nonlocal name
        if name is None:
            name = func.__name__
        _REGISTERED_CONSTRAINTS[name] = func
        return func
    return decorator

# Registriert die horizontale Nebenbedingung
@register_constraint("horizontal")
def constraint_horizontal(x, y, total_function, z_threshold):
    """Horizontale Ebenen-Nebenbedingung."""
    return total_function(x, y) - z_threshold

# Weitere Nebenbedingungen mit dem @register_constraint-Dekorator hinzufügen

# Funktion zum Abrufen einer bestimmten Nebenbedingung anhand des Namens
def get_constraint(name):
    """Ruft eine bestimmte Nebenbedingung anhand des Namens ab."""
    if name in _REGISTERED_CONSTRAINTS:
        return _REGISTERED_CONSTRAINTS[name]
    else:
        raise ValueError(f"Nebenbedingung '{name}' nicht gefunden. Verfügbare Nebenbedingungen: {list(_REGISTERED_CONSTRAINTS.keys())}")

# Erstellt horizontale Nebenbedingung mit t
def create_horizontal_constraint_with_t(total_function_with_t):
    """Erstellt eine horizontale Nebenbedingung mit t als Variable."""
    def constraint_horizontal_with_t(x, y, t, implant_parameters, z_threshold):
        """Horizontale Ebenen-Nebenbedingung mit t als Variable."""
        return total_function_with_t(x, y, t, implant_parameters) - z_threshold
    
    return constraint_horizontal_with_t

# Für benutzerfreundliche Terminalausgabe später in der Hauptfunktion
def get_all_constraints():
    """Gibt alle verfügbaren Nebenbedingungen zurück."""
    return list(_REGISTERED_CONSTRAINTS.values())