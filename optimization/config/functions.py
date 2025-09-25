"""Konfigurationsdatei für Zielfunktionen."""
import numpy as np

# Wörterbuch zum Speichern aller registrierten Funktionen
_REGISTERED_FUNCTIONS = {}

# Dekorator zum Registrieren von Funktionen
def register_function(name=None):
    """Registriert eine Funktion, um sie für die Optimierung verfügbar zu machen."""
    def decorator(func):
        nonlocal name
        if name is None:
            name = func.__name__
        _REGISTERED_FUNCTIONS[name] = func
        return func
    return decorator

# Funktion zum Abrufen aller registrierten Funktionen
def get_all_functions():
    """Gibt alle registrierten Funktionen zurück."""
    return _REGISTERED_FUNCTIONS

# Registriert die Teilfunktionen
@register_function("f1")
def f1(x, y, params=None):
    """Erste Teilfunktion: x*exp(-(x^2+y^2)/a_1)"""
    if params is None:
        return x * np.exp(-(x**2 + y**2))
    
    a_1 = params['a_1']
    return x * np.exp(-(x**2 + y**2) / a_1)

@register_function("f2")
def f2(x, y, params=None):
    """Zweite Teilfunktion: (x^2+y^2)/a_4"""
    if params is None:
        return (x**2 + y**2) / 20
    
    a_4 = params['a_4']
    return (x**2 + y**2) / a_4

@register_function("f3")
def f3(x, y, params=None):
    """Dritte Teilfunktion: (a_3 * sin(x*y*a_2))"""
    if params is None:
        return np.sin(x * y)
    
    a_2 = params['a_2']
    a_3 = params['a_3']
    return a_3 * np.sin(x * y * a_2)

@register_function("f4")
def f4(x, y, params=None):
    """Vierte Teilfunktion: a_5 * cos(x*y/a_6)"""
    if params is None:
        return np.cos(x * y / 2)
    
    a_5 = params.get('a_5', 1.0)  # Default, wenn nicht in params definiert
    a_6 = params.get('a_6', 2.0)
    return a_5 * np.cos(x * y / a_6)

# Weitere Funktionen mit dem @register_function-Dekorator hinzufügen


def create_total_function(function_names, weights):
    """Erstellt eine Gesamtfunktion aus Teilfunktionen mit Gewichten."""
    functions = [_REGISTERED_FUNCTIONS[name] for name in function_names]
    
    def total_function(x, y, implant_params=None):
        result = 0
        for i, func in enumerate(functions):
            # KORRIGIERT: Ermittle den ursprünglichen Funktionsindex aus dem Namen
            func_name = function_names[i]  # z.B. "f3"
            original_index = int(func_name[1:]) - 1  # "f3" → 3-1 = 2
            
            if original_index in weights and weights[original_index] != 0:
                result += weights[original_index] * func(x, y, implant_params)
        return result
    
    return total_function

def create_total_function_with_t(function_names, weights):
    """Erstellt eine Gesamtfunktion mit t als Variable."""
    total_func = create_total_function(function_names, weights)
    
    def total_function_with_t(x, y, t, implant_parameters):
        # t ist bereits eine Ganzzahl, beschränke nur auf gültige Werte
        t = max(0, min(t, len(implant_parameters) - 1))
        # Verwende den entsprechenden Implantattyp
        implant_params = implant_parameters[t]
        # Verwende die bestehende Gesamtfunktion mit den gewählten Parametern
        return total_func(x, y, implant_params)
    
    return total_function_with_t

'''
# Erstellt eine gewichtete Gesamtfunktion - KORRIGIERT
def create_total_function(function_names, weights):
    """Erstellt eine Gesamtfunktion aus Teilfunktionen mit Gewichten."""
    functions = [_REGISTERED_FUNCTIONS[name] for name in function_names]
    
    def total_function(x, y, implant_params=None):
        result = 0
        for i, func in enumerate(functions):
            # KORRIGIERT: Ermittle den ursprünglichen Funktionsindex aus dem Namen
            func_name = function_names[i]  # z.B. "f3"
            original_index = int(func_name[1:]) - 1  # "f3" → 3-1 = 2
            
            if original_index in weights and weights[original_index] != 0:
                result += weights[original_index] * func(x, y, implant_params)
        return result
    
    return total_function

# Erstellt eine Gesamtfunktion mit t als Variable - UNVERÄNDERT
def create_total_function_with_t(function_names, weights):
    """Erstellt eine Gesamtfunktion mit t als Variable."""
    total_func = create_total_function(function_names, weights)
    
    def total_function_with_t(x, y, t, implant_parameters):
        # t ist bereits eine Ganzzahl, beschränke nur auf gültige Werte
        t = max(0, min(t, len(implant_parameters) - 1))
        # Verwende den entsprechenden Implantattyp
        implant_params = implant_parameters[t]
        # Verwende die bestehende Gesamtfunktion mit den gewählten Parametern
        return total_func(x, y, implant_params)
    
    return total_function_with_t
'''

# Für benutzerfreundliche Terminalausgabe später in der Hauptfunktion
def get_function_info(name):
    """Gibt Informationen über eine bestimmte Funktion zurück."""
    if name in _REGISTERED_FUNCTIONS:
        func = _REGISTERED_FUNCTIONS[name]
        return {
            "id": name,
            "name": name,
            "formula": getattr(func, "formula", name),
            "description": func.__doc__ or "Keine Beschreibung verfügbar"
        }
    return None

def get_all_function_info():
    """Gibt Informationen über alle registrierten Funktionen zurück."""
    return [get_function_info(name) for name in _REGISTERED_FUNCTIONS.keys()]