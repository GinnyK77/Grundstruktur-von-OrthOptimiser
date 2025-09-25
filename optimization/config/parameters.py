"""Konfigurationsdatei für Implantatparameter."""

# Wörterbuch der Implantatparameter - leicht erweiterbar
IMPLANT_PARAMETERS = [
    {
        "name": "Typ A", 
        "a_1": 1.0, 
        "a_2": 1.0, 
        "a_3": 1.0, 
        "a_4": 20.0, 
        "a_5": 1.2, 
        "a_6": 2.5
    },
    #{
    #    "name": "Typ B", 
    #    "a_1": 1.5, 
    #    "a_2": 0.8, 
    #    "a_3": 1.2, 
    #    "a_4": 25.0, 
    #    "a_5": 0.9, 
    #    "a_6": 7.0
    #},
    {
        'name': 'Typ B',
        'a_1': 1.8,
        'a_2': 2.0,
        'a_3': 1.2,
        'a_4': 15.0,
        'a_5': 1.0,
        'a_6': 1.0
    },
    {
        "name": "Typ C", 
        "a_1": 0.8, 
        "a_2": 1.2, 
        "a_3": 0.9, 
        "a_4": 15.0, 
        "a_5": 7, 
        "a_6": 1.3
    },
    #{
    #    'name': 'Implantat Typ D',
    #    'a_1': 2.2,
    #    'a_2': 1.2,
    #    'a_3': 1.0,
    #    'a_4': 8.0,
    #    'a_5': 3,
    #    'a_6': 7
    #},
    {
        "name": "Typ E", 
        "a_1": 1.0, 
        "a_2": 3.0, 
        "a_3": 1.5, 
        "a_4": 20.0, 
        "a_5": 15, 
        "a_6": 1.0
    },
    {
        "name": "Typ F", 
        "a_1": 2.0, 
        "a_2": 0.5, 
        "a_3": 1.8, 
        "a_4": 35.0, 
        "a_5": 1.0, 
        "a_6": 15
    },
    # Nach Bedarf weitere Implantattypen hinzufügen
]

def get_implant_params(t):
    """Holt Implantatparameter anhand des Index."""
    if 0 <= t < len(IMPLANT_PARAMETERS):
        return IMPLANT_PARAMETERS[t]
    else:
        raise ValueError(f"Ungültiger Implantattyp-Index: {t}. Muss zwischen 0 und {len(IMPLANT_PARAMETERS)-1} liegen")

def get_all_implant_params():
    """Liefert alle Implantatparameter."""
    return IMPLANT_PARAMETERS