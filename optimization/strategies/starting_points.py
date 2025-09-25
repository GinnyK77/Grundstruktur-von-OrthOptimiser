"""Strategien für die Generierung von Startwerten."""
import numpy as np

def get_starting_points(strategy, x_bounds, y_bounds, t_bounds, implant_parameters, num_points=10):
    """
    Generiert Startwerte für die Optimierung.
    
    Args:
        strategy: 'fixed', 'random', oder 'combined'
        x_bounds, y_bounds: Grenzen für x und y
        t_bounds: Grenzen für t
        implant_parameters: Implantatparameter
        num_points: Anzahl der Startwerte (für 'random' und 'combined')
        
    Returns:
        list: Liste von Startpunkten
    """
    if strategy == "fixed":
        return generate_fixed_points(implant_parameters, t_bounds)
    elif strategy == "random":
        return generate_random_points(x_bounds, y_bounds, t_bounds, num_points)
    elif strategy == "combined":
        fixed = generate_fixed_points(implant_parameters, t_bounds)
        random = generate_random_points(x_bounds, y_bounds, t_bounds, max(0, num_points - len(fixed)))
        return fixed + random
    else:
        raise ValueError(f"Unbekannte Startwert-Strategie: {strategy}")

def generate_fixed_points(implant_parameters, t_bounds):
    """Generiert feste Startwerte, die systematisch alle Implantattypen abdecken."""
    points = []
    
    # Für jeden Implantattyp
    for t in range(min(len(implant_parameters), t_bounds[1] + 1)):
        if t >= t_bounds[0]:
            # Erzeuge eine Reihe von strategischen Punkten für diesen Typ
            points.append({
                "name": f"Ursprung mit {implant_parameters[t]['name']}", 
                "x0": 0.0, "y0": 0.0, "t0": t
            })
            points.append({
                "name": f"Q1 mit {implant_parameters[t]['name']}", 
                "x0": 1.0, "y0": 1.0, "t0": t
            })
            points.append({
                "name": f"Q3 mit {implant_parameters[t]['name']}", 
                "x0": -1.0, "y0": -1.0, "t0": t
            })
            points.append({
                "name": f"Q4 mit {implant_parameters[t]['name']}", 
                "x0": 1.0, "y0": -1.0, "t0": t
            })
            points.append({
                "name": f"Q2 mit {implant_parameters[t]['name']}", 
                "x0": -1.0, "y0": 1.0, "t0": t
            })
    
    return points

def generate_random_points(x_bounds, y_bounds, t_bounds, num_points):
    """Generiert zufällige Startwerte."""
    points = []
    for i in range(num_points):
        x0 = np.random.uniform(x_bounds[0], x_bounds[1])
        y0 = np.random.uniform(y_bounds[0], y_bounds[1])
        t0 = np.random.randint(t_bounds[0], t_bounds[1] + 1)
        points.append({
            "name": f"Random {i+1}", 
            "x0": x0, "y0": y0, "t0": t0
        })
    return points