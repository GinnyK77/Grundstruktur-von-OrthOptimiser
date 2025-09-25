"""Numerische Hilfsfunktionen für die Optimierung."""
import numpy as np

def augmented_lagrangian_with_t(x, y, t, lambda_k, mu_k, total_function, constraint_function, implant_parameters, z_threshold):
    """Berechnet den Wert der Augmented-Lagrangian-Funktion mit t als Variable."""
    f_val = total_function(x, y, t, implant_parameters)
    g_val = constraint_function(x, y, t, implant_parameters, z_threshold)
    
    # Augmented Lagrangian für Maximierungsproblem
    penalty_term = (mu_k/2) * (min(0, g_val))**2
    
    return f_val - lambda_k * g_val + penalty_term

def augmented_lagrangian_gradient(x, y, t, lambda_k, mu_k, total_function, constraint_function, implant_parameters, z_threshold, h=1e-6):
    """Berechnet den Gradienten der Augmented-Lagrangian-Funktion bezüglich x und y."""
    # Partielle Ableitung nach x
    dL_dx = (augmented_lagrangian_with_t(x + h, y, t, lambda_k, mu_k, total_function, constraint_function, implant_parameters, z_threshold) - 
             augmented_lagrangian_with_t(x - h, y, t, lambda_k, mu_k, total_function, constraint_function, implant_parameters, z_threshold)) / (2 * h)
    
    # Partielle Ableitung nach y
    dL_dy = (augmented_lagrangian_with_t(x, y + h, t, lambda_k, mu_k, total_function, constraint_function, implant_parameters, z_threshold) - 
             augmented_lagrangian_with_t(x, y - h, t, lambda_k, mu_k, total_function, constraint_function, implant_parameters, z_threshold)) / (2 * h)
    
    return np.array([dL_dx, dL_dy])

def compute_discrete_direction_t(x, y, t, augmented_lagrangian_with_t, lambda_k, mu_k, 
                               total_function, constraint_function, implant_parameters, z_threshold):
    """
    Berechnet die diskrete Richtung für t basierend auf dem steilsten Anstieg (ΔL_A/Δt).
    Da Δt = 1 (diskrete Schritte), ist die Steigung die Differenz der Lagrangian-Werte.
    
    Returns:
    - direction: -1 (t verringern), 0 (t unverändert lassen) oder 1 (t erhöhen)
    """
    # Aktueller Wert (t ist bereits ein Integer)
    current_val = augmented_lagrangian_with_t(x, y, t, lambda_k, mu_k, 
                                           total_function, constraint_function, 
                                           implant_parameters, z_threshold)
    
    # Wert für t+1 (wenn möglich)
    next_val = None
    if t < len(implant_parameters) - 1:
        next_val = augmented_lagrangian_with_t(x, y, t + 1, lambda_k, mu_k, 
                                             total_function, constraint_function, 
                                             implant_parameters, z_threshold)
    
    # Wert für t-1 (wenn möglich)
    prev_val = None
    if t > 0:
        prev_val = augmented_lagrangian_with_t(x, y, t - 1, lambda_k, mu_k, 
                                             total_function, constraint_function, 
                                             implant_parameters, z_threshold)
    
    # Berechne die Änderungen (Steigungen)
    delta_plus = next_val - current_val if next_val is not None else float('-inf')
    delta_minus = prev_val - current_val if prev_val is not None else float('-inf')
    
    # Bestimme die beste Richtung für t basierend auf der größten Steigung
    direction = 0  # Standardrichtung: keine Änderung
    
    # Wähle die Richtung mit der größten positiven Steigung
    if delta_plus > 0 and delta_plus >= delta_minus:
        direction = 1  # t erhöhen
    elif delta_minus > 0:
        direction = -1  # t verringern
        
    return direction

def numerical_gradient(x, y, total_function, h=1e-6):
    """Berechnet den numerischen Gradienten der Gesamtfunktion an der Stelle (x,y)."""
    df_dx = (total_function(x + h, y) - total_function(x - h, y)) / (2 * h)
    df_dy = (total_function(x, y + h) - total_function(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])

def numerical_gradient_with_t(x, y, t, total_function_with_t, implant_parameters, h=1e-6):
    """Berechnet den numerischen Gradienten bezüglich x und y bei festem t."""
    df_dx = (total_function_with_t(x + h, y, t, implant_parameters) - 
             total_function_with_t(x - h, y, t, implant_parameters)) / (2 * h)
    
    df_dy = (total_function_with_t(x, y + h, t, implant_parameters) - 
             total_function_with_t(x, y - h, t, implant_parameters)) / (2 * h)
    
    return np.array([df_dx, df_dy])

def verify_local_maximum(x, y, t, total_function_with_t, implant_parameters, h=1e-5):
    """
    Prüft, ob ein Punkt ein lokales Maximum ist, unter Verwendung der Hesse-Matrix.
    Für ein Maximum: f_xx < 0 und det(H) > 0
    """
    # Aktuelle Implantatparameter
    implant_params = implant_parameters[t]
    
    # Definiere eine lokale Funktion für einfachere Berechnung
    def f(x_val, y_val):
        return total_function_with_t(x_val, y_val, t, implant_parameters)
    
    # Berechne zweite Ableitungen
    f_xx = (f(x+h, y) - 2*f(x, y) + f(x-h, y)) / (h**2)
    f_yy = (f(x, y+h) - 2*f(x, y) + f(x, y-h)) / (h**2)
    f_xy = (f(x+h, y+h) - f(x+h, y-h) - f(x-h, y+h) + f(x-h, y-h)) / (4*h**2)
    
    # Berechne die Determinante der Hesse-Matrix
    det_hessian = f_xx * f_yy - f_xy**2
    
    # Für ein Maximum: f_xx < 0 und det_hessian > 0
    is_maximum = f_xx < 0 and det_hessian > 0
    
    return is_maximum