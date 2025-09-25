"""Abstrakte Basisklasse für alle Optimierungsverfahren."""
from abc import ABC, abstractmethod
import time

class BaseSolver(ABC):
    """Abstrakte Basisklasse für alle Optimierungsverfahren."""
    
    def __init__(self, total_function, constraint_function, x_bounds, y_bounds, z_threshold,
                implant_parameters, t_bounds=None):
        """
        Initialisiert einen Solver.
        
        Args:
            total_function: Gesamtzielfunktion mit t-Abhängigkeit
            constraint_function: Nebenbedingungsfunktion mit t-Abhängigkeit
            x_bounds: Grenzen für x-Variable (tuple: (min, max))
            y_bounds: Grenzen für y-Variable (tuple: (min, max))
            z_threshold: Schwellenwert für die Nebenbedingung
            implant_parameters: Liste der Implantatparameter
            t_bounds: Grenzen für t-Variable (tuple: (min, max))
        """
        self.total_function = total_function
        self.constraint_function = constraint_function
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_threshold = z_threshold
        self.implant_parameters = implant_parameters
        
        if t_bounds is None:
            t_bounds = (0, len(implant_parameters) - 1)
        self.t_bounds = t_bounds
        
        # Standardwerte für die Ausgabe
        self.verbose = True
    
    def print_colored(self, text, color=None):
        """Gibt Text in Farbe aus."""
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'purple': '\033[95m',
            'cyan': '\033[96m',
            'bold': '\033[1m',
            'end': '\033[0m'
        }
        
        if color in colors and self.verbose:
            print(f"{colors[color]}{text}{colors['end']}")
        elif self.verbose:
            print(text)
    
    @abstractmethod
    def solve(self, **kwargs):
        """
        Führt die Optimierung durch.
        
        Returns:
            dict: Optimierungsergebnis
        """
        pass
    
    def format_result(self, x_opt, y_opt, t_opt, f_opt, lambda_opt, path, 
                   constraint_value, computation_time, termination_reason,
                   outer_iterations=None, inner_iterations=None):
        """
        Formatiert das Optimierungsergebnis in ein einheitliches Format.
        
        Returns:
            dict: Standardisiertes Ergebnisformat
        """
        # Implant-Parameter des optimalen Types
        implant_params_opt = self.implant_parameters[t_opt]
        
        return {
            "x_opt": x_opt,
            "y_opt": y_opt,
            "t_opt": t_opt,
            "f_opt": f_opt,
            "lambda_opt": lambda_opt,
            "implant_params_opt": implant_params_opt,
            "constraint_value": constraint_value,
            "path": path,
            "is_feasible": constraint_value >= -1e-6,  # Toleranz für Rundungsfehler
            "computation_time": computation_time,
            "termination_reason": termination_reason,
            "outer_iterations": outer_iterations,
            "inner_iterations": inner_iterations,
            "total_steps": len(path) if path else 0
        }
    
    def print_result(self, result):
        """Gibt ein Optimierungsergebnis formatiert aus."""
        if not result:
            self.print_colored("\nKeine zulässige Lösung gefunden!", "red")
            return
            
        self.print_colored("\nOptimaler Punkt:", "cyan")
        print(f"  x = {result['x_opt']:.6f}")
        print(f"  y = {result['y_opt']:.6f}")
        print(f"  t = {result['t_opt']} ({self.implant_parameters[result['t_opt']]['name']})")
        print(f"  f(x,y,t) = {result['f_opt']:.6f}")
        
        self.print_colored("\nImplantatparameter:", "cyan")
        for key, value in result['implant_params_opt'].items():
            if key != 'name':
                print(f"  {key} = {value:.6f}")
        
        self.print_colored("\nNebenbedingung:", "cyan")
        print(f"  g(x,y,t) = {result['constraint_value']:.6f} " +
              ("✓" if result['is_feasible'] else "✗"))
        
        self.print_colored("\nPerformance-Metriken:", "cyan")
        print(f"  Berechnungszeit: {result['computation_time']:.6f} Sekunden")
        
        if result['outer_iterations'] is not None:
            print(f"  Äußere Iterationen: {result['outer_iterations']}")
        if result['inner_iterations'] is not None:
            print(f"  Innere Iterationen: {result['inner_iterations']}")
        print(f"  Gesamtschritte: {result['total_steps']}")
        print(f"  Terminierungsgrund: {result['termination_reason']}")