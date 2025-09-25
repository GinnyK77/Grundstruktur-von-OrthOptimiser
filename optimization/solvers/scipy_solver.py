"""Solver basierend auf SciPy SLSQP mit Enumeration."""
import numpy as np
import time
from .base_solver import BaseSolver

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class ScipySolver(BaseSolver):
    """Solver mit SciPy SLSQP und Enumerationsmethode für t."""
    
    def __init__(self, total_function, constraint_function, x_bounds, y_bounds, z_threshold,
                implant_parameters, t_bounds=None):
        """
        Initialisiert den SciPy Solver.
        
        Args:
            total_function: Gesamtzielfunktion mit t-Abhängigkeit
            constraint_function: Nebenbedingungsfunktion mit t-Abhängigkeit (wird ignoriert)
            x_bounds: Grenzen für x-Variable (tuple: (min, max))
            y_bounds: Grenzen für y-Variable (tuple: (min, max))
            z_threshold: Schwellenwert für die Nebenbedingung (wird ignoriert)
            implant_parameters: Liste der Implantatparameter
            t_bounds: Grenzen für t-Variable (tuple: (min, max))
        """
        super().__init__(total_function, constraint_function, x_bounds, y_bounds, z_threshold,
                        implant_parameters, t_bounds)
        
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy ist nicht installiert. Installieren Sie es mit: pip install scipy")
    
    def solve(self, integer_constraints=False, verbose=True):
        """
        Löst das Optimierungsproblem mit SciPy SLSQP und Enumeration.
        
        Args:
            integer_constraints: Wenn True, werden x und y auf ganzzahlige Werte beschränkt
            verbose: Ausführliche Ausgabe aktivieren
            
        Returns:
            dict: Optimierungsergebnis oder None, falls keine zulässige Lösung gefunden
        """
        self.verbose = verbose
        
        self.print_colored("=== SCIPY SLSQP MIT ENUMERATION ===", "bold")
        print(f"Optimierungsbereich: x: [{self.x_bounds[0]}, {self.x_bounds[1]}], " +
              f"y: [{self.y_bounds[0]}, {self.y_bounds[1]}], " +
              f"t: [0, {len(self.implant_parameters)-1}] (Enumeration)")
        
        if verbose:
            print("\n=== DETAIL: SCIPY SLSQP ENUMERATION ===")
            print(f"Anzahl Implantattypen: {len(self.implant_parameters)}")
            print(f"Ganzzahlige Koordinaten: {integer_constraints}")
            print(f"Verwende Enumerationsmethode für t...")
        
        # Startpunkte definieren
        starting_points = self._generate_starting_points(integer_constraints)
        
        # Finde das beste Ergebnis über verschiedene Implantattypen und Startpunkte
        best_result = None
        best_objective = float('-inf')
        best_solve_time = 0
        best_status = ""
        
        start_time_total = time.time()
        
        # Für jeden Implantattyp separat optimieren
        for t in range(len(self.implant_parameters)):
            implant_params = self.implant_parameters[t]
            implant_name = implant_params['name']
            
            if verbose:
                print(f"\nOptimiere für {implant_name} (t={t})...")
            
            # Beste Lösung für diesen Implantattyp
            t_best_result = None
            t_best_objective = float('inf')  # Wir minimieren, daher beginnen wir mit inf
            t_best_solve_time = 0
            t_best_status = ""
            
            # Definition der Zielfunktion für diesen Implantattyp (negativ für Maximierung)
            def objective_function(xy):
                x, y = xy
                
                # Falls ganzzahlige Optimierung, runde die Werte
                if integer_constraints:
                    x = round(x)
                    y = round(y)
                
                # Berechne den Funktionswert
                z = self.total_function(x, y, t, self.implant_parameters)
                
                # Negativ zurückgeben für Maximierung
                return -z
            
            # Versuche verschiedene Startpunkte für diesen Implantattyp
            for start_x, start_y in starting_points:
                try:
                    solve_start_time = time.time()
                    
                    # Grenzen für die Optimierung
                    bounds = [(self.x_bounds[0], self.x_bounds[1]), 
                             (self.y_bounds[0], self.y_bounds[1])]
                    
                    # Optimierungsproblem lösen mit SLSQP
                    result = minimize(
                        objective_function, 
                        [start_x, start_y], 
                        method='SLSQP',
                        bounds=bounds,
                        options={'ftol': 1e-6, 'disp': False, 'maxiter': 1000}
                    )
                    
                    solve_time = time.time() - solve_start_time
                    
                    # Wenn erfolgreich optimiert
                    if result.success:
                        x_opt, y_opt = result.x
                        
                        # Falls ganzzahlige Optimierung, runde die Werte
                        if integer_constraints:
                            x_opt = round(x_opt)
                            y_opt = round(y_opt)
                        
                        # Berechne den tatsächlichen Funktionswert (positiv)
                        z_opt = -result.fun
                        
                        if verbose:
                            print(f"  Startpunkt ({start_x}, {start_y}): x={x_opt:.4f}, y={y_opt:.4f}, f={z_opt:.4f}")
                            print(f"  Lösungszeit: {solve_time:.3f}s, Status: {result.message}")
                        
                        # Prüfen, ob dies das beste Ergebnis für diesen Implantattyp ist
                        if result.fun < t_best_objective:
                            t_best_objective = result.fun
                            t_best_result = (x_opt, y_opt, z_opt)
                            t_best_solve_time = solve_time
                            t_best_status = result.message
                            if verbose:
                                print(f"  → Neues bestes Ergebnis für {implant_name}!")
                    else:
                        if verbose:
                            print(f"  Optimierung fehlgeschlagen: {result.message}")
                
                except Exception as e:
                    if verbose:
                        print(f"  Optimierung fehlgeschlagen: {e}")
            
            # Wenn eine Lösung für diesen Implantattyp gefunden wurde
            if t_best_result is not None:
                x_opt, y_opt, z_opt = t_best_result
                
                # Prüfen, ob dies das beste Ergebnis insgesamt ist
                if z_opt > best_objective:
                    best_objective = z_opt
                    best_result = (x_opt, y_opt, t, z_opt)
                    best_solve_time = t_best_solve_time
                    best_status = t_best_status
                    if verbose:
                        print(f"  → Neues globales Optimum mit {implant_name}!")
        
        end_time_total = time.time()
        computation_time_total = end_time_total - start_time_total
        
        # Wenn kein gültiges Ergebnis gefunden wurde
        if best_result is None:
            if verbose:
                print("Alle Optimierungsversuche fehlgeschlagen.")
            return None
        
        # Ergebnis formatieren und zurückgeben
        x_opt, y_opt, t_opt, z_opt = best_result
        
        # Erstelle Pfad (SciPy liefert nur Endpunkt)
        convergence_path = [(x_opt, y_opt, t_opt)]
        
        result = self.format_result(
            x_opt=x_opt,
            y_opt=y_opt,
            t_opt=t_opt,
            f_opt=z_opt,
            lambda_opt=0.0,  # SciPy verwendet keine Lagrange-Multiplikatoren in diesem Setup
            path=convergence_path,
            constraint_value=0.0,  # SciPy arbeitet ohne explizite Constraints in diesem Setup
            computation_time=computation_time_total,
            termination_reason=f"SciPy SLSQP Enumeration ({best_status})",
            outer_iterations=len(self.implant_parameters),
            inner_iterations=len(starting_points)
        )
        
        self.print_colored("\n=== ERGEBNIS DER SCIPY SLSQP ENUMERATION ===", "green")
        self.print_result(result)
        
        return result
    
    def _generate_starting_points(self, integer_constraints):
        """Generiert Startpunkte für die SciPy-Optimierung."""
        if integer_constraints:
            starting_points = [
                (0, 0), (1, 1), (-1, 1), (2, -2), (-2, 2)
            ]
        else:
            starting_points = [
                (0.0, 0.0), (1.0, 1.0), (-1.0, -1.0), (2.0, -2.0), (-2.0, 2.0)
            ]
        
        return starting_points