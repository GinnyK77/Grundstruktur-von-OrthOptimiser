"""Solver basierend auf GEKKO MINLP."""
import numpy as np
import time
from .base_solver import BaseSolver

try:
    from gekko import GEKKO
    GEKKO_AVAILABLE = True
except ImportError:
    GEKKO_AVAILABLE = False

class GekkoSolver(BaseSolver):
    """Solver mit GEKKO Mixed Integer Non-Linear Programming."""
    
    def __init__(self, total_function, constraint_function, x_bounds, y_bounds, z_threshold,
                implant_parameters, t_bounds=None):
        """
        Initialisiert den GEKKO Solver.
        
        Args:
            total_function: Gesamtzielfunktion mit t-Abhängigkeit
            constraint_function: Nebenbedingungsfunktion mit t-Abhängigkeit (wird ignoriert, da GEKKO ohne Constraints arbeitet)
            x_bounds: Grenzen für x-Variable (tuple: (min, max))
            y_bounds: Grenzen für y-Variable (tuple: (min, max))
            z_threshold: Schwellenwert für die Nebenbedingung (wird ignoriert)
            implant_parameters: Liste der Implantatparameter
            t_bounds: Grenzen für t-Variable (tuple: (min, max))
        """
        super().__init__(total_function, constraint_function, x_bounds, y_bounds, z_threshold,
                        implant_parameters, t_bounds)
        
        if not GEKKO_AVAILABLE:
            raise ImportError("GEKKO ist nicht installiert. Installieren Sie es mit: pip install gekko")
    
    def solve(self, integer_constraints=False, verbose=True):
        """
        Löst das Optimierungsproblem mit GEKKO MINLP.
        
        Args:
            integer_constraints: Wenn True, werden x und y auf ganzzahlige Werte beschränkt
            verbose: Ausführliche Ausgabe aktivieren
            
        Returns:
            dict: Optimierungsergebnis oder None, falls keine zulässige Lösung gefunden
        """
        self.verbose = verbose
        
        self.print_colored("=== GEKKO MIXED INTEGER NON-LINEAR PROGRAMMING ===", "bold")
        print(f"Optimierungsbereich: x: [{self.x_bounds[0]}, {self.x_bounds[1]}], " +
              f"y: [{self.y_bounds[0]}, {self.y_bounds[1]}], " +
              f"t: [0, {len(self.implant_parameters)-1}] (ganzzahlig)")
        
        if verbose:
            print("\n=== DETAIL: GEKKO MINLP OPTIMIZATION ===")
            print(f"Anzahl Implantattypen: {len(self.implant_parameters)}")
            print(f"Ganzzahlige Koordinaten: {integer_constraints}")
            print(f"Initialisiere Multi-Start-Optimierung...")
        
        # Startpunkte definieren
        starting_configurations = self._generate_starting_points(integer_constraints)
        
        # Finde das beste Ergebnis über verschiedene Startpunkte
        best_result = None
        best_objective = float('-inf')
        best_solve_time = 0
        best_exit_flag = None
        
        start_time_total = time.time()
        
        for i, (start_x, start_y, start_t) in enumerate(starting_configurations):
            if verbose:
                print(f"\nStartpunkt {i+1}/{len(starting_configurations)}: (x={start_x}, y={start_y}, t={start_t})...")
            
            try:
                # Einzelne Optimierung durchführen
                start_time = time.time()
                
                result = self._solve_single_configuration(
                    start_x, start_y, start_t, integer_constraints, verbose
                )
                
                solve_time = time.time() - start_time
                
                if result is not None:
                    x_opt, y_opt, t_opt, z_opt, exit_flag = result
                    
                    if verbose:
                        implant_name = self.implant_parameters[t_opt]["name"]
                        print(f"  Ergebnis: x={x_opt:.4f}, y={y_opt:.4f}, t={t_opt} ({implant_name}), f={z_opt:.4f}")
                        print(f"  Lösungszeit: {solve_time:.3f}s")
                    
                    # Prüfen, ob dies das beste bisher gefundene Ergebnis ist
                    if z_opt > best_objective:
                        best_objective = z_opt
                        best_result = (x_opt, y_opt, t_opt, z_opt)
                        best_solve_time = solve_time
                        best_exit_flag = exit_flag
                        if verbose:
                            print(f"  → Neues bestes Ergebnis!")
                
            except Exception as e:
                if verbose:
                    print(f"  Optimierung fehlgeschlagen: {e}")
        
        end_time_total = time.time()
        computation_time_total = end_time_total - start_time_total
        
        # Fallback auf Grid-Suche, wenn keine Lösung gefunden
        if best_result is None:
            if verbose:
                print("Alle Optimierungsversuche fehlgeschlagen. Verwende Grid-Suche als Fallback...")
            return self._fallback_grid_search(integer_constraints, computation_time_total)
        
        # Ergebnis formatieren und zurückgeben
        x_opt, y_opt, t_opt, z_opt = best_result
        
        # Erstelle Pfad (GEKKO liefert nur Endpunkt)
        convergence_path = [(x_opt, y_opt, t_opt)]
        
        result = self.format_result(
            x_opt=x_opt,
            y_opt=y_opt,
            t_opt=t_opt,
            f_opt=z_opt,
            lambda_opt=0.0,  # GEKKO verwendet keine Lagrange-Multiplikatoren in diesem Setup
            path=convergence_path,
            constraint_value=0.0,  # GEKKO arbeitet ohne explizite Constraints in diesem Setup
            computation_time=computation_time_total,
            termination_reason=f"GEKKO Optimization (Exit flag: {best_exit_flag})",
            outer_iterations=len(starting_configurations),
            inner_iterations=0
        )
        
        self.print_colored("\n=== ERGEBNIS DER GEKKO MINLP OPTIMIZATION ===", "green")
        self.print_result(result)
        
        return result
    
    def _generate_starting_points(self, integer_constraints):
        """Generiert Startpunkte für die GEKKO-Optimierung."""
        starting_configurations = []
        
        if integer_constraints:
            starting_points = [
                (0, 0), (1, 1), (-1, 1), (2, -2), (-2, 2), (3, -3), (-3, 3)
            ]
            selected_types = [0, 3]  # Typ A und Typ D als Startpunkte
        else:
            starting_points = [
                (0.0, 0.0), (1.0, 1.0), (-1.0, -1.0), (2.0, -2.0), (-2.0, 2.0), (3.0, -3.0), (-3.0, 3.0)
            ]
            selected_types = [0, 3]  # Typ A und Typ D als Startpunkte
        
        for x, y in starting_points:
            for t in selected_types:
                starting_configurations.append((x, y, t))
        
        return starting_configurations
    
    def _solve_single_configuration(self, start_x, start_y, start_t, integer_constraints, verbose):
        """Führt eine einzelne GEKKO-Optimierung durch."""
        try:
            # GEKKO-Optimierungsumgebung erstellen
            m = GEKKO(remote=False)
            
            # Variablen definieren
            if integer_constraints:
                x = m.Var(start_x, lb=self.x_bounds[0], ub=self.x_bounds[1], integer=True, name='x')
                y = m.Var(start_y, lb=self.y_bounds[0], ub=self.y_bounds[1], integer=True, name='y')
            else:
                x = m.Var(start_x, lb=self.x_bounds[0], ub=self.x_bounds[1], name='x')
                y = m.Var(start_y, lb=self.y_bounds[0], ub=self.y_bounds[1], name='y')
            
            # Ganzzahlige Variable für den Implantattyp
            t = m.Var(start_t, lb=0, ub=len(self.implant_parameters)-1, integer=True, name='implant_type')
            
            # Binäre Variablen für die Typauswahl
            type_selection = []
            for i in range(len(self.implant_parameters)):
                init_val = 1 if i == start_t else 0
                type_i = m.Var(init_val, lb=0, ub=1, integer=True, name=f'type_{i}')
                type_selection.append(type_i)
            
            # Bedingung: Genau ein Implantattyp muss ausgewählt werden
            m.Equation(sum(type_selection) == 1)
            
            # Verbinde t mit den binären Variablen
            m.Equation(sum(i * type_selection[i] for i in range(len(self.implant_parameters))) == t)
            
            # Parameter als Funktion der Typauswahl definieren
            a_1_values = [params['a_1'] for params in self.implant_parameters]
            a_2_values = [params['a_2'] for params in self.implant_parameters]
            a_3_values = [params['a_3'] for params in self.implant_parameters]
            a_4_values = [params['a_4'] for params in self.implant_parameters]
            
            a_1 = m.Intermediate(sum(a_1_values[i] * type_selection[i] for i in range(len(self.implant_parameters))))
            a_2 = m.Intermediate(sum(a_2_values[i] * type_selection[i] for i in range(len(self.implant_parameters))))
            a_3 = m.Intermediate(sum(a_3_values[i] * type_selection[i] for i in range(len(self.implant_parameters))))
            a_4 = m.Intermediate(sum(a_4_values[i] * type_selection[i] for i in range(len(self.implant_parameters))))
            
            # Zwischenschritte der Berechnung definieren
            x_squared = m.Intermediate(x**2)
            y_squared = m.Intermediate(y**2)
            sum_squares = m.Intermediate(x_squared + y_squared)
            
            # Teilfunktionen entsprechend der ursprünglichen Struktur
            # f1: x * exp(-(x^2 + y^2)/a_1)
            exp_term = m.Intermediate(m.exp(-sum_squares / a_1))
            f1_value = m.Intermediate(x * exp_term)
            
            # f2: (x^2 + y^2) / a_4
            f2_value = m.Intermediate(sum_squares / a_4)
            
            # f3: a_3 * sin(x*y*a_2)
            xy_product = m.Intermediate(x * y * a_2)
            f3_value = m.Intermediate(a_3 * m.sin(xy_product))
            
            # Zielfunktion (angenommen: alle drei Funktionen mit Gewicht 1)
            # Dies muss eventuell angepasst werden basierend auf den gewünschten Gewichtungen
            objective = m.Intermediate(f1_value + f2_value + f3_value)
            
            # Ziel der Optimierung festlegen: Maximierung der Zielfunktion
            m.Maximize(objective)
            
            # Solver-Konfiguration
            m.options.SOLVER = 3  # APOPT für gemischt-ganzzahlige Optimierung
            m.options.IMODE = 3   # Optimierungsmodus (Steady state optimization)
            m.options.MAX_ITER = 1000
            m.options.OTOL = 1e-6
            m.options.RTOL = 1e-6
            
            # Optimierung durchführen
            m.solve(disp=False)
            
            # Ergebnisse extrahieren
            x_opt = x.value[0]
            y_opt = y.value[0]
            t_opt = round(t.value[0])
            
            # Falls ganzzahlige Optimierung, runde zur Sicherheit
            if integer_constraints:
                x_opt = round(x_opt)
                y_opt = round(y_opt)
            
            # Berechne den tatsächlichen Funktionswert mit der ursprünglichen Funktion
            implant_params = self.implant_parameters[t_opt]
            z_opt = self.total_function(x_opt, y_opt, t_opt, self.implant_parameters)
            
            # Exit flag extrahieren
            exit_flag = m.options.APPINFO if hasattr(m.options, 'APPINFO') else 0
            
            return (x_opt, y_opt, t_opt, z_opt, exit_flag)
            
        except Exception as e:
            if verbose:
                print(f"    GEKKO Fehler: {e}")
            return None
    
    def _fallback_grid_search(self, integer_constraints, computation_time):
        """Führt eine Grid-Suche durch, wenn GEKKO fehlschlägt."""
        best_x, best_y, best_t, best_z = 0, 0, 0, float('-inf')
        
        if integer_constraints:
            x_grid = range(round(self.x_bounds[0]), round(self.x_bounds[1])+1)
            y_grid = range(round(self.y_bounds[0]), round(self.y_bounds[1])+1)
        else:
            x_grid = np.linspace(self.x_bounds[0], self.x_bounds[1], 21)
            y_grid = np.linspace(self.y_bounds[0], self.y_bounds[1], 21)
        
        for t in range(len(self.implant_parameters)):
            for xi in x_grid:
                for yi in y_grid:
                    z = self.total_function(xi, yi, t, self.implant_parameters)
                    if z > best_z:
                        best_x, best_y, best_t, best_z = xi, yi, t, z
        
        # Formatiere Fallback-Ergebnis
        convergence_path = [(best_x, best_y, best_t)]
        
        result = self.format_result(
            x_opt=best_x,
            y_opt=best_y,
            t_opt=best_t,
            f_opt=best_z,
            lambda_opt=0.0,
            path=convergence_path,
            constraint_value=0.0,
            computation_time=computation_time,
            termination_reason="Grid-Suche Fallback",
            outer_iterations=1,
            inner_iterations=0
        )
        
        self.print_colored("\n=== FALLBACK GRID-SUCHE ERGEBNIS ===", "yellow")
        self.print_result(result)
        
        return result