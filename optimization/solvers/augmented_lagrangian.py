"""Solver basierend auf der Augmented-Lagrangian-Methode."""
import numpy as np
import time
from .base_solver import BaseSolver
from ..utils.numeric import augmented_lagrangian_with_t, augmented_lagrangian_gradient, compute_discrete_direction_t

class AugmentedLagrangianSolver(BaseSolver):
    """Solver mit der Augmented-Lagrangian-Methode."""
    
    def solve(self, starting_points=None, strategy="combined", num_points=10, 
            lambda0=0.0, mu0=1.0, beta=10.0, max_outer_iter=20, max_inner_iter=50, tol=1e-6, 
            verbose=True):
        """
        Löst das Optimierungsproblem mit der Augmented-Lagrangian-Methode.
        
        Args:
            starting_points: Liste von Startpunkten, falls bereits vorhanden
            strategy: Strategie für Startwerte ('fixed', 'random', 'combined'), falls keine Punkte gegeben
            num_points: Anzahl der Startwerte (für 'random' und 'combined')
            lambda0: Initialer Lagrange-Multiplikator
            mu0: Initialer Penalty-Parameter
            beta: Wachstumsfaktor für den Penalty-Parameter
            max_outer_iter: Maximale Anzahl äußerer Iterationen
            max_inner_iter: Maximale Anzahl innerer Iterationen
            tol: Toleranz für Konvergenz
            verbose: Ausführliche Ausgabe (True/False)
            
        Returns:
            dict: Optimierungsergebnis oder None, falls keine zulässige Lösung gefunden
        """
        self.verbose = verbose
        
        # Startwerte holen, falls nicht explizit gegeben
        if starting_points is None:
            from ..strategies.starting_points import get_starting_points
            starting_points = get_starting_points(
                strategy=strategy, 
                num_points=num_points, 
                x_bounds=self.x_bounds,
                y_bounds=self.y_bounds,
                t_bounds=self.t_bounds,
                implant_parameters=self.implant_parameters
            )
        
        # Ausgabe der Startbedingungen
        self.print_colored(f"\n=== OPTIMIERUNG MIT AUGMENTED LAGRANGIAN UND {len(starting_points)} STARTPUNKTEN ===", "bold")
        print(f"Nebenbedingung: f(x,y,t) ≥ {self.z_threshold}")
        print(f"Optimierungsbereich: x: [{self.x_bounds[0]}, {self.x_bounds[1]}], " +
              f"y: [{self.y_bounds[0]}, {self.y_bounds[1]}], " +
              f"t: [{self.t_bounds[0]}, {self.t_bounds[1]}] (ganzzahlig)")
        
        # Optimierung für jeden Startwert durchführen
        results = []
        start_time_total = time.time()
        
        for i, start_point in enumerate(starting_points):
            x0, y0, t0 = start_point["x0"], start_point["y0"], start_point["t0"]
            
            if verbose:
                print(f"\nStartpunkt {i+1}/{len(starting_points)}: '{start_point['name']}' bei (x={x0:.4f}, y={y0:.4f}, t={t0})...")
            
            try:
                # Einzelne Optimierung durchführen
                start_time = time.time()
                
                x_opt, y_opt, t_opt, f_opt, lambda_opt, implant_params_opt, path, termination_reason, outer_iterations, inner_iterations = self._augmented_lagrangian_method(
                    x0=x0, y0=y0, t0=t0, 
                    lambda0=lambda0, mu0=mu0, beta=beta, 
                    max_outer_iter=max_outer_iter, max_inner_iter=max_inner_iter, 
                    tol=tol, verbose=verbose and i < 3  # Nur für die ersten Punkte ausführlich
                )
                
                end_time = time.time()
                computation_time = end_time - start_time
                
                # Nebenbedingung prüfen
                constraint_value = self.constraint_function(x_opt, y_opt, t_opt, self.implant_parameters, self.z_threshold)
                
                # Ergebnis formatieren und speichern
                result = self.format_result(
                    x_opt, y_opt, t_opt, f_opt, lambda_opt, path,
                    constraint_value, computation_time, termination_reason,
                    outer_iterations, inner_iterations
                )
                
                # Zusätzliche Informationen für diesen speziellen Lauf
                result["start_point"] = start_point
                
                results.append(result)
                
                if verbose:
                    print(f"  Optimum gefunden bei: (x={x_opt:.4f}, y={y_opt:.4f}, t={t_opt})")
                    print(f"  Implantattyp: {self.implant_parameters[t_opt]['name']}")
                    print(f"  Zielfunktionswert: f(x,y,t) = {f_opt:.6f}")
                    print(f"  Nebenbedingung: {constraint_value:.6f}, Zulässig: {constraint_value >= -1e-6}")
                
            except Exception as e:
                if verbose:
                    print(f"  Fehler bei Startpunkt {start_point['name']}: {str(e)}")
        
        end_time_total = time.time()
        computation_time_total = end_time_total - start_time_total
        
        # Zulässige Ergebnisse filtern
        feasible_results = [r for r in results if r["is_feasible"]]
        
        if feasible_results:
            # Bestes Ergebnis auswählen (höchster Funktionswert)
            best_result = max(feasible_results, key=lambda r: r["f_opt"])
            
            # Ausgabe des Gesamtergebnisses
            self.print_colored("\n=== ERGEBNIS DER OPTIMIERUNG MIT AUGMENTED LAGRANGIAN ===", "green")
            print(f"Bester Startpunkt: {best_result['start_point']['name']} " +
                 f"(x={best_result['start_point']['x0']}, y={best_result['start_point']['y0']}, t={best_result['start_point']['t0']})")
            
            # Detaillierte Ausgabe
            self.print_result(best_result)
            
            # Zusammenfassung aller getesteten Startwerte
            if verbose:
                self.print_colored("\nZusammenfassung aller getesteten Startwerte:", "cyan")
                print("{:<30} {:<10} {:<10} {:<10} {:<15} {:<15} {:<15} {:<15}".format(
                    "Startpunkt", "x_opt", "y_opt", "t_opt", "f(x,y,t)", "g(x,y,t)", "Zeit (s)", "Schritte"))
                print("-" * 130)
                
                for r in results:
                    if r["is_feasible"]:
                        status = "✓"
                    else:
                        status = "✗"
                        
                    print("{:<30} {:<10.4f} {:<10.4f} {:<10.0f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15}".format(
                        r["start_point"]["name"], 
                        r["x_opt"], 
                        r["y_opt"], 
                        r["t_opt"],
                        r["f_opt"], 
                        r["constraint_value"],
                        r["computation_time"],
                        r["total_steps"]))
                
                print(f"\nGesamtberechnungszeit für alle Startwerte: {computation_time_total:.6f} Sekunden")
            
            return best_result
        else:
            self.print_colored("\nKeine zulässige Lösung gefunden! Alle getesteten Startpunkte führten zu ungültigen Lösungen.", "red")
            return None
    
    def _gradient_descent_step(self, x, y, t, lambda_k, mu_k, learning_rate=0.1, max_iter=100, tol=1e-6, verbose=False):
        """Führt einen internen Optimierungsschritt mit Gradient Descent aus."""
        
        if verbose:
            print(f"  Start innere Iteration: x={x:.4f}, y={y:.4f}, t={t}")
        
        x_k, y_k = x, y
        t_k = t
        
        path = [(x_k, y_k, t_k)]
        
        # Initialisiere step_size_xy mit einem Wert >> tol
        step_size_xy = float('inf')
        
        # Informationen zum Abbruchkriterium
        termination_reason = "Maximale Iterationen erreicht"
        iterations_completed = 0
        
        # Flag für Änderung von t
        t_changed = False
        
        for i in range(max_iter):
            iterations_completed = i + 1
            
            # 1. Optimiere x und y bei festem t
            grad = augmented_lagrangian_gradient(
                x_k, y_k, t_k, lambda_k, mu_k, 
                self.total_function, self.constraint_function, 
                self.implant_parameters, self.z_threshold
            )
            
            # Maximierung statt Minimierung, daher plus statt minus
            x_new = x_k + learning_rate * grad[0]
            y_new = y_k + learning_rate * grad[1]
            
            # Beschränke auf den zulässigen Bereich
            x_new = max(min(x_new, self.x_bounds[1]), self.x_bounds[0])
            y_new = max(min(y_new, self.y_bounds[1]), self.y_bounds[0])
            
            # Berechne die Schrittweite in x,y - Vor der t-Optimierung!
            step_size_xy = np.linalg.norm([x_new - x_k, y_new - y_k])
            
            # 2. Optimiere t (diskret) alle 5 Iterationen oder nach einer Änderung
            t_new = t_k  # Standardmäßig keine Änderung
            
            if i % 5 == 0 or (i > 0 and i % 5 == 1 and t_changed):
                # Verwende die Funktion zur Bestimmung der Richtung basierend auf der größten Steigung
                direction = compute_discrete_direction_t(
                    x_new, y_new, t_k, augmented_lagrangian_with_t, 
                    lambda_k, mu_k, self.total_function, self.constraint_function, 
                    self.implant_parameters, self.z_threshold
                )
                
                t_new = t_k + direction
                t_new = max(min(t_new, self.t_bounds[1]), self.t_bounds[0])  # Beschränke auf gültige Werte
                
                t_changed = (t_new != t_k)
                
                # Informationen zur t-Optimierung
                if verbose and (i < 20 or t_changed):
                    f_old = augmented_lagrangian_with_t(
                        x_new, y_new, t_k, lambda_k, mu_k, 
                        self.total_function, self.constraint_function, 
                        self.implant_parameters, self.z_threshold
                    )
                    f_new = augmented_lagrangian_with_t(
                        x_new, y_new, t_new, lambda_k, mu_k, 
                        self.total_function, self.constraint_function, 
                        self.implant_parameters, self.z_threshold
                    )
                    
                    delta_f = f_new - f_old
                    print(f"    [Iter {i+1}] Wechsel t: {t_k} -> {t_new}, x={x_new:.4f}, y={y_new:.4f}, "
                        f"f={f_new:.6f}, c={f_new-self.z_threshold:.6f}")
            
            # Aktualisiere t_k, wenn t geändert wurde
            if t_changed:
                t_k = t_new
            
            # Füge den neuen Punkt zum Pfad hinzu
            path.append((x_new, y_new, t_new))
            
            # Informationen zum Fortschritt, falls t nicht geändert wurde
            if verbose and (i % 10 == 0 or i == max_iter - 1) and not t_changed:
                f_val = self.total_function(x_new, y_new, t_new, self.implant_parameters)
                c_val = self.constraint_function(x_new, y_new, t_new, self.implant_parameters, self.z_threshold)
                print(f"    [Iter {i+1}] x={x_new:.4f}, y={y_new:.4f}, t={t_new}, "
                    f"f={f_val:.6f}, c={c_val:.6f}, Schritt={step_size_xy:.8f}")
            
            # Überprüfe Konvergenz in x und y
            if step_size_xy < tol and not t_changed and i > 10:
                termination_reason = f"Konvergenz in x,y (Schrittweite = {step_size_xy:.8f}, tol = {tol}) und t stabil bei {t_k}"
                break
            
            x_k, y_k = x_new, y_new
        
        if verbose:
            print(f"  Position: (x={x_k:.4f}, y={y_k:.4f}, t={t_k}) " 
                f"f={self.total_function(x_k, y_k, t_k, self.implant_parameters):.6f}, "
                f"g={self.constraint_function(x_k, y_k, t_k, self.implant_parameters, self.z_threshold):.6f}")
        
        return x_k, y_k, t_k, path, termination_reason, iterations_completed
    
    def _augmented_lagrangian_method(self, x0=0.0, y0=0.0, t0=0, lambda0=0.0, mu0=1.0,
                                   beta=10.0, max_outer_iter=20, max_inner_iter=100,
                                   tol=1e-6, verbose=False):
        """Implementiert die Augmented-Lagrangian-Methode mit t als diskrete ganzzahlige Optimierungsvariable."""
        x_k, y_k = x0, y0
        t_k = t0
        lambda_k = lambda0
        mu_k = mu0
        
        complete_path = [(x_k, y_k, t_k)]
        
        # Terminierungsinformationen
        termination_reason = "Maximale Iterationen erreicht"
        outer_iterations_completed = 0
        total_inner_iterations = 0
        
        for k in range(max_outer_iter):
            if verbose:
                print(f"\n=== Äußere Iteration {k+1}/{max_outer_iter} ===")
                print(f"  Lagrange-Multiplikator λ = {lambda_k:.6f}, Penalty-Parameter μ = {mu_k:.6f}")
            
            outer_iterations_completed = k + 1
            
            # Minimiere die Augmented-Lagrangian-Funktion mit festen lambda_k und mu_k
            x_k, y_k, t_k, inner_path, inner_termination, inner_iterations = self._gradient_descent_step(
                x_k, y_k, t_k, lambda_k, mu_k,
                max_iter=max_inner_iter, tol=tol, verbose=verbose
            )
            
            # Füge inneren Pfad zum vollständigen Pfad hinzu
            complete_path.extend(inner_path[1:])  # Erstes Element ist bereits enthalten
            
            # Zähle innere Iterationen
            total_inner_iterations += inner_iterations
            
            # Aktuelle Nebenbedingungswerte
            g_val = self.constraint_function(x_k, y_k, t_k, self.implant_parameters, self.z_threshold)
            
            if verbose:
                print(f"  Position: (x={x_k:.4f}, y={y_k:.4f}, t={t_k}) "
                    f"f={self.total_function(x_k, y_k, t_k, self.implant_parameters):.6f}, g={g_val:.6f}")
            
            # Prüfe Konvergenz der Nebenbedingung
            if abs(min(0, g_val)) < tol and k > 0:
                termination_reason = f"Konvergenz der Nebenbedingung (g = {g_val:.8f}, tol = {tol})"
                if verbose:
                    print(f"  ABBRUCH: {termination_reason}")
                break
            
            # Aktualisiere Lagrange-Multiplikator
            lambda_new = max(0, lambda_k - mu_k * g_val)
            
            # Prüfe Konvergenz des Lagrange-Multiplikators
            if abs(lambda_new - lambda_k) < tol and k > 0:
                termination_reason = f"Konvergenz des Lagrange-Multiplikators (Δλ = {abs(lambda_new - lambda_k):.8f}, tol = {tol})"
                if verbose:
                    print(f"  ABBRUCH: {termination_reason}")
                break
            
            if verbose:
                print(f"  Update: λ = {lambda_k:.6f} -> {lambda_new:.6f}, μ = {mu_k:.6f} -> {beta * mu_k:.6f}")
            
            lambda_k = lambda_new
            
            # Erhöhe Penalty-Parameter
            mu_k = beta * mu_k
        
        # Berechne endgültigen Funktionswert
        f_val = self.total_function(x_k, y_k, t_k, self.implant_parameters)
        
        # Verwende endgültige Implantat-Parameter
        final_implant_params = self.implant_parameters[t_k]
        
        return x_k, y_k, t_k, f_val, lambda_k, final_implant_params, complete_path, termination_reason, outer_iterations_completed, total_inner_iterations