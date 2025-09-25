"""Gradient-Enhanced Particle Swarm Optimization Solver."""
import numpy as np
import time
from .base_solver import BaseSolver
from ..utils.numeric import numerical_gradient_with_t

class GradientEnhancedPSOSolver(BaseSolver):
    """PSO Solver erweitert um Gradienteninformation - mit korrekten Statistiken."""
    
    def solve(self, n_particles=20, max_iterations=50, c3_gradient=0.5, 
             gradient_probability=0.7, adaptive_gradient=True, verbose=True):
        """
        Löst das Optimierungsproblem mit Gradient-Enhanced PSO.
        
        Args:
            n_particles: Anzahl der Partikel im Schwarm
            max_iterations: Maximale Anzahl an Iterationen
            c3_gradient: Gewichtungsfaktor für den Gradiententerm
            gradient_probability: Wahrscheinlichkeit, Gradient zu verwenden
            adaptive_gradient: Ob der Gradientengewicht adaptiv angepasst wird
            verbose: Ausführliche Ausgabe aktivieren
            
        Returns:
            dict: Optimierungsergebnis
        """
        self.verbose = verbose
        
        self.print_colored(f"=== GRADIENT-ENHANCED PARTICLE SWARM OPTIMIZATION ===", "bold")
        print(f"Partikel: {n_particles}, Iterationen: {max_iterations}")
        print(f"Gradient-Gewicht c3: {c3_gradient}, Gradient-Wahrscheinlichkeit: {gradient_probability}")
        print(f"Adaptive Gradientenanpassung: {adaptive_gradient}")
        
        # PSO-Parameter
        w = 0.7      # Trägheitsgewicht
        c1 = 1.5     # Kognitives Gewicht
        c2 = 1.5     # Soziales Gewicht
        c3 = c3_gradient  # Gradientengewicht
        
        # KORRIGIERTE Statistiken für Vergleich
        gradient_attempts = 0           # Wie oft wurde versucht, Gradient zu verwenden
        gradient_successes = 0          # Wie oft war Gradientenberechnung erfolgreich
        gradient_improvements = 0       # Wie oft führte Gradient zu Verbesserung
        total_improvements = 0          # Gesamtzahl aller Verbesserungen
        
        start_time = time.time()
        
        # Partikel initialisieren (wie im Standard PSO)
        particles = self._initialize_particles(n_particles)
        
        # Globales Optimum initialisieren
        g_best_pos, g_best_val = self._find_initial_global_best(particles)
        convergence_path = [g_best_pos] if g_best_pos else []
        
        # Haupt-PSO-Schleife mit Gradientenverbesserung
        for iteration in range(max_iterations):
            improvements_this_iteration = 0
            gradients_used_this_iteration = 0
            gradient_improvements_this_iteration = 0
            
            # Adaptive Anpassung des Gradientengewichts
            if adaptive_gradient:
                # Reduziere Gradientengewicht mit der Zeit (Exploitation vs Exploration)
                progress = iteration / max_iterations
                c3_current = c3 * (1 - progress * 0.5)  # Reduziere um bis zu 50%
            else:
                c3_current = c3
            
            for i, p in enumerate(particles):
                old_f_val = p["f_val"]
                
                # Standard PSO Update
                x_new, y_new, t_new, vx_new, vy_new, vt_new = self._standard_pso_update(
                    p, g_best_pos, w, c1, c2
                )
                
                # FLAG: Wurde Gradient für dieses Partikel verwendet?
                gradient_used_for_this_particle = False
                
                # Gradientenverbesserung mit Wahrscheinlichkeit
                if np.random.random() < gradient_probability:
                    gradient_attempts += 1  # Versuch zählen
                    
                    try:
                        x_grad, y_grad, t_grad, vx_grad, vy_grad, vt_grad = self._gradient_enhanced_update(
                            x_new, y_new, t_new, vx_new, vy_new, vt_new, c3_current
                        )
                        
                        # Verwende Gradientenverbesserung
                        x_new, y_new, t_new = x_grad, y_grad, t_grad
                        vx_new, vy_new, vt_new = vx_grad, vy_grad, vt_grad
                        
                        gradients_used_this_iteration += 1
                        gradient_successes += 1  # Erfolgreiche Gradientenberechnung
                        gradient_used_for_this_particle = True
                        
                    except Exception as e:
                        if verbose and iteration < 10:
                            print(f"    Gradient-Berechnung fehlgeschlagen für Partikel {i}: {e}")
                
                # Bewerte neue Position
                f_val = self.total_function(x_new, y_new, t_new, self.implant_parameters)
                constraint_val = self.constraint_function(x_new, y_new, t_new, 
                                                        self.implant_parameters, self.z_threshold)
                is_feasible = constraint_val >= 0
                
                # Aktualisiere Partikel
                particles[i]["position"] = (x_new, y_new, t_new)
                particles[i]["velocity"] = (vx_new, vy_new, vt_new)
                particles[i]["f_val"] = f_val
                particles[i]["constraint_val"] = constraint_val
                particles[i]["is_feasible"] = is_feasible
                
                # KORRIGIERTE Verbesserungslogik
                if f_val > old_f_val:
                    improvements_this_iteration += 1
                    total_improvements += 1
                    
                    # NUR zählen als Gradient-Verbesserung wenn Gradient tatsächlich verwendet wurde
                    if gradient_used_for_this_particle:
                        gradient_improvements += 1
                        gradient_improvements_this_iteration += 1
                
                # Aktualisiere persönliches Optimum
                if is_feasible and f_val > p["p_best_val"]:
                    particles[i]["p_best_pos"] = (x_new, y_new, t_new)
                    particles[i]["p_best_val"] = f_val
                    
                    # Aktualisiere globales Optimum
                    if f_val > g_best_val:
                        g_best_val = f_val
                        g_best_pos = (x_new, y_new, t_new)
                        convergence_path.append(g_best_pos)
            
            # Mutation wie im Standard PSO
            if improvements_this_iteration == 0 and iteration > max_iterations // 3:
                self._apply_mutation(particles, max_iterations // 10)
            
            # Fortschrittsausgabe
            if iteration % 5 == 0 or iteration == max_iterations - 1:
                if g_best_pos:
                    g_best_x, g_best_y, g_best_t = g_best_pos
                    implant_name = self.implant_parameters[g_best_t]["name"]
                    
                    # KORRIGIERTE Gradient-Nutzungsberechnung
                    total_particle_updates = (iteration + 1) * n_particles
                    gradient_usage_percent = (gradient_successes / total_particle_updates) * 100
                    
                    print(f"Iter {iteration+1}: Optimum (x={g_best_x:.4f}, y={g_best_y:.4f}, "
                          f"t={g_best_t}({implant_name})) f={g_best_val:.6f}")
                    print(f"  Gradient-Nutzung: {gradient_usage_percent:.1f}%, "
                          f"Verbesserungen durch Gradient: {gradient_improvements}")
        
        # Erstelle Ergebnis
        end_time = time.time()
        computation_time = end_time - start_time
        
        if g_best_pos:
            x_opt, y_opt, t_opt = g_best_pos
            
            # KORRIGIERTE Statistiken für Vergleich
            total_particle_updates = max_iterations * n_particles
            gradient_usage_percent = (gradient_successes / total_particle_updates) * 100
            
            # KORRIGIERTE Erfolgsraten
            gradient_success_rate = (gradient_successes / max(gradient_attempts, 1)) * 100  # Erfolgreiche Berechnungen
            gradient_improvement_rate = (gradient_improvements / max(gradient_successes, 1)) * 100  # Verbesserungen bei erfolgreichen Gradienten
            
            result = self.format_result(
                x_opt=x_opt, y_opt=y_opt, t_opt=t_opt, f_opt=g_best_val,
                lambda_opt=0.0, path=convergence_path,
                constraint_value=self.constraint_function(x_opt, y_opt, t_opt, 
                                                        self.implant_parameters, self.z_threshold),
                computation_time=computation_time,
                termination_reason="Gradient-Enhanced PSO Maximum Iterations",
                outer_iterations=max_iterations, inner_iterations=0
            )
            
            # KORRIGIERTE Erweiterte Statistiken hinzufügen
            result["gradient_statistics"] = {
                "gradient_usage_percent": gradient_usage_percent,
                "gradient_success_rate": gradient_success_rate,           # % erfolgreiche Berechnungen
                "gradient_improvement_rate": gradient_improvement_rate,   # % Verbesserungen bei erfolgreichen Gradienten
                "total_gradient_attempts": gradient_attempts,             # Gesamtversuche
                "total_gradient_successes": gradient_successes,           # Erfolgreiche Berechnungen
                "total_gradient_calls": gradient_successes,               # Alias für Kompatibilität
                "gradient_improvements": gradient_improvements,           # Verbesserungen durch Gradient
                "total_improvements": total_improvements                  # Alle Verbesserungen
            }
            
            self.print_colored("\n=== GRADIENT-ENHANCED PSO ERGEBNIS ===", "green")
            self.print_result(result)
            
            print(f"\nKorrigierte Gradientenstatistiken:")
            print(f"  Gradient-Versuche: {gradient_attempts}")
            print(f"  Gradient-Erfolge: {gradient_successes} ({gradient_success_rate:.1f}%)")
            print(f"  Gradient-Nutzung: {gradient_usage_percent:.1f}% aller Partikel-Updates")
            print(f"  Verbesserungen durch Gradient: {gradient_improvements}/{gradient_successes} ({gradient_improvement_rate:.1f}%)")
            print(f"  Gesamtverbesserungen: {total_improvements}")
            
            return result
        else:
            print("Keine zulässige Lösung gefunden!")
            return None
    
    def _initialize_particles(self, n_particles):
        """Initialisiert Partikel wie im Standard PSO."""
        particles = []
        implantat_typen = len(self.implant_parameters)
        particles_per_type = max(2, n_particles // implantat_typen)
        remaining_particles = n_particles - (particles_per_type * implantat_typen)
        
        # Systematische Verteilung über Implantattypen
        for t in range(implantat_typen):
            for _ in range(particles_per_type):
                x = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
                y = np.random.uniform(self.y_bounds[0], self.y_bounds[1])
                
                vx = np.random.uniform(-0.5, 0.5)
                vy = np.random.uniform(-0.5, 0.5)
                vt = np.random.uniform(-0.5, 0.5)
                
                f_val = self.total_function(x, y, t, self.implant_parameters)
                constraint_val = self.constraint_function(x, y, t, self.implant_parameters, self.z_threshold)
                is_feasible = constraint_val >= 0
                
                particles.append({
                    "position": (x, y, t),
                    "velocity": (vx, vy, vt),
                    "f_val": f_val,
                    "constraint_val": constraint_val,
                    "is_feasible": is_feasible,
                    "p_best_pos": (x, y, t),
                    "p_best_val": f_val if is_feasible else float('-inf')
                })
        
        # Restliche Partikel
        for _ in range(remaining_particles):
            x = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
            y = np.random.uniform(self.y_bounds[0], self.y_bounds[1])
            t = np.random.randint(0, implantat_typen)
            
            vx = np.random.uniform(-0.5, 0.5)
            vy = np.random.uniform(-0.5, 0.5)
            vt = np.random.uniform(-0.5, 0.5)
            
            f_val = self.total_function(x, y, t, self.implant_parameters)
            constraint_val = self.constraint_function(x, y, t, self.implant_parameters, self.z_threshold)
            is_feasible = constraint_val >= 0
            
            particles.append({
                "position": (x, y, t),
                "velocity": (vx, vy, vt),
                "f_val": f_val,
                "constraint_val": constraint_val,
                "is_feasible": is_feasible,
                "p_best_pos": (x, y, t),
                "p_best_val": f_val if is_feasible else float('-inf')
            })
        
        return particles
    
    def _find_initial_global_best(self, particles):
        """Findet das initiale globale Optimum."""
        g_best_pos = None
        g_best_val = float('-inf')
        
        for p in particles:
            if p["is_feasible"] and p["f_val"] > g_best_val:
                g_best_val = p["f_val"]
                g_best_pos = p["position"]
        
        return g_best_pos, g_best_val
    
    def _standard_pso_update(self, particle, g_best_pos, w, c1, c2):
        """Standard PSO Geschwindigkeits- und Positionsupdate."""
        x, y, t = particle["position"]
        vx, vy, vt = particle["velocity"]
        p_best_x, p_best_y, p_best_t = particle["p_best_pos"]
        
        # Zufällige Faktoren
        r1, r2 = np.random.random(2)
        
        # Standard PSO Geschwindigkeitsupdate
        if g_best_pos:
            g_best_x, g_best_y, g_best_t = g_best_pos
            vx_new = w * vx + c1 * r1 * (p_best_x - x) + c2 * r2 * (g_best_x - x)
            vy_new = w * vy + c1 * r1 * (p_best_y - y) + c2 * r2 * (g_best_y - y)
            vt_new = w * vt + c1 * r1 * (p_best_t - t) + c2 * r2 * (g_best_t - t)
        else:
            vx_new = w * vx + c1 * r1 * (p_best_x - x)
            vy_new = w * vy + c1 * r1 * (p_best_y - y)
            vt_new = w * vt + c1 * r1 * (p_best_t - t)
        
        # Neue Position berechnen
        x_new = x + vx_new
        y_new = y + vy_new
        t_new_float = t + vt_new
        
        # Auf Grenzen beschränken
        x_new = np.clip(x_new, self.x_bounds[0], self.x_bounds[1])
        y_new = np.clip(y_new, self.y_bounds[0], self.y_bounds[1])
        
        # Probabilistische Rundung für t
        t_frac = t_new_float - int(t_new_float)
        t_new = int(t_new_float) + (1 if np.random.random() < t_frac else 0)
        t_max = len(self.implant_parameters) - 1
        t_new = max(0, min(t_new, t_max))
        
        return x_new, y_new, t_new, vx_new, vy_new, vt_new
    
    def _gradient_enhanced_update(self, x, y, t, vx, vy, vt, c3):
        """Gradientenverbessertes Update."""
        # Berechne numerischen Gradienten für x und y
        grad_xy = numerical_gradient_with_t(x, y, t, self.total_function, self.implant_parameters)
        
        # Diskrete "Gradient"-Approximation für t
        grad_t = self._compute_discrete_gradient_t(x, y, t)
        
        # Gradiententerm zur Geschwindigkeit hinzufügen
        r3 = np.random.random()
        
        vx_grad = vx + c3 * r3 * grad_xy[0]
        vy_grad = vy + c3 * r3 * grad_xy[1]
        vt_grad = vt + c3 * r3 * grad_t
        
        # Neue Position mit Gradientenverbesserung
        x_grad = x + vx_grad
        y_grad = y + vy_grad
        t_grad_float = t + vt_grad
        
        # Grenzen anwenden
        x_grad = np.clip(x_grad, self.x_bounds[0], self.x_bounds[1])
        y_grad = np.clip(y_grad, self.y_bounds[0], self.y_bounds[1])
        
        # t runden
        t_frac = t_grad_float - int(t_grad_float)
        t_grad = int(t_grad_float) + (1 if np.random.random() < t_frac else 0)
        t_max = len(self.implant_parameters) - 1
        t_grad = max(0, min(t_grad, t_max))
        
        return x_grad, y_grad, t_grad, vx_grad, vy_grad, vt_grad
    
    def _compute_discrete_gradient_t(self, x, y, t):
        """Berechnet eine diskrete Gradientenapproximation für t."""
        current_val = self.total_function(x, y, t, self.implant_parameters)
        
        # Berechne Werte bei t+1 und t-1
        next_val = None
        if t < len(self.implant_parameters) - 1:
            next_val = self.total_function(x, y, t + 1, self.implant_parameters)
        
        prev_val = None
        if t > 0:
            prev_val = self.total_function(x, y, t - 1, self.implant_parameters)
        
        # Zentrale Differenz, falls möglich
        if next_val is not None and prev_val is not None:
            grad_t = (next_val - prev_val) / 2.0
        elif next_val is not None:
            grad_t = next_val - current_val
        elif prev_val is not None:
            grad_t = current_val - prev_val
        else:
            grad_t = 0.0
        
        return grad_t
    
    def _apply_mutation(self, particles, num_mutations):
        """Anwendung von Mutation wie im Standard PSO."""
        mutation_indices = np.random.choice(range(len(particles)), num_mutations, replace=False)
        
        for idx in mutation_indices:
            x_new = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
            y_new = np.random.uniform(self.y_bounds[0], self.y_bounds[1])
            t_new = np.random.randint(0, len(self.implant_parameters))
            
            f_val = self.total_function(x_new, y_new, t_new, self.implant_parameters)
            constraint_val = self.constraint_function(x_new, y_new, t_new, 
                                                    self.implant_parameters, self.z_threshold)
            is_feasible = constraint_val >= 0
            
            particles[idx]["position"] = (x_new, y_new, t_new)
            particles[idx]["velocity"] = (np.random.uniform(-0.5, 0.5), 
                                         np.random.uniform(-0.5, 0.5),
                                         np.random.uniform(-0.5, 0.5))
            particles[idx]["f_val"] = f_val
            particles[idx]["constraint_val"] = constraint_val
            particles[idx]["is_feasible"] = is_feasible
            
            if is_feasible and f_val > particles[idx]["p_best_val"]:
                particles[idx]["p_best_pos"] = (x_new, y_new, t_new)
                particles[idx]["p_best_val"] = f_val