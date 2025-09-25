
"""Solver basierend auf Particle Swarm Optimization."""
import numpy as np
import time
from .base_solver import BaseSolver

class ParticleSwarmSolver(BaseSolver):
    """Solver mit der Particle-Swarm-Optimization (PSO)."""
    
    def solve(self, n_particles=20, max_iterations=50, verbose=True):
        """
        Löst das Optimierungsproblem mit Particle Swarm Optimization.
        
        Args:
            n_particles: Anzahl der Partikel im Schwarm
            max_iterations: Maximale Anzahl an Iterationen
            verbose: Ausführliche Ausgabe aktivieren
            
        Returns:
            dict: Optimierungsergebnis oder None, falls keine zulässige Lösung gefunden
        """
        self.verbose = verbose
        
        self.print_colored(f"=== PARTICLE SWARM OPTIMIZATION MIT {n_particles} PARTIKELN ===", "bold")
        print(f"Nebenbedingung: f(x,y,t) ≥ {self.z_threshold}")
        print(f"Optimierungsbereich: x: [{self.x_bounds[0]}, {self.x_bounds[1]}], " +
              f"y: [{self.y_bounds[0]}, {self.y_bounds[1]}], " +
              f"t: [0, {len(self.implant_parameters)-1}] (ganzzahlig)")
        
        # PSO-Parameter - angepasst für bessere Exploration
        w = 0.7      # Trägheitsgewicht
        c1 = 1.5     # Kognitives Gewicht (persönliches Optimum)
        c2 = 1.5     # Soziales Gewicht (globales Optimum)
        
        if verbose:
            print("\n=== DETAIL: PARTICLE SWARM OPTIMIZATION ===")
            print(f"Partikel: {n_particles}, Max. Iterationen: {max_iterations}")
            print(f"Parameter: w={w}, c1={c1}, c2={c2}")
            print(f"Initialisiere Partikelschwarm...")
        
        start_time = time.time()
        
        # Partikel initialisieren - WICHTIG: Bessere Verteilung über den Suchraum
        particles = []
        
        # Stelle sicher, dass wir jeden Implantattyp testen
        implantat_typen = len(self.implant_parameters)
        particles_per_type = max(2, n_particles // implantat_typen)
        remaining_particles = n_particles - (particles_per_type * implantat_typen)
        
        # Zuerst systematisch für jeden Implantattyp Partikel erstellen
        for t in range(implantat_typen):
            for _ in range(particles_per_type):
                # Zufällige Position für x und y
                x = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
                y = np.random.uniform(self.y_bounds[0], self.y_bounds[1])
                
                # Geschwindigkeit
                vx = np.random.uniform(-0.5, 0.5)
                vy = np.random.uniform(-0.5, 0.5)
                vt = np.random.uniform(-0.5, 0.5)
                
                # Bewerte Position
                f_val = self.total_function(x, y, t, self.implant_parameters)
                constraint_val = self.constraint_function(x, y, t, self.implant_parameters, self.z_threshold)
                is_feasible = constraint_val >= 0
                
                # Initialisiere persönliches Optimum
                p_best_pos = (x, y, t)
                p_best_val = f_val if is_feasible else float('-inf')
                
                particles.append({
                    "position": (x, y, t),
                    "velocity": (vx, vy, vt),
                    "f_val": f_val,
                    "constraint_val": constraint_val,
                    "is_feasible": is_feasible,
                    "p_best_pos": p_best_pos,
                    "p_best_val": p_best_val
                })
        
        # Restliche Partikel zufällig verteilen
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
            
            p_best_pos = (x, y, t)
            p_best_val = f_val if is_feasible else float('-inf')
            
            particles.append({
                "position": (x, y, t),
                "velocity": (vx, vy, vt),
                "f_val": f_val,
                "constraint_val": constraint_val,
                "is_feasible": is_feasible,
                "p_best_pos": p_best_pos,
                "p_best_val": p_best_val
            })
        
        # Initialisiere globales Optimum
        g_best_pos = None
        g_best_val = float('-inf')
        
        # Finde das erste globale Optimum unter den initialen Partikeln
        for p in particles:
            if p["is_feasible"] and p["f_val"] > g_best_val:
                g_best_val = p["f_val"]
                g_best_pos = p["position"]
        
        # Liste der Konvergenzpunkte für die Visualisierung
        convergence_path = []
        if g_best_pos:
            convergence_path.append(g_best_pos)
        
        # Haupt-PSO-Schleife
        for iteration in range(max_iterations):
            # Zähler für Verbesserungen in dieser Iteration
            improvements = 0
            
            for i, p in enumerate(particles):
                # Extrahiere aktuelle Position, Geschwindigkeit und persönliches Optimum
                x, y, t = p["position"]
                vx, vy, vt = p["velocity"]
                p_best_x, p_best_y, p_best_t = p["p_best_pos"]
                
                # Zufällige Faktoren
                r1, r2 = np.random.random(2)
                
                # Berechne neue Geschwindigkeit
                if g_best_pos:
                    g_best_x, g_best_y, g_best_t = g_best_pos
                    
                    # Standard-PSO-Geschwindigkeitsaktualisierung
                    vx_new = w * vx + c1 * r1 * (p_best_x - x) + c2 * r2 * (g_best_x - x)
                    vy_new = w * vy + c1 * r1 * (p_best_y - y) + c2 * r2 * (g_best_y - y)
                    vt_new = w * vt + c1 * r1 * (p_best_t - t) + c2 * r2 * (g_best_t - t)
                else:
                    # Wenn noch kein globales Optimum gefunden wurde, verwende nur persönliches Optimum
                    vx_new = w * vx + c1 * r1 * (p_best_x - x)
                    vy_new = w * vy + c1 * r1 * (p_best_y - y)
                    vt_new = w * vt + c1 * r1 * (p_best_t - t)
                
                # Berechne neue Position
                x_new = x + vx_new
                y_new = y + vy_new
                t_new_float = t + vt_new
                
                # Auf Grenzen beschränken
                x_new = np.clip(x_new, self.x_bounds[0], self.x_bounds[1])
                y_new = np.clip(y_new, self.y_bounds[0], self.y_bounds[1])
                
                # WICHTIG: Verschiedene Strategien für die Rundung von t
                # Probabilistische Rundung für bessere Exploration des diskreten Raums
                t_frac = t_new_float - int(t_new_float)
                t_new = int(t_new_float) + (1 if np.random.random() < t_frac else 0)
                
                # Anstatt t einfach zu beschränken, wende modulo an,
                # so dass Werte außerhalb des gültigen Bereichs auf der anderen Seite wieder hereinkommen
                t_max = len(self.implant_parameters) - 1
                if t_new > t_max:
                    t_new = t_new % (t_max + 1)
                elif t_new < 0:
                    t_new = (t_new % (t_max + 1)) + t_max + 1
                    
                t_new = max(0, min(t_new, t_max))  # Sicherheitscheck
                
                # Bewerte neue Position
                f_val = self.total_function(x_new, y_new, t_new, self.implant_parameters)
                constraint_val = self.constraint_function(x_new, y_new, t_new, self.implant_parameters, self.z_threshold)
                is_feasible = constraint_val >= 0
                
                # Aktualisiere Partikel
                particles[i]["position"] = (x_new, y_new, t_new)
                particles[i]["velocity"] = (vx_new, vy_new, vt_new)
                particles[i]["f_val"] = f_val
                particles[i]["constraint_val"] = constraint_val
                particles[i]["is_feasible"] = is_feasible
                
                # Aktualisiere persönliches Optimum, wenn die neue Position besser und zulässig ist
                if is_feasible and f_val > p["p_best_val"]:
                    particles[i]["p_best_pos"] = (x_new, y_new, t_new)
                    particles[i]["p_best_val"] = f_val
                    improvements += 1
                    
                    # Aktualisiere globales Optimum, wenn die neue Position besser ist
                    if f_val > g_best_val:
                        g_best_val = f_val
                        g_best_pos = (x_new, y_new, t_new)
                        
                        # Zum Konvergenzpfad hinzufügen
                        convergence_path.append(g_best_pos)
            
            # Rein zufällige Mutation für einige Partikel, um Stagnation zu vermeiden
            if improvements == 0 and iteration > max_iterations // 3:
                # Wenn keine Verbesserungen, führe zusätzliche Exploration durch
                num_mutations = max(2, n_particles // 10)
                mutation_indices = np.random.choice(range(n_particles), num_mutations, replace=False)
                
                for idx in mutation_indices:
                    # Komplett neue Position für das Partikel
                    x_new = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
                    y_new = np.random.uniform(self.y_bounds[0], self.y_bounds[1])
                    t_new = np.random.randint(0, len(self.implant_parameters))
                    
                    # Bewerte neue Position
                    f_val = self.total_function(x_new, y_new, t_new, self.implant_parameters)
                    constraint_val = self.constraint_function(x_new, y_new, t_new, self.implant_parameters, self.z_threshold)
                    is_feasible = constraint_val >= 0
                    
                    # Aktualisiere Partikel
                    particles[idx]["position"] = (x_new, y_new, t_new)
                    particles[idx]["velocity"] = (np.random.uniform(-0.5, 0.5), 
                                                 np.random.uniform(-0.5, 0.5),
                                                 np.random.uniform(-0.5, 0.5))
                    particles[idx]["f_val"] = f_val
                    particles[idx]["constraint_val"] = constraint_val
                    particles[idx]["is_feasible"] = is_feasible
                    
                    # Aktualisiere persönliches Optimum, wenn die neue Position besser und zulässig ist
                    if is_feasible and f_val > particles[idx]["p_best_val"]:
                        particles[idx]["p_best_pos"] = (x_new, y_new, t_new)
                        particles[idx]["p_best_val"] = f_val
                        
                        # Aktualisiere globales Optimum, wenn die neue Position besser ist
                        if f_val > g_best_val:
                            g_best_val = f_val
                            g_best_pos = (x_new, y_new, t_new)
                            
                            # Zum Konvergenzpfad hinzufügen
                            convergence_path.append(g_best_pos)
            
            # Ausgabe des aktuellen Status
            if iteration % 5 == 0 or iteration == max_iterations - 1:
                if g_best_pos:
                    g_best_x, g_best_y, g_best_t = g_best_pos
                    implant_name = self.implant_parameters[g_best_t]["name"]
                    print(f"Iteration {iteration+1}/{max_iterations}: Bestes Optimum bei (x={g_best_x:.4f}, y={g_best_y:.4f}, t={g_best_t} ({implant_name})) mit f={g_best_val:.6f}")
                else:
                    print(f"Iteration {iteration+1}/{max_iterations}: Noch keine zulässige Lösung gefunden")
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Ausgabe der endgültigen Ergebnisse
        if g_best_pos:
            x_opt, y_opt, t_opt = g_best_pos
            
            # Erstelle Ergebnisobjekt
            result = self.format_result(
                x_opt=x_opt,
                y_opt=y_opt,
                t_opt=t_opt,
                f_opt=g_best_val,
                lambda_opt=0.0,  # PSO verwendet keinen Lagrange-Multiplikator
                path=convergence_path,
                constraint_value=self.constraint_function(x_opt, y_opt, t_opt, self.implant_parameters, self.z_threshold),
                computation_time=computation_time,
                termination_reason="Maximale Iterationen erreicht",
                outer_iterations=max_iterations,
                inner_iterations=0
            )
            
            self.print_colored("\n=== ERGEBNIS DER PARTICLE SWARM OPTIMIZATION ===", "green")
            self.print_result(result)
            
            return result
        else:
            print("\nKeine zulässige Lösung gefunden!")
            return None














'''
"""Solver basierend auf Particle Swarm Optimization."""
import numpy as np
import time
from .base_solver import BaseSolver

class ParticleSwarmSolver(BaseSolver):
    """Solver mit der Particle-Swarm-Optimization (PSO)."""
    
    def solve(self, n_particles=20, max_iterations=50, verbose=True):
        """
        Löst das Optimierungsproblem mit Particle Swarm Optimization.
        
        Args:
            n_particles: Anzahl der Partikel im Schwarm
            max_iterations: Maximale Anzahl an Iterationen
            verbose: Ausführliche Ausgabe aktivieren
            
        Returns:
            dict: Optimierungsergebnis oder None, falls keine zulässige Lösung gefunden
        """
        self.verbose = verbose
        
        self.print_colored(f"=== PARTICLE SWARM OPTIMIZATION MIT {n_particles} PARTIKELN ===", "bold")
        print(f"Nebenbedingung: f(x,y,t) ≥ {self.z_threshold}")
        print(f"Optimierungsbereich: x: [{self.x_bounds[0]}, {self.x_bounds[1]}], " +
              f"y: [{self.y_bounds[0]}, {self.y_bounds[1]}], " +
              f"t: [0, {len(self.implant_parameters)-1}] (ganzzahlig)")
        
        # PSO-Parameter
        w = 0.7      # Trägheitsgewicht
        c1 = 1.5     # Kognitives Gewicht (persönliches Optimum)
        c2 = 1.5     # Soziales Gewicht (globales Optimum)
        
        if verbose:
            print("\n=== DETAIL: PARTICLE SWARM OPTIMIZATION ===")
            print(f"Partikel: {n_particles}, Max. Iterationen: {max_iterations}")
            print(f"Initialisiere Partikelschwarm...")
        
        start_time = time.time()
        
        # Partikel initialisieren
        particles = []
        for i in range(n_particles):
            # Zufällige Position (x, y, t)
            x = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
            y = np.random.uniform(self.y_bounds[0], self.y_bounds[1])
            t = np.random.randint(0, len(self.implant_parameters))
            
            # Zufällige Geschwindigkeit
            vx = np.random.uniform(-0.5, 0.5)
            vy = np.random.uniform(-0.5, 0.5)
            vt = np.random.uniform(-0.5, 0.5)
            
            # Bewerte initiale Position
            f_val = self.total_function(x, y, t, self.implant_parameters)
            constraint_val = self.constraint_function(x, y, t, self.implant_parameters, self.z_threshold)
            is_feasible = constraint_val >= 0
            
            # Initialisiere persönliches Optimum
            p_best_pos = (x, y, t)
            p_best_val = f_val if is_feasible else float('-inf')
            
            particles.append({
                "position": (x, y, t),
                "velocity": (vx, vy, vt),
                "f_val": f_val,
                "constraint_val": constraint_val,
                "is_feasible": is_feasible,
                "p_best_pos": p_best_pos,
                "p_best_val": p_best_val
            })
        
        # Initialisiere globales Optimum
        g_best_pos = None
        g_best_val = float('-inf')
        
        # Finde das erste globale Optimum unter den initialen Partikeln
        for p in particles:
            if p["is_feasible"] and p["f_val"] > g_best_val:
                g_best_val = p["f_val"]
                g_best_pos = p["position"]
        
        # Liste der Konvergenzpunkte für die Visualisierung
        convergence_path = []
        if g_best_pos:
            convergence_path.append(g_best_pos)
        
        # Haupt-PSO-Schleife
        for iteration in range(max_iterations):
            for i, p in enumerate(particles):
                # Extrahiere aktuelle Position, Geschwindigkeit und persönliches Optimum
                x, y, t = p["position"]
                vx, vy, vt = p["velocity"]
                p_best_x, p_best_y, p_best_t = p["p_best_pos"]
                
                # Zufällige Faktoren
                r1, r2 = np.random.random(2)
                
                # Berechne neue Geschwindigkeit
                if g_best_pos:
                    g_best_x, g_best_y, g_best_t = g_best_pos
                    
                    # Standard-PSO-Geschwindigkeitsaktualisierung
                    vx_new = w * vx + c1 * r1 * (p_best_x - x) + c2 * r2 * (g_best_x - x)
                    vy_new = w * vy + c1 * r1 * (p_best_y - y) + c2 * r2 * (g_best_y - y)
                    vt_new = w * vt + c1 * r1 * (p_best_t - t) + c2 * r2 * (g_best_t - t)
                else:
                    # Wenn noch kein globales Optimum gefunden wurde, verwende nur persönliches Optimum
                    vx_new = w * vx + c1 * r1 * (p_best_x - x)
                    vy_new = w * vy + c1 * r1 * (p_best_y - y)
                    vt_new = w * vt + c1 * r1 * (p_best_t - t)
                
                # Berechne neue Position
                x_new = x + vx_new
                y_new = y + vy_new
                t_new_float = t + vt_new
                
                # Auf Grenzen beschränken
                x_new = np.clip(x_new, self.x_bounds[0], self.x_bounds[1])
                y_new = np.clip(y_new, self.y_bounds[0], self.y_bounds[1])
                
                # Runde t auf Ganzzahl (verschiedene Rundungsstrategien möglich)
                # 1. Deterministische Rundung:
                t_new = int(round(t_new_float))
                
                # Beschränke t auf gültige Werte
                t_new = max(0, min(t_new, len(self.implant_parameters) - 1))
                
                # Bewerte neue Position
                f_val = self.total_function(x_new, y_new, t_new, self.implant_parameters)
                constraint_val = self.constraint_function(x_new, y_new, t_new, self.implant_parameters, self.z_threshold)
                is_feasible = constraint_val >= 0
                
                # Aktualisiere Partikel
                particles[i]["position"] = (x_new, y_new, t_new)
                particles[i]["velocity"] = (vx_new, vy_new, vt_new)
                particles[i]["f_val"] = f_val
                particles[i]["constraint_val"] = constraint_val
                particles[i]["is_feasible"] = is_feasible
                
                # Aktualisiere persönliches Optimum, wenn die neue Position besser und zulässig ist
                if is_feasible and f_val > p["p_best_val"]:
                    particles[i]["p_best_pos"] = (x_new, y_new, t_new)
                    particles[i]["p_best_val"] = f_val
                    
                    # Aktualisiere globales Optimum, wenn die neue Position besser ist
                    if f_val > g_best_val:
                        g_best_val = f_val
                        g_best_pos = (x_new, y_new, t_new)
                        
                        # Zum Konvergenzpfad hinzufügen
                        convergence_path.append(g_best_pos)
            
            # Ausgabe des aktuellen Status
            if iteration % 5 == 0 or iteration == max_iterations - 1:
                if g_best_pos:
                    g_best_x, g_best_y, g_best_t = g_best_pos
                    implant_name = self.implant_parameters[g_best_t]["name"]
                    print(f"Iteration {iteration+1}/{max_iterations}: Bestes Optimum bei (x={g_best_x:.4f}, y={g_best_y:.4f}, t={g_best_t} ({implant_name})) mit f={g_best_val:.6f}")
                else:
                    print(f"Iteration {iteration+1}/{max_iterations}: Noch keine zulässige Lösung gefunden")
            
            # Detaillierte Fortschrittsanzeige
            if verbose and (iteration % 5 == 0 or iteration == max_iterations - 1):
                elapsed = time.time() - start_time
                if g_best_pos:
                    g_best_x, g_best_y, g_best_t = g_best_pos
                    implant_name = self.implant_parameters[g_best_t]["name"]
                    progress = (iteration + 1) / max_iterations * 100
                    print(f"[{progress:3.0f}%] Iteration {iteration+1}/{max_iterations} "
                        f"({elapsed:.6f}s): Beste Position (x={g_best_x:.4f}, y={g_best_y:.4f}, "
                        f"t={g_best_t} ({implant_name})), f={g_best_val:.6f}")
                else:
                    progress = (iteration + 1) / max_iterations * 100
                    print(f"[{progress:3.0f}%] Iteration {iteration+1}/{max_iterations} "
                        f"({elapsed:.1f}s): Noch keine zulässige Lösung gefunden")
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Ausgabe der endgültigen Ergebnisse
        if g_best_pos:
            x_opt, y_opt, t_opt = g_best_pos
            
            # Erstelle Ergebnisobjekt
            result = self.format_result(
                x_opt=x_opt,
                y_opt=y_opt,
                t_opt=t_opt,
                f_opt=g_best_val,
                lambda_opt=0.0,  # PSO verwendet keinen Lagrange-Multiplikator
                path=convergence_path,
                constraint_value=self.constraint_function(x_opt, y_opt, t_opt, self.implant_parameters, self.z_threshold),
                computation_time=computation_time,
                termination_reason="Maximale Iterationen erreicht",
                outer_iterations=max_iterations,
                inner_iterations=0
            )
            
            self.print_colored("\n=== ERGEBNIS DER PARTICLE SWARM OPTIMIZATION ===", "green")
            self.print_result(result)
            
            return result
        else:
            print("\nKeine zulässige Lösung gefunden!")
            return None
'''