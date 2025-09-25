"""Hauptprogramm für die Optimierung mit ganzzahliger Variable t."""
import argparse
import numpy as np
import time
import sys
import os

# Importiere Konfigurationsmodule
from optimization.config.parameters import get_all_implant_params
from optimization.config.functions import create_total_function, create_total_function_with_t, get_all_function_info
from optimization.config.constraints import get_constraint, create_horizontal_constraint_with_t, get_all_constraints

# Importiere Solver
from optimization.solvers.augmented_lagrangian import AugmentedLagrangianSolver
from optimization.solvers.particle_swarm import ParticleSwarmSolver

# Importiere Visualisierung
from optimization.utils.visualization import visualize_results

# Für farbige Terminalausgabe
def print_colored(text, color=None):
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
    
    if color in colors:
        print(f"{colors[color]}{text}{colors['end']}")
    else:
        print(text)

def print_optimization_summary(result, implant_parameters, total_time, z_threshold):
    """Zeigt eine übersichtliche Zusammenfassung der Optimierungsergebnisse an."""
    
    print("\n" + "="*80)
    print_colored(" OPTIMIERUNGSZUSAMMENFASSUNG ".center(80, "="), "bold")
    print("="*80)
    
    print_colored("\nOPTIMALER PUNKT:", "cyan")
    print(f"  x = {result['x_opt']:.6f}")
    print(f"  y = {result['y_opt']:.6f}")
    print(f"  t = {result['t_opt']} ({implant_parameters[result['t_opt']]['name']})")
    print(f"  f(x,y,t) = {result['f_opt']:.6f}")
    
    print_colored("\nIMPLANTATPARAMETER:", "cyan")
    for key, value in result['implant_params_opt'].items():
        if key != 'name':
            print(f"  {key} = {value:.6f}")
    
    print_colored("\nPERFORMANCE-METRIKEN:", "cyan")
    print(f"  Gesamtlaufzeit: {total_time:.6f} Sekunden")
    print(f"  Berechnungszeit: {result.get('computation_time', 'N/A'):.6f} Sekunden")
    
    print_colored("\nITERATIONEN:", "cyan")
    print(f"  Äußere Iterationen: {result.get('outer_iterations', 'N/A')}")
    print(f"  Innere Iterationen: {result.get('inner_iterations', 'N/A')}")
    print(f"  Gesamtschritte: {len(result.get('path', []))}")
    
    print_colored(f"\nKONVERGENZ:", "cyan")
    print(f"  Abbruchkriterium: {result.get('termination_reason', 'Unbekannt')}")
    constraint_value = result.get('constraint_value', 0)
    print(f"  Nebenbedingungswert: f(x,y,t) - {z_threshold} = {constraint_value:.8f}")
    
    if constraint_value >= -1e-6:
        print_colored("  ✓ Nebenbedingung erfüllt", "green")
    else:
        print_colored("  ✗ Nebenbedingung nicht erfüllt", "red")
    
    print("\n" + "="*80)

def generate_formula_string(weights, function_infos):
    """Generiert eine nutzerfreundliche Darstellung der Formel."""
    terms = []
    
    for i, w in weights.items():
        if w == 0:
            continue
        formula = function_infos[i].get("formula", f"f{i+1}")
        term = f"{w:.1f} · [{formula}]"
        terms.append(term)
    
    if not terms:
        return "f(x,y,t) = 0"
    
    return "f(x,y,t) = " + " + ".join(terms)

def print_optimization_setup(weights, function_infos, constraints, solver_name, strategy_name, params):
    """Zeigt die Optimierungsparameter nutzerfreundlich an."""
    print("\n" + "="*60)
    print(" OPTIMIERUNGSAUFBAU ".center(60, "="))
    print("="*60)
    
    # Zielfunktion anzeigen
    formula = generate_formula_string(weights, function_infos)
    print(f"\nZIELFUNKTION:")
    print(f"  {formula}")
    
    # Gewichtungen anzeigen
    print("\nGEWICHTUNGEN:")
    for i, w in weights.items():
        if w > 0:
            print(f"  {function_infos[i]['name']}: {w:.1f}")
    
    # Nebenbedingungen anzeigen
    print("\nNEBENBEDINGUNGEN:")
    for name, value in constraints.items():
        if name == "horizontal" and value is not None:
            print(f"  Horizontale Ebene: f(x,y,t) ≥ {value}")
        elif name == "elliptical" and value is not None:
            print(f"  Elliptische Domäne: x²/{value['a']} + y²/{value['b']} ≤ 1")
    
    # Solver und Parameter anzeigen
    print("\nOPTIMIERUNGSSOLVER:")
    if solver_name == "augmented_lagrangian":
        print(f"  Augmented Lagrangian")
        print(f"  Startwert-Strategie: {strategy_name}")
        if strategy_name in ["random", "combined"]:
            print(f"  Anzahl Startwerte: {params.get('num_points', 10)}")
    elif solver_name == "pso":
        print(f"  Particle Swarm Optimization:")
        print(f"    Anzahl Partikel: {params.get('n_particles', 20)}")
        print(f"    Anzahl Iterationen: {params.get('max_iterations', 50)}")
    
    # Optimierungsbereich anzeigen
    print("\nOPTIMIERUNGSBEREICH:")
    print(f"  x ∈ [{params.get('x_bounds', (-3, 3))[0]}, {params.get('x_bounds', (-3, 3))[1]}]")
    print(f"  y ∈ [{params.get('y_bounds', (-3, 3))[0]}, {params.get('y_bounds', (-3, 3))[1]}]")
    
    print("\n" + "="*60)

def parse_args():
    """Parst Kommandozeilenargumente."""
    parser = argparse.ArgumentParser(description='Optimierung mit ganzzahliger Implantattyp-Variable.')
    
    # Basale Parameter
    parser.add_argument('--interactive', action='store_true', help='Ausführung im interaktiven Modus')
    
    # Solverauswahl
    parser.add_argument('--solver', type=str, choices=['augmented_lagrangian', 'pso', 'pso_gradient', 'gekko', 'scipy'], default='pso',
                      help='Zu verwendender Optimierungssolver')
    
    # Parameter für Augmented Lagrangian
    parser.add_argument('--strategy', type=str, choices=['fixed', 'random', 'combined'], default='combined',
                      help='Strategie für Startwerte (nur für Augmented Lagrangian)')
    parser.add_argument('--num_points', type=int, default=150, 
                      help='Anzahl der Startwerte für random/combined Strategie')
    
    # PSO-spezifische Parameter
    parser.add_argument('--particles', type=int, default=20, help='Anzahl der Partikel für PSO')
    parser.add_argument('--iterations', type=int, default=50, help='Maximale Iterationen für PSO')

    # PSO-Vergleichsoptionen
    parser.add_argument('--compare_pso', action='store_true', 
                  help='Vergleiche Standard PSO mit Gradient-Enhanced PSO')

    # PSO-Gradient spezifische Parameter    
    parser.add_argument('--gradient_weight', type=float, default=0.3, 
                    help='Gewichtung des Gradiententerms (c3)')
    parser.add_argument('--gradient_prob', type=float, default=0.5, 
                    help='Wahrscheinlichkeit der Gradientenverwendung')
    
    # NEUE ARGUMENTE HIER HINZUFÜGEN:
    # ===================================
    parser.add_argument('--comprehensive_gradient_study', action='store_true', 
                      help='Führe umfassende Gradientenparameter-Studie durch')
    parser.add_argument('--gradient_study_runs', type=int, default=3,
                      help='Anzahl Läufe pro Konfiguration in der Studie')

    # Zusätzliche Parameter für Gekko und Scipy
    parser.add_argument('--integer_constraints', action='store_true', 
                      help='Verwende ganzzahlige Koordinaten (für GEKKO und SciPy)')
    
    # Funktionsgewichte
    parser.add_argument('--omega1', type=float, default=1.0, help='Gewicht für Funktion 1')
    parser.add_argument('--omega2', type=float, default=1.0, help='Gewicht für Funktion 2')
    parser.add_argument('--omega3', type=float, default=1.0, help='Gewicht für Funktion 3')
    parser.add_argument('--omega4', type=float, default=0.0, help='Gewicht für Funktion 4')
    
    # Nebenbedingungsparameter
    parser.add_argument('--threshold', type=float, default=0.3, help='Schwellenwert für horizontale Nebenbedingung')
    
    # Grenzen
    parser.add_argument('--x_min', type=float, default=-3.0, help='Minimaler x-Wert')
    parser.add_argument('--x_max', type=float, default=3.0, help='Maximaler x-Wert')
    parser.add_argument('--y_min', type=float, default=-3.0, help='Minimaler y-Wert')
    parser.add_argument('--y_max', type=float, default=3.0, help='Maximaler y-Wert')
    
    # Visualisierungsoptionen
    parser.add_argument('--no_vis', action='store_true', help='Visualisierung deaktivieren')

    # Ausführlicher Modus zur Steuerung der Ausgabemenge
    parser.add_argument('--verbose', '-v', action='count', default=0, 
                        help='Erhöht Ausführlichkeit (verwenden Sie -v, -vv oder -vvv für mehr Details)')
    
    return parser.parse_args()

def interactive_function_selection(all_functions):
    """Interaktive Auswahl der Teilfunktionen und Gewichtungen."""
    print("\n===== VERFÜGBARE TEILFUNKTIONEN =====")
    
    for i, func_info in enumerate(all_functions):
        print(f"{i+1}. {func_info['name']}(x,y) = {func_info['formula']} [{func_info['description']}]")
    
    selected_funcs = input("\nWählen Sie die Teilfunktionen (z.B. '1,2,3' für f₁, f₂ und f₃): ")
    func_indices = [int(f.strip())-1 for f in selected_funcs.split(',') if f.strip().isdigit() 
                   and 1 <= int(f.strip()) <= len(all_functions)]
    
    if not func_indices:
        print("Keine gültigen Funktionen ausgewählt. Verwende alle Funktionen.")
        func_indices = list(range(len(all_functions)))
    
    weights = {}
    for idx in func_indices:
        weight = input(f"Gewichtung für {all_functions[idx]['name']} (Standard=1.0): ")
        try:
            weights[idx] = float(weight) if weight.strip() else 1.0
        except ValueError:
            weights[idx] = 1.0
            print(f"Ungültige Eingabe. Verwende Standardgewichtung 1.0 für {all_functions[idx]['name']}")
    
    return weights

def interactive_constraint_selection(all_constraints):
    """Interaktive Auswahl der Nebenbedingungen."""
    print("\n===== VERFÜGBARE NEBENBEDINGUNGEN =====")
    
    for i, constraint_info in enumerate(all_constraints):
        print(f"{i+1}. {constraint_info['name']}: {constraint_info['formula']}")
    
    selected_constraints = input("\nWählen Sie die Nebenbedingungen (z.B. '1,2'): ")
    constraint_indices = [int(c.strip())-1 for c in selected_constraints.split(',') 
                         if c.strip().isdigit() and 1 <= int(c.strip()) <= len(all_constraints)]
    
    if not constraint_indices:
        print("Keine gültigen Nebenbedingungen ausgewählt. Verwende horizontale Ebene.")
        # Default: erste Nebenbedingung (horizontale Ebene)
        constraint_indices = [0]
    
    constraints = {}
    for idx in constraint_indices:
        constraint_type = all_constraints[idx]['id']
        
        if constraint_type == "horizontal":
            z = input("Schwellenwert z für Horizontale Ebene (Standard=0.3): ")
            try:
                constraints[constraint_type] = float(z) if z.strip() else 0.3
            except ValueError:
                constraints[constraint_type] = 0.3
                print("Ungültige Eingabe. Verwende Standardwert 0.3 für horizontale Ebene.")
        
        elif constraint_type == "elliptical":
            try:
                a = input("Parameter a für elliptische Domäne (Standard=4.0): ")
                b = input("Parameter b für elliptische Domäne (Standard=9.0): ")
                constraints[constraint_type] = {
                    "a": float(a) if a.strip() else 4.0,
                    "b": float(b) if b.strip() else 9.0
                }
            except ValueError:
                constraints[constraint_type] = {"a": 4.0, "b": 9.0}
                print("Ungültige Eingabe. Verwende Standardwerte a=4.0, b=9.0 für elliptische Domäne.")
    
    return constraints

def interactive_solver_selection():
    """Interaktive Auswahl des Optimierungssolvers und der zugehörigen Parameter."""
    print("\n===== VERFÜGBARE OPTIMIERUNGSSOLVER =====")
    print("1. Augmented Lagrangian mit Startpunktstrategie")
    print("2. Particle Swarm Optimization (PSO)")
    print("3. GEKKO Mixed Integer Non-Linear Programming")
    print("4. SciPy SLSQP mit Enumeration")
    
    solver_choice = input("\nWählen Sie den Optimierungssolver (1-4): ")
    
    solver_params = {}
    
    if solver_choice == "1":
        solver = "augmented_lagrangian"
        
        # Auswahl der Startpunktstrategie
        print("\n- STARTPUNKT-STRATEGIEN FÜR AUGMENTED LAGRANGIAN -")
        print("1. Feste Startpunkte (systematisch)")
        print("2. Zufällige Startpunkte")
        print("3. Kombinierte Strategie (fest + zufällig)")
        
        strategy_choice = input("\nWählen Sie die Strategie für Startpunkte (1-3): ")
        
        if strategy_choice == "1":
            strategy = "fixed"
        elif strategy_choice == "2":
            strategy = "random"
            num_points = input("Anzahl zufälliger Startpunkte (Standard=10): ")
            try:
                solver_params["num_points"] = int(num_points) if num_points.strip() else 10
            except ValueError:
                solver_params["num_points"] = 10
                print("Ungültige Eingabe. Verwende 10 zufällige Startpunkte.")
        else:
            # Default oder explizite Wahl von kombinierten Startpunkten
            strategy = "combined"
            num_points = input("Gesamtzahl der Startpunkte (Standard=15): ")
            try:
                solver_params["num_points"] = int(num_points) if num_points.strip() else 15
            except ValueError:
                solver_params["num_points"] = 15
                print("Ungültige Eingabe. Verwende 15 Startpunkte (fest + zufällig).")
                
        solver_params["strategy"] = strategy
        
    elif solver_choice == "2":
        # Default oder explizite Wahl von PSO
        solver = "pso"
        particles = input("Anzahl der Partikel (Standard=20): ")
        iterations = input("Maximale Anzahl Iterationen (Standard=50): ")
        
        try:
            solver_params["n_particles"] = int(particles) if particles.strip() else 20
        except ValueError:
            solver_params["n_particles"] = 20
            print("Ungültige Eingabe. Verwende 20 Partikel.")
            
        try:
            solver_params["max_iterations"] = int(iterations) if iterations.strip() else 50
        except ValueError:
            solver_params["max_iterations"] = 50
            print("Ungültige Eingabe. Verwende 50 Iterationen.")

    elif solver_choice == "3":
        solver = "gekko"
        int_constr = input("Ganzzahlige Koordinaten verwenden? (j/n, Standard: n): ")
        solver_params["integer_constraints"] = int_constr.lower() in ('j', 'ja', 'y', 'yes')
        
    elif solver_choice == "4":
        solver = "scipy"
        int_constr = input("Ganzzahlige Koordinaten verwenden? (j/n, Standard: n): ")
        solver_params["integer_constraints"] = int_constr.lower() in ('j', 'ja', 'y', 'yes')

    else:
        # Default
        solver = "pso"
    
    # Optimierungsbereich
    print("\n===== OPTIMIERUNGSBEREICH =====")
    try:
        x_min = input("Minimaler x-Wert (Standard=-3.0): ")
        x_max = input("Maximaler x-Wert (Standard=3.0): ")
        y_min = input("Minimaler y-Wert (Standard=-3.0): ")
        y_max = input("Maximaler y-Wert (Standard=3.0): ")
        
        x_bounds = (float(x_min) if x_min.strip() else -3.0, float(x_max) if x_max.strip() else 3.0)
        y_bounds = (float(y_min) if y_min.strip() else -3.0, float(y_max) if y_max.strip() else 3.0)
        
        solver_params["x_bounds"] = x_bounds
        solver_params["y_bounds"] = y_bounds
    except ValueError:
        solver_params["x_bounds"] = (-3.0, 3.0)
        solver_params["y_bounds"] = (-3.0, 3.0)
        print("Ungültige Eingabe. Verwende Standardbereich: x,y ∈ [-3.0, 3.0]")
    
    return solver, solver_params

def interactive_mode():
    """Führt den interaktiven Modus aus."""
    print("="*60)
    print(" INTERAKTIVER MODUS DER OPTIMIERUNGSSOFTWARE ".center(60, "="))
    print("="*60)
    
    # Teilfunktionen definieren
    all_functions = [
        {"id": "f1", "name": "f₁", "formula": "x·exp(-(x²+y²)/a₁)", "description": "Gauß-förmige Funktion"},
        {"id": "f2", "name": "f₂", "formula": "(x²+y²)/a₄", "description": "Quadratische Funktion"},
        {"id": "f3", "name": "f₃", "formula": "a₃·sin(x·y·a₂)", "description": "Sinus-Oszillation"},
        {"id": "f4", "name": "f₄", "formula": "a₅·cos(x·y/a₆)", "description": "Kosinus-Oszillation"}
    ]
    
    # Nebenbedingungen definieren
    all_constraints = [
        {"id": "horizontal", "name": "Horizontale Ebene", "formula": "f(x,y,t) ≥ z"},
        {"id": "elliptical", "name": "Elliptische Domäne", "formula": "x²/a + y²/b ≤ 1"}
    ]
    
    # Interaktive Auswahl
    weights = interactive_function_selection(all_functions)
    constraints = interactive_constraint_selection(all_constraints)
    solver, solver_params = interactive_solver_selection()
    
    # Visualisierung aktivieren/deaktivieren
    vis_option = input("\nVisualisierung aktivieren? (j/n, Standard: j): ")
    solver_params["visualize"] = vis_option.lower() not in ('n', 'nein', 'no')
    
    # Ergebnisparameter zusammenstellen
    params = {
        'function_weights': weights,
        'constraints': constraints,
        'solver': solver,
        **solver_params
    }
    
    # Optimierungssetup anzeigen
    print_optimization_setup(
        weights, all_functions, constraints, 
        solver, solver_params.get("strategy", ""), solver_params
    )
    
    confirm = input("\nOptimierung mit diesen Einstellungen starten? (j/n): ")
    if confirm.lower() not in ('j', 'ja', 'y', 'yes'):
        print("Optimierung abgebrochen.")
        sys.exit(0)
    
    return params

def main():
    """Hauptfunktion zur Ausführung der Optimierung mit ganzzahliger Variable t."""
    # Parse Kommandozeilenargumente
    args = parse_args()
    
    # Teilfunktionen definieren
    all_functions = [
        {"id": "f1", "name": "f₁", "formula": "x·exp(-(x²+y²)/a₁)", "description": "Gauß-förmige Funktion"},
        {"id": "f2", "name": "f₂", "formula": "(x²+y²)/a₄", "description": "Quadratische Funktion"},
        {"id": "f3", "name": "f₃", "formula": "a₃·sin(x·y·a₂)", "description": "Sinus-Oszillation"},
        {"id": "f4", "name": "f₄", "formula": "a₅·cos(x·y/a₆)", "description": "Kosinus-Oszillation"}
    ]
    
    # Wenn interaktiver Modus, Benutzereingabe holen
    if args.interactive:
        params = interactive_mode()
    else:
        # Standard: Alle Funktionen mit Standardgewichten
        weights = {
            0: args.omega1, 
            1: args.omega2, 
            2: args.omega3,
            3: args.omega4
        }
        
        # Alle aktivierten Funktionen anzeigen
        active_functions = {i: w for i, w in weights.items() if w > 0}
        print(f"Verwende Funktionen: {', '.join([all_functions[i]['id'] for i in active_functions.keys()])}")
        print(f"Gewichtungen: {active_functions}")
        
        # Ansonsten Kommandozeilenargumente verwenden
        params = {
            'function_weights': weights,
            'constraints': {'horizontal': args.threshold},
            'solver': args.solver,
            'strategy': args.strategy,
            'num_points': args.num_points,
            'n_particles': args.particles,
            'max_iterations': args.iterations,
            'x_bounds': (args.x_min, args.x_max),
            'y_bounds': (args.y_min, args.y_max),
            'visualize': not args.no_vis,
            'verbose': args.verbose > 0
        }
    
    # Hol Implantatparameter
    implant_parameters = get_all_implant_params()
    
    # Erstelle Gewichtungen und definiere, welche Funktionen zu verwenden sind
    weights = params['function_weights']
    function_names = [f"f{i+1}" for i in weights.keys() if weights[i] > 0]  # Welche Funktionen überhaupt aktiv sind

    # DEBUG-AUSGABE HINZUFÜGEN
    print(f"DEBUG: Ausgewählte Gewichtungen: {weights}")
    print(f"DEBUG: Generierte Funktionsnamen: {function_names}")
    print(f"DEBUG: Aktive Funktionen: {', '.join(function_names)}")
    
    # Test der Gesamtfunktion
    total_function = create_total_function(function_names, weights)
    total_function_with_t = create_total_function_with_t(function_names, weights)
    
    # VERIFIKATION: Teste die Funktion bei (0,0) mit Typ A
    test_result = total_function_with_t(0.0, 0.0, 0, implant_parameters)
    print(f"DEBUG: Test der Gesamtfunktion bei (0,0) mit Typ A: {test_result:.6f}")
    
    # Erstelle Gesamtfunktion und Nebenbedingung
    total_function = create_total_function(function_names, weights)
    total_function_with_t = create_total_function_with_t(function_names, weights)
    
    
    # Hol Nebenbedingungen
    constraint_horizontal = get_constraint("horizontal")
    z_threshold = params['constraints'].get('horizontal', 0.3)
    constraint_with_t = create_horizontal_constraint_with_t(total_function_with_t)
    
    # Zeige Optimierungssetup
    if not args.interactive:
        print_optimization_setup(
            weights, all_functions, 
            {'horizontal': z_threshold}, 
            params['solver'], params['strategy'], 
            params
        )
    
    # NEUE AUSGABE: Vor dem Starten der Optimierung
    print("\n" + "="*60)
    print_colored(" STARTE OPTIMIERUNG ".center(60, "="), "bold")
    print("="*60)
    print(f"Solver: {params['solver'].upper()}")
    if params['solver'] == 'augmented_lagrangian':
        print(f"Strategie: {params['strategy'].upper()}")
    print(f"Optimierungsbereich: x ∈ {params['x_bounds']}, y ∈ {params['y_bounds']}")
    print(f"Nebenbedingung: f(x,y,t) ≥ {z_threshold}")
    
    # Messen der Laufzeit
    start_time = time.time()
    
    # Führe Optimierung mit gewähltem Solver aus
    if params['solver'] == 'augmented_lagrangian':
        print_colored("\nStarte Optimierung mit Augmented Lagrangian...", "purple")
        
        solver = AugmentedLagrangianSolver(
            total_function_with_t, constraint_with_t, 
            params['x_bounds'], params['y_bounds'], z_threshold,
            implant_parameters
        )
        
        result = solver.solve(
            strategy=params['strategy'],
            num_points=params['num_points'],
            verbose=params.get('verbose', True)
        )
        
    elif params['solver'] == 'pso':
        n_particles = params.get('n_particles', 20)
        max_iterations = params.get('max_iterations', 50)
        print_colored(f"\nStarte Particle Swarm Optimization mit {n_particles} Partikeln und {max_iterations} Iterationen...", "purple")
        
        solver = ParticleSwarmSolver(
            total_function_with_t, constraint_with_t, 
            params['x_bounds'], params['y_bounds'], z_threshold,
            implant_parameters
        )
        
        result = solver.solve(
            n_particles=n_particles,
            max_iterations=max_iterations,
            verbose=params.get('verbose', True)
        )

    elif params['solver'] == 'gekko':
        try:
            from optimization.solvers import GekkoSolver
            print_colored("\nStarte GEKKO MINLP Optimization...", "purple")
            
            solver = GekkoSolver(
                total_function_with_t, constraint_with_t, 
                params['x_bounds'], params['y_bounds'], z_threshold,
                implant_parameters
            )
            
            result = solver.solve(
                integer_constraints=params.get('integer_constraints', False),
                verbose=params.get('verbose', True)
            )
            
        except ImportError:
            print_colored("GEKKO ist nicht installiert. Installieren Sie es mit: pip install gekko", "red")
            return

    elif params['solver'] == 'scipy':
        try:
            from optimization.solvers import ScipySolver
            print_colored("\nStarte SciPy SLSQP Enumeration...", "purple")
            
            solver = ScipySolver(
                total_function_with_t, constraint_with_t, 
                params['x_bounds'], params['y_bounds'], z_threshold,
                implant_parameters
            )
            
            result = solver.solve(
                integer_constraints=params.get('integer_constraints', False),
                verbose=params.get('verbose', True)
            )
            
        except ImportError:
            print_colored("SciPy ist nicht installiert. Installieren Sie es mit: pip install scipy", "red")
            return
        
    elif params['solver'] == 'pso_gradient':
        try:
            from optimization.solvers.pso_gradient import GradientEnhancedPSOSolver
            print_colored(f"\nStarte Gradient-Enhanced PSO...", "purple")
            
            solver = GradientEnhancedPSOSolver(
                total_function_with_t, constraint_with_t, 
                params['x_bounds'], params['y_bounds'], z_threshold,
                implant_parameters
            )
            
            result = solver.solve(
                n_particles=params.get('n_particles', 20),
                max_iterations=params.get('max_iterations', 50),
                c3_gradient=params.get('gradient_weight', 0.3),
                gradient_probability=params.get('gradient_prob', 0.5),
                verbose=params.get('verbose', True)
            )
            
        except ImportError as e:
            print_colored(f"Gradient-Enhanced PSO nicht verfügbar: {e}", "red")
            return
        
    # Vergleichsfunktion
    if args.compare_pso and not args.interactive:
        from optimization.comparison.pso_comparison import compare_pso_algorithms
        
        print_colored("\nStarte PSO-Algorithmus-Vergleich...", "blue")
        comparison_results = compare_pso_algorithms(
            total_function_with_t, constraint_with_t, implant_parameters, z_threshold,
            x_bounds=params['x_bounds'], y_bounds=params['y_bounds'],
            n_particles=params.get('n_particles', 20),
            max_iterations=params.get('max_iterations', 50),
            num_runs=5
        )
        return
    
    # Umfassendere Gradientenparameter-Studie
    if args.comprehensive_gradient_study and not args.interactive:
        from optimization.comparison.gradient_parameter_comparison import comprehensive_gradient_pso_comparison
        
        print_colored("\nStarte umfassende Gradientenparameter-Studie...", "blue")
        print(f"Anzahl Läufe pro Konfiguration: {args.gradient_study_runs}")
        
        study_results = comprehensive_gradient_pso_comparison(
            total_function_with_t, constraint_with_t, implant_parameters, z_threshold,
            x_bounds=params['x_bounds'], y_bounds=params['y_bounds'],
            n_particles=params.get('n_particles', 20),
            max_iterations=params.get('max_iterations', 50),
            num_runs=args.gradient_study_runs
        )
        
        print_colored("\nGradientenparameter-Studie abgeschlossen!", "green")
        return
        
    # Messung der Gesamtlaufzeit beenden
    end_time = time.time()
    total_duration = end_time - start_time
    
    # NEUE AUSGABE: Nach der Optimierung
    if result:
        print_colored("\nOptimierung erfolgreich abgeschlossen!", "green")
        print_optimization_summary(result, implant_parameters, total_duration, z_threshold)
    else:
        print_colored("\nOptimierung konnte keine zulässige Lösung finden.", "red")
        print(f"Gesamtlaufzeit: {total_duration:.6f} Sekunden")
    
    # Visualisiere Ergebnisse falls gewünscht
    if result and params.get('visualize', True):
        print_colored("\nErstelle Visualisierung...", "blue")
        
        # t_opt wird für die Visualisierung benötigt
        t_opt = result["t_opt"]
        
        visualize_results(
            result, total_function, constraint_horizontal, 
            implant_parameters, z_threshold,
            x_range=params['x_bounds'], y_range=params['y_bounds']
        )
    elif not result:
        print_colored("\nKeine Visualisierung möglich, da keine zulässige Lösung gefunden wurde.", "yellow")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\nOptimierung durch Benutzer abgebrochen.", "yellow")
        sys.exit(0)
    except Exception as e:
        print_colored(f"Fehler: {str(e)}", "red")
        import traceback
        traceback.print_exc()
        sys.exit(1)