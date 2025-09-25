"""Framework zum Vergleich von Standard PSO und Gradient-Enhanced PSO."""
import time
import numpy as np
import matplotlib.pyplot as plt

def _analyze_convergence_speed(standard_results, gradient_results):
    """Analysiert Konvergenzgeschwindigkeit - einfach zu bestehender Analyse hinzufügen."""
    
    print(f"\n=== KONVERGENZGESCHWINDIGKEIT-ANALYSE ===")
    
    for run in range(len(standard_results)):
        if run < len(gradient_results):
            standard = standard_results[run]
            gradient = gradient_results[run]
            
            print(f"\nLauf {run + 1}:")
            print(f"  Standard PSO: f={standard['f_opt']:.4f}, Zeit: {standard['time']:.3f}s")
            print(f"  Gradient PSO: f={gradient['f_opt']:.4f}, Zeit: {gradient['time']:.3f}s")
            
            # Berechne wie viel schneller Gradient PSO zu einem besseren Ergebnis kam
            improvement = ((gradient['f_opt'] - standard['f_opt']) / standard['f_opt'] * 100)
            overhead = ((gradient['time'] - standard['time']) / standard['time'] * 100)
            efficiency = improvement / max(overhead, 1)
            
            print(f"  Verbesserung: {improvement:+.2f}%, Overhead: {overhead:+.1f}%, Effizienz: {efficiency:.3f}")
            
            # Schätze wann Gradient PSO Standard PSO übertroffen hätte
            # Annahme: linearer Fortschritt (grober Schätzwert)
            gradient_path_length = gradient.get('iterations_to_best', 50)
            estimated_iteration_to_beat = gradient_path_length * 0.3  # Schätzung: 30% der Konvergenzzeit
            estimated_time_to_beat = gradient['time'] * (estimated_iteration_to_beat / 50)
            
            if improvement > 0:
                print(f"  Geschätzte Zeit bis Übertreffung: {estimated_time_to_beat:.3f}s")
                print(f"  Das sind {(estimated_time_to_beat/standard['time']*100):.1f}% der Standard PSO Gesamtzeit")

def compare_pso_algorithms(total_function_with_t, constraint_with_t, implant_parameters, z_threshold,
                         x_bounds=(-3, 3), y_bounds=(-3, 3), 
                         n_particles=20, max_iterations=50, num_runs=5):
    """
    Vergleicht Standard PSO mit Gradient-Enhanced PSO.
    
    Args:
        total_function_with_t: Zielfunktion
        constraint_with_t: Nebenbedingung
        implant_parameters: Implantatparameter
        z_threshold: Nebenbedingungsschwelle
        x_bounds, y_bounds: Optimierungsgrenzen
        n_particles: Anzahl Partikel
        max_iterations: Max. Iterationen
        num_runs: Anzahl Vergleichsläufe
        
    Returns:
        dict: Vergleichsergebnisse
    """
    from ..solvers.particle_swarm import ParticleSwarmSolver
    from ..solvers.pso_gradient import GradientEnhancedPSOSolver
    
    print(f"=== PSO ALGORITHMUS-VERGLEICH ===")
    print(f"Läufe: {num_runs}, Partikel: {n_particles}, Iterationen: {max_iterations}")
    
    # Ergebnisspeicher
    standard_results = []
    gradient_results = []
    
    # Mehrere Läufe für statistische Aussagekraft
    for run in range(num_runs):
        print(f"\nLauf {run + 1}/{num_runs}")
        
        # Standard PSO
        print("  Standard PSO...")
        standard_solver = ParticleSwarmSolver(
            total_function_with_t, constraint_with_t, 
            x_bounds, y_bounds, z_threshold, implant_parameters
        )
        
        start_time = time.time()
        standard_result = standard_solver.solve(
            n_particles=n_particles, max_iterations=max_iterations, verbose=False
        )
        standard_time = time.time() - start_time
        
        if standard_result:
            standard_results.append({
                'f_opt': standard_result['f_opt'],
                'time': standard_time,
                'iterations_to_best': len(standard_result['path']),
                'x_opt': standard_result['x_opt'],
                'y_opt': standard_result['y_opt'],
                't_opt': standard_result['t_opt']
            })
            print(f"    Ergebnis: f={standard_result['f_opt']:.4f}, Zeit: {standard_time:.3f}s")
        
        # Gradient-Enhanced PSO
        print("  Gradient-Enhanced PSO...")
        gradient_solver = GradientEnhancedPSOSolver(
            total_function_with_t, constraint_with_t, 
            x_bounds, y_bounds, z_threshold, implant_parameters
        )
        
        start_time = time.time()
        gradient_result = gradient_solver.solve(
            n_particles=n_particles, max_iterations=max_iterations, 
            c3_gradient=0.3, gradient_probability=0.5, verbose=False
        )
        gradient_time = time.time() - start_time
        
        if gradient_result:
            gradient_results.append({
                'f_opt': gradient_result['f_opt'],
                'time': gradient_time,
                'iterations_to_best': len(gradient_result['path']),
                'x_opt': gradient_result['x_opt'],
                'y_opt': gradient_result['y_opt'],
                't_opt': gradient_result['t_opt'],
                'gradient_stats': gradient_result.get('gradient_statistics', {})
            })
            print(f"    Ergebnis: f={gradient_result['f_opt']:.4f}, Zeit: {gradient_time:.3f}s")
            print(f"    Gradient-Nutzung: {gradient_result.get('gradient_statistics', {}).get('gradient_usage_percent', 0):.1f}%")
    
    # Statistische Auswertung
    comparison_results = _analyze_results(standard_results, gradient_results)
    
    _analyze_convergence_speed(standard_results, gradient_results)
    
    # Visualisierung
    _plot_comparison(comparison_results, num_runs)
    # Konvergenzplot
    _plot_convergence_speed(standard_results, gradient_results)
    
    return comparison_results

def _analyze_results(standard_results, gradient_results):
    """Analysiert die Vergleichsergebnisse statistisch."""
    
    if not standard_results or not gradient_results:
        print("Nicht genügend Ergebnisse für Vergleich!")
        return None
    
    # Standard PSO Statistiken
    standard_f_values = [r['f_opt'] for r in standard_results]
    standard_times = [r['time'] for r in standard_results]
    standard_convergence = [r['iterations_to_best'] for r in standard_results]
    
    # Gradient PSO Statistiken
    gradient_f_values = [r['f_opt'] for r in gradient_results]
    gradient_times = [r['time'] for r in gradient_results]
    gradient_convergence = [r['iterations_to_best'] for r in gradient_results]
    
    # Gradientenstatistiken
    gradient_usage = [r['gradient_stats'].get('gradient_usage_percent', 0) 
                     for r in gradient_results if 'gradient_stats' in r]
    gradient_success_rate = [r['gradient_stats'].get('gradient_success_rate', 0) 
                           for r in gradient_results if 'gradient_stats' in r]
    
    results = {
        'standard_pso': {
            'f_opt_mean': np.mean(standard_f_values),
            'f_opt_std': np.std(standard_f_values),
            'f_opt_best': np.max(standard_f_values),
            'time_mean': np.mean(standard_times),
            'time_std': np.std(standard_times),
            'convergence_mean': np.mean(standard_convergence),
            'convergence_std': np.std(standard_convergence)
        },
        'gradient_pso': {
            'f_opt_mean': np.mean(gradient_f_values),
            'f_opt_std': np.std(gradient_f_values),
            'f_opt_best': np.max(gradient_f_values),
            'time_mean': np.mean(gradient_times),
            'time_std': np.std(gradient_times),
            'convergence_mean': np.mean(gradient_convergence),
            'convergence_std': np.std(gradient_convergence),
            'gradient_usage_mean': np.mean(gradient_usage) if gradient_usage else 0,
            'gradient_success_mean': np.mean(gradient_success_rate) if gradient_success_rate else 0
        }
    }
    
    # Vergleichsstatistiken
    f_improvement = ((results['gradient_pso']['f_opt_mean'] - results['standard_pso']['f_opt_mean']) 
                    / results['standard_pso']['f_opt_mean'] * 100)
    time_overhead = ((results['gradient_pso']['time_mean'] - results['standard_pso']['time_mean']) 
                    / results['standard_pso']['time_mean'] * 100)
    
    results['comparison'] = {
        'f_improvement_percent': f_improvement,
        'time_overhead_percent': time_overhead,
        'better_results_count': sum(1 for g, s in zip(gradient_f_values, standard_f_values) if g > s),
        'total_runs': len(standard_results)
    }
    
    # Ausgabe der Ergebnisse
    print(f"\n=== VERGLEICHSERGEBNISSE ===")
    print(f"Standard PSO:")
    print(f"  Bester Wert: {results['standard_pso']['f_opt_best']:.6f}")
    print(f"  Mittelwert: {results['standard_pso']['f_opt_mean']:.6f} ± {results['standard_pso']['f_opt_std']:.6f}")
    print(f"  Zeit: {results['standard_pso']['time_mean']:.3f} ± {results['standard_pso']['time_std']:.3f} s")
    
    print(f"\nGradient-Enhanced PSO:")
    print(f"  Bester Wert: {results['gradient_pso']['f_opt_best']:.6f}")
    print(f"  Mittelwert: {results['gradient_pso']['f_opt_mean']:.6f} ± {results['gradient_pso']['f_opt_std']:.6f}")
    print(f"  Zeit: {results['gradient_pso']['time_mean']:.3f} ± {results['gradient_pso']['time_std']:.3f} s")
    print(f"  Gradient-Nutzung: {results['gradient_pso']['gradient_usage_mean']:.1f}%")
    print(f"  Gradient-Erfolgsrate: {results['gradient_pso']['gradient_success_mean']:.1f}%")
    
    print(f"\nVergleich:")
    print(f"  Verbesserung: {f_improvement:+.2f}%")
    print(f"  Zeit-Overhead: {time_overhead:+.2f}%")
    print(f"  Bessere Ergebnisse: {results['comparison']['better_results_count']}/{results['comparison']['total_runs']}")
    
    return results

def _plot_comparison(results, num_runs):
    """Erstellt Vergleichsplots."""
    if not results:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Zielfunktionswerte
    methods = ['Standard PSO', 'Gradient PSO']
    means = [results['standard_pso']['f_opt_mean'], results['gradient_pso']['f_opt_mean']]
    stds = [results['standard_pso']['f_opt_std'], results['gradient_pso']['f_opt_std']]
    
    ax1.bar(methods, means, yerr=stds, capsize=5, color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Zielfunktionswert')
    ax1.set_title('Vergleich der Zielfunktionswerte')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rechenzeiten
    time_means = [results['standard_pso']['time_mean'], results['gradient_pso']['time_mean']]
    time_stds = [results['standard_pso']['time_std'], results['gradient_pso']['time_std']]
    
    ax2.bar(methods, time_means, yerr=time_stds, capsize=5, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Rechenzeit (s)')
    ax2.set_title('Vergleich der Rechenzeiten')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Verbesserung vs. Overhead
    improvement = results['comparison']['f_improvement_percent']
    overhead = results['comparison']['time_overhead_percent']
    
    ax3.scatter([overhead], [improvement], s=100, c='green', marker='o')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Zeit-Overhead (%)')
    ax3.set_ylabel('Funktionswert-Verbesserung (%)')
    ax3.set_title('Effizienz-Analyse')
    ax3.grid(True, alpha=0.3)
    
    # Füge Annotation hinzu
    ax3.annotate(f'({overhead:.1f}%, {improvement:.1f}%)', 
                (overhead, improvement), xytext=(10, 10), 
                textcoords='offset points')
    
    # Plot 4: Erfolgsrate
    better_count = results['comparison']['better_results_count']
    total_count = results['comparison']['total_runs']
    success_rate = (better_count / total_count) * 100
    
    labels = ['Standard PSO besser', 'Gradient PSO besser']
    sizes = [total_count - better_count, better_count]
    colors = ['blue', 'red']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title(f'Erfolgsrate über {num_runs} Läufe')
    
    plt.tight_layout()
    plt.show()
    
    return fig




# Zusätzlicher Plot für Konvergenzgeschwindigkeit:
def _plot_convergence_speed(standard_results, gradient_results):
    """Zusätzlicher Plot für Konvergenzvergleich."""
    
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    runs = range(1, len(standard_results) + 1)
    standard_f = [r['f_opt'] for r in standard_results]
    gradient_f = [r['f_opt'] for r in gradient_results[:len(standard_results)]]
    standard_times = [r['time'] for r in standard_results]
    gradient_times = [r['time'] for r in gradient_results[:len(standard_results)]]
    
    # Plot 1: Funktionswerte pro Lauf
    ax1.plot(runs, standard_f, 'bo-', label='Standard PSO', linewidth=2, markersize=8)
    ax1.plot(runs, gradient_f, 'ro-', label='Gradient PSO', linewidth=2, markersize=8)
    ax1.set_xlabel('Lauf')
    ax1.set_ylabel('Zielfunktionswert')
    ax1.set_title('Funktionswerte pro Lauf')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Füge Verbesserungspfeile hinzu
    for i, (s_f, g_f) in enumerate(zip(standard_f, gradient_f)):
        if g_f > s_f:
            ax1.annotate('', xy=(i+1, g_f), xytext=(i+1, s_f),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Plot 2: Zeit vs. Qualität
    ax2.scatter(standard_times, standard_f, c='blue', s=100, alpha=0.7, label='Standard PSO')
    ax2.scatter(gradient_times, gradient_f, c='red', s=100, alpha=0.7, label='Gradient PSO')
    
    # Verbinde korrespondierende Punkte
    for s_time, s_f, g_time, g_f in zip(standard_times, standard_f, gradient_times, gradient_f):
        ax2.plot([s_time, g_time], [s_f, g_f], 'k--', alpha=0.3)
    
    ax2.set_xlabel('Rechenzeit (s)')
    ax2.set_ylabel('Zielfunktionswert')
    ax2.set_title('Zeit vs. Qualität')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig












'''
"""Framework zum Vergleich von Standard PSO und Gradient-Enhanced PSO."""
import time
import numpy as np
import matplotlib.pyplot as plt

def compare_pso_algorithms(total_function_with_t, constraint_with_t, implant_parameters, z_threshold,
                         x_bounds=(-3, 3), y_bounds=(-3, 3), 
                         n_particles=20, max_iterations=50, num_runs=5):
    """
    Vergleicht Standard PSO mit Gradient-Enhanced PSO.
    
    Args:
        total_function_with_t: Zielfunktion
        constraint_with_t: Nebenbedingung
        implant_parameters: Implantatparameter
        z_threshold: Nebenbedingungsschwelle
        x_bounds, y_bounds: Optimierungsgrenzen
        n_particles: Anzahl Partikel
        max_iterations: Max. Iterationen
        num_runs: Anzahl Vergleichsläufe
        
    Returns:
        dict: Vergleichsergebnisse
    """
    from ..solvers.particle_swarm import ParticleSwarmSolver
    from ..solvers.pso_gradient import GradientEnhancedPSOSolver
    
    print(f"=== PSO ALGORITHMUS-VERGLEICH ===")
    print(f"Läufe: {num_runs}, Partikel: {n_particles}, Iterationen: {max_iterations}")
    
    # Ergebnisspeicher
    standard_results = []
    gradient_results = []
    
    # Mehrere Läufe für statistische Aussagekraft
    for run in range(num_runs):
        print(f"\nLauf {run + 1}/{num_runs}")
        
        # Standard PSO
        print("  Standard PSO...")
        standard_solver = ParticleSwarmSolver(
            total_function_with_t, constraint_with_t, 
            x_bounds, y_bounds, z_threshold, implant_parameters
        )
        
        start_time = time.time()
        standard_result = standard_solver.solve(
            n_particles=n_particles, max_iterations=max_iterations, verbose=False
        )
        standard_time = time.time() - start_time
        
        if standard_result:
            standard_results.append({
                'f_opt': standard_result['f_opt'],
                'time': standard_time,
                'iterations_to_best': len(standard_result['path']),
                'x_opt': standard_result['x_opt'],
                'y_opt': standard_result['y_opt'],
                't_opt': standard_result['t_opt']
            })
            print(f"    Ergebnis: f={standard_result['f_opt']:.4f}, Zeit: {standard_time:.3f}s")
        
        # Gradient-Enhanced PSO
        print("  Gradient-Enhanced PSO...")
        gradient_solver = GradientEnhancedPSOSolver(
            total_function_with_t, constraint_with_t, 
            x_bounds, y_bounds, z_threshold, implant_parameters
        )
        
        start_time = time.time()
        gradient_result = gradient_solver.solve(
            n_particles=n_particles, max_iterations=max_iterations, 
            c3_gradient=0.3, gradient_probability=0.5, verbose=False
        )
        gradient_time = time.time() - start_time
        
        if gradient_result:
            gradient_results.append({
                'f_opt': gradient_result['f_opt'],
                'time': gradient_time,
                'iterations_to_best': len(gradient_result['path']),
                'x_opt': gradient_result['x_opt'],
                'y_opt': gradient_result['y_opt'],
                't_opt': gradient_result['t_opt'],
                'gradient_stats': gradient_result.get('gradient_statistics', {})
            })
            print(f"    Ergebnis: f={gradient_result['f_opt']:.4f}, Zeit: {gradient_time:.3f}s")
            print(f"    Gradient-Nutzung: {gradient_result.get('gradient_statistics', {}).get('gradient_usage_percent', 0):.1f}%")
    
    # Statistische Auswertung
    comparison_results = _analyze_results(standard_results, gradient_results)
    
    # Visualisierung
    _plot_comparison(comparison_results, num_runs)
    
    return comparison_results

def _analyze_results(standard_results, gradient_results):
    """Analysiert die Vergleichsergebnisse statistisch."""
    
    if not standard_results or not gradient_results:
        print("Nicht genügend Ergebnisse für Vergleich!")
        return None
    
    # Standard PSO Statistiken
    standard_f_values = [r['f_opt'] for r in standard_results]
    standard_times = [r['time'] for r in standard_results]
    standard_convergence = [r['iterations_to_best'] for r in standard_results]
    
    # Gradient PSO Statistiken
    gradient_f_values = [r['f_opt'] for r in gradient_results]
    gradient_times = [r['time'] for r in gradient_results]
    gradient_convergence = [r['iterations_to_best'] for r in gradient_results]
    
    # Gradientenstatistiken
    gradient_usage = [r['gradient_stats'].get('gradient_usage_percent', 0) 
                     for r in gradient_results if 'gradient_stats' in r]
    gradient_success_rate = [r['gradient_stats'].get('gradient_success_rate', 0) 
                           for r in gradient_results if 'gradient_stats' in r]
    
    results = {
        'standard_pso': {
            'f_opt_mean': np.mean(standard_f_values),
            'f_opt_std': np.std(standard_f_values),
            'f_opt_best': np.max(standard_f_values),
            'time_mean': np.mean(standard_times),
            'time_std': np.std(standard_times),
            'convergence_mean': np.mean(standard_convergence),
            'convergence_std': np.std(standard_convergence)
        },
        'gradient_pso': {
            'f_opt_mean': np.mean(gradient_f_values),
            'f_opt_std': np.std(gradient_f_values),
            'f_opt_best': np.max(gradient_f_values),
            'time_mean': np.mean(gradient_times),
            'time_std': np.std(gradient_times),
            'convergence_mean': np.mean(gradient_convergence),
            'convergence_std': np.std(gradient_convergence),
            'gradient_usage_mean': np.mean(gradient_usage) if gradient_usage else 0,
            'gradient_success_mean': np.mean(gradient_success_rate) if gradient_success_rate else 0
        }
    }
    
    # Vergleichsstatistiken
    f_improvement = ((results['gradient_pso']['f_opt_mean'] - results['standard_pso']['f_opt_mean']) 
                    / results['standard_pso']['f_opt_mean'] * 100)
    time_overhead = ((results['gradient_pso']['time_mean'] - results['standard_pso']['time_mean']) 
                    / results['standard_pso']['time_mean'] * 100)
    
    results['comparison'] = {
        'f_improvement_percent': f_improvement,
        'time_overhead_percent': time_overhead,
        'better_results_count': sum(1 for g, s in zip(gradient_f_values, standard_f_values) if g > s),
        'total_runs': len(standard_results)
    }
    
    # Ausgabe der Ergebnisse
    print(f"\n=== VERGLEICHSERGEBNISSE ===")
    print(f"Standard PSO:")
    print(f"  Bester Wert: {results['standard_pso']['f_opt_best']:.6f}")
    print(f"  Mittelwert: {results['standard_pso']['f_opt_mean']:.6f} ± {results['standard_pso']['f_opt_std']:.6f}")
    print(f"  Zeit: {results['standard_pso']['time_mean']:.3f} ± {results['standard_pso']['time_std']:.3f} s")
    
    print(f"\nGradient-Enhanced PSO:")
    print(f"  Bester Wert: {results['gradient_pso']['f_opt_best']:.6f}")
    print(f"  Mittelwert: {results['gradient_pso']['f_opt_mean']:.6f} ± {results['gradient_pso']['f_opt_std']:.6f}")
    print(f"  Zeit: {results['gradient_pso']['time_mean']:.3f} ± {results['gradient_pso']['time_std']:.3f} s")
    print(f"  Gradient-Nutzung: {results['gradient_pso']['gradient_usage_mean']:.1f}%")
    print(f"  Gradient-Erfolgsrate: {results['gradient_pso']['gradient_success_mean']:.1f}%")
    
    print(f"\nVergleich:")
    print(f"  Verbesserung: {f_improvement:+.2f}%")
    print(f"  Zeit-Overhead: {time_overhead:+.2f}%")
    print(f"  Bessere Ergebnisse: {results['comparison']['better_results_count']}/{results['comparison']['total_runs']}")
    
    return results

def _plot_comparison(results, num_runs):
    """Erstellt Vergleichsplots."""
    if not results:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Zielfunktionswerte
    methods = ['Standard PSO', 'Gradient PSO']
    means = [results['standard_pso']['f_opt_mean'], results['gradient_pso']['f_opt_mean']]
    stds = [results['standard_pso']['f_opt_std'], results['gradient_pso']['f_opt_std']]
    
    ax1.bar(methods, means, yerr=stds, capsize=5, color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Zielfunktionswert')
    ax1.set_title('Vergleich der Zielfunktionswerte')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rechenzeiten
    time_means = [results['standard_pso']['time_mean'], results['gradient_pso']['time_mean']]
    time_stds = [results['standard_pso']['time_std'], results['gradient_pso']['time_std']]
    
    ax2.bar(methods, time_means, yerr=time_stds, capsize=5, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Rechenzeit (s)')
    ax2.set_title('Vergleich der Rechenzeiten')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Verbesserung vs. Overhead
    improvement = results['comparison']['f_improvement_percent']
    overhead = results['comparison']['time_overhead_percent']
    
    ax3.scatter([overhead], [improvement], s=100, c='green', marker='o')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Zeit-Overhead (%)')
    ax3.set_ylabel('Funktionswert-Verbesserung (%)')
    ax3.set_title('Effizienz-Analyse')
    ax3.grid(True, alpha=0.3)
    
    # Füge Annotation hinzu
    ax3.annotate(f'({overhead:.1f}%, {improvement:.1f}%)', 
                (overhead, improvement), xytext=(10, 10), 
                textcoords='offset points')
    
    # Plot 4: Erfolgsrate
    better_count = results['comparison']['better_results_count']
    total_count = results['comparison']['total_runs']
    success_rate = (better_count / total_count) * 100
    
    labels = ['Standard PSO besser', 'Gradient PSO besser']
    sizes = [total_count - better_count, better_count]
    colors = ['blue', 'red']
    
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title(f'Erfolgsrate über {num_runs} Läufe')
    
    plt.tight_layout()
    plt.show()
    
    return fig
'''