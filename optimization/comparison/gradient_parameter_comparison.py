"""Erweiterte Vergleichsstudie für PSO-Gradient-Parameter."""
import time
import numpy as np
import matplotlib.pyplot as plt

def comprehensive_gradient_pso_comparison(total_function_with_t, constraint_with_t, implant_parameters, z_threshold,
                                        x_bounds=(-3, 3), y_bounds=(-3, 3), 
                                        n_particles=20, max_iterations=50, num_runs=3):
    """Umfassender Vergleich verschiedener Gradient-PSO Konfigurationen."""
    
    from ..solvers.particle_swarm import ParticleSwarmSolver
    from ..solvers.pso_gradient import GradientEnhancedPSOSolver
    
    print(f"=== UMFASSENDE PSO-GRADIENT PARAMETER-STUDIE ===")
    print(f"Läufe pro Konfiguration: {num_runs}")
    print(f"Partikel: {n_particles}, Iterationen: {max_iterations}")
    
    # Test-Konfigurationen definieren
    test_configurations = [
        # Baseline: Standard PSO
        {"name": "Standard PSO", "type": "standard", "c3": 0.0, "prob": 0.0},
        
        # Verschiedene Gradientengewichte bei konstanter Wahrscheinlichkeit (0.5)
        {"name": "Gradient c3=0.1, p=0.5", "type": "gradient", "c3": 0.1, "prob": 0.5},
        {"name": "Gradient c3=0.3, p=0.5", "type": "gradient", "c3": 0.3, "prob": 0.5},
        {"name": "Gradient c3=0.5, p=0.5", "type": "gradient", "c3": 0.5, "prob": 0.5},
        {"name": "Gradient c3=1.0, p=0.5", "type": "gradient", "c3": 1.0, "prob": 0.5},
        
        # Verschiedene Wahrscheinlichkeiten bei konstantem Gradientengewicht (0.3)
        {"name": "Gradient c3=0.3, p=0.2", "type": "gradient", "c3": 0.3, "prob": 0.2},
        {"name": "Gradient c3=0.3, p=0.7", "type": "gradient", "c3": 0.3, "prob": 0.7},
        {"name": "Gradient c3=0.3, p=0.9", "type": "gradient", "c3": 0.3, "prob": 0.9},
        
        # Extreme Konfigurationen
        {"name": "Gradient c3=0.1, p=0.9", "type": "gradient", "c3": 0.1, "prob": 0.9},
        {"name": "Gradient c3=1.0, p=0.2", "type": "gradient", "c3": 1.0, "prob": 0.2},
    ]
    
    # Ergebnisspeicher
    all_results = {}
    
    # Führe Tests für alle Konfigurationen durch
    for config in test_configurations:
        print(f"\n=== Testing {config['name']} ===")
        
        config_results = []
        
        for run in range(num_runs):
            print(f"  Lauf {run + 1}/{num_runs}...")
            
            start_time = time.time()
            
            if config["type"] == "standard":
                # Standard PSO
                solver = ParticleSwarmSolver(
                    total_function_with_t, constraint_with_t, 
                    x_bounds, y_bounds, z_threshold, implant_parameters
                )
                
                result = solver.solve(
                    n_particles=n_particles, max_iterations=max_iterations, verbose=False
                )
                
                # Standard-PSO hat keine Gradientenstatistiken
                gradient_stats = {
                    "gradient_usage_percent": 0.0,
                    "gradient_success_rate": 0.0,
                    "gradient_improvement_rate": 0.0,
                    "total_gradient_calls": 0,
                    "gradient_improvements": 0,
                    "total_gradient_attempts": 0  # HINZUGEFÜGT
                }
                
            else:
                # Gradient-Enhanced PSO
                solver = GradientEnhancedPSOSolver(
                    total_function_with_t, constraint_with_t, 
                    x_bounds, y_bounds, z_threshold, implant_parameters
                )
                
                result = solver.solve(
                    n_particles=n_particles, max_iterations=max_iterations, 
                    c3_gradient=config["c3"], gradient_probability=config["prob"], 
                    adaptive_gradient=True, verbose=False
                )
                
                gradient_stats = result.get('gradient_statistics', {}) if result else {}
                
                # BUG-FIX: Berechne korrekte Erfolgsrate
                total_calls = gradient_stats.get('total_gradient_calls', 0)
                improvements = gradient_stats.get('gradient_improvements', 0)
                
                # Erfolgsrate = Verbesserungen / Gradientenverwendungen (nicht über 100%)
                if total_calls > 0:
                    gradient_stats['gradient_success_rate'] = min(100.0, (improvements / total_calls) * 100)
                else:
                    gradient_stats['gradient_success_rate'] = 0.0
            
            solve_time = time.time() - start_time
            
            if result:
                config_results.append({
                    'f_opt': result['f_opt'],
                    'time': solve_time,
                    'convergence_iterations': len(result.get('path', [])),
                    'x_opt': result['x_opt'],
                    'y_opt': result['y_opt'],
                    't_opt': result['t_opt'],
                    'gradient_stats': gradient_stats
                })
                
                print(f"    Ergebnis: f={result['f_opt']:.4f}, Zeit: {solve_time:.3f}s")
                if config["type"] == "gradient":
                    print(f"    Gradient-Nutzung: {gradient_stats.get('gradient_usage_percent', 0):.1f}%")
                    print(f"    Gradient-Erfolgsrate: {gradient_stats.get('gradient_success_rate', 0):.1f}%")
            else:
                print(f"    Keine Lösung gefunden")
        
        all_results[config['name']] = {
            'config': config,
            'results': config_results
        }
    
    # Analysiere und visualisiere Ergebnisse
    analysis_results = _analyze_comprehensive_results(all_results)
    _visualize_comprehensive_results(analysis_results, test_configurations)
    
    return analysis_results

def _analyze_comprehensive_results(all_results):
    """Analysiert die umfassenden Testergebnisse."""
    
    analysis = {}
    
    for config_name, data in all_results.items():
        if not data['results']:
            continue
            
        results = data['results']
        config = data['config']
        
        # Grundstatistiken
        f_values = [r['f_opt'] for r in results]
        times = [r['time'] for r in results]
        gradient_usages = [r['gradient_stats'].get('gradient_usage_percent', 0) for r in results]
        gradient_success_rates = [r['gradient_stats'].get('gradient_success_rate', 0) for r in results]
        gradient_improvements = [r['gradient_stats'].get('gradient_improvements', 0) for r in results]
        
        analysis[config_name] = {
            'config': config,
            'f_opt_mean': np.mean(f_values),
            'f_opt_std': np.std(f_values),
            'f_opt_best': np.max(f_values),
            'time_mean': np.mean(times),
            'time_std': np.std(times),
            'gradient_usage_mean': np.mean(gradient_usages),
            'gradient_success_rate_mean': np.mean(gradient_success_rates),
            'gradient_improvements_mean': np.mean(gradient_improvements),
            'success_rate': len([f for f in f_values if f > 0]) / len(f_values) * 100,
            'num_runs': len(results)
        }
    
    # BUG-FIX: Berechne Vergleiche zur Standard-PSO
    if "Standard PSO" in analysis:
        baseline = analysis["Standard PSO"]
        
        for config_name, stats in analysis.items():
            if config_name != "Standard PSO":
                # Relative Verbesserung
                f_improvement = ((stats['f_opt_mean'] - baseline['f_opt_mean']) / 
                               baseline['f_opt_mean'] * 100)
                
                # Zeit-Overhead
                time_overhead = ((stats['time_mean'] - baseline['time_mean']) / 
                               baseline['time_mean'] * 100)
                
                # Effizienz
                efficiency = f_improvement / max(time_overhead, 1) if time_overhead > 0 else float('inf')
                
                stats['vs_baseline'] = {
                    'f_improvement_percent': f_improvement,
                    'time_overhead_percent': time_overhead,
                    'efficiency_ratio': efficiency
                }
    
    return analysis

def _visualize_comprehensive_results(analysis, configurations):
    """Erstellt umfassende Visualisierungen der Ergebnisse."""
    
    if not analysis:
        print("Keine Ergebnisse zum Visualisieren.")
        return
    
    # Erstelle DataFrame-ähnliche Struktur für einfachere Handhabung
    data_for_plots = []
    for config_name, stats in analysis.items():
        config = stats['config']
        row = {
            'name': config_name,
            'short_name': config_name.replace('Gradient ', '').replace('Standard ', 'Std'),
            'type': config['type'],
            'c3': config.get('c3', 0),
            'prob': config.get('prob', 0),
            'f_opt_mean': stats['f_opt_mean'],
            'f_opt_std': stats['f_opt_std'],
            'time_mean': stats['time_mean'],
            'gradient_usage': stats['gradient_usage_mean'],
            'gradient_success_rate': stats['gradient_success_rate_mean'],
            'improvement': stats.get('vs_baseline', {}).get('f_improvement_percent', 0),
            'overhead': stats.get('vs_baseline', {}).get('time_overhead_percent', 0),
            'efficiency': stats.get('vs_baseline', {}).get('efficiency_ratio', 0)
        }
        data_for_plots.append(row)
    
    # Erstelle Multi-Plot Visualisierung
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Funktionswerte nach Gradientengewicht
    ax1 = plt.subplot(2, 4, 1)
    gradient_configs = [d for d in data_for_plots if d['type'] == 'gradient']
    c3_varies = [d for d in gradient_configs if d['prob'] == 0.5]  # Konstante Wahrscheinlichkeit
    
    if c3_varies:
        c3_values = [d['c3'] for d in c3_varies]
        f_means = [d['f_opt_mean'] for d in c3_varies]
        f_stds = [d['f_opt_std'] for d in c3_varies]
        
        ax1.errorbar(c3_values, f_means, yerr=f_stds, marker='o', capsize=5)
        
        # Baseline hinzufügen
        baseline = next((d for d in data_for_plots if d['name'] == 'Standard PSO'), None)
        if baseline:
            ax1.axhline(y=baseline['f_opt_mean'], color='red', linestyle='--', label='Standard PSO')
        
        ax1.set_xlabel('Gradientengewicht c3')
        ax1.set_ylabel('Funktionswert (Mittelwert)')
        ax1.set_title('Funktionswert vs. Gradientengewicht\n(p=0.5 konstant)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Funktionswerte nach Wahrscheinlichkeit
    ax2 = plt.subplot(2, 4, 2)
    prob_varies = [d for d in gradient_configs if d['c3'] == 0.3]  # Konstantes Gewicht
    
    if prob_varies:
        prob_values = [d['prob'] for d in prob_varies]
        f_means = [d['f_opt_mean'] for d in prob_varies]
        f_stds = [d['f_opt_std'] for d in prob_varies]
        
        ax2.errorbar(prob_values, f_means, yerr=f_stds, marker='s', capsize=5, color='green')
        
        # Baseline hinzufügen
        if baseline:
            ax2.axhline(y=baseline['f_opt_mean'], color='red', linestyle='--', label='Standard PSO')
        
        ax2.set_xlabel('Gradientenwahrscheinlichkeit')
        ax2.set_ylabel('Funktionswert (Mittelwert)')
        ax2.set_title('Funktionswert vs. Wahrscheinlichkeit\n(c3=0.3 konstant)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Verbesserung vs. Overhead (Effizienzanalyse)
    ax3 = plt.subplot(2, 4, 3)
    gradient_only = [d for d in data_for_plots if d['type'] == 'gradient']
    
    if gradient_only:
        overheads = [d['overhead'] for d in gradient_only]
        improvements = [d['improvement'] for d in gradient_only]
        c3_colors = [d['c3'] for d in gradient_only]
        
        scatter = ax3.scatter(overheads, improvements, c=c3_colors, s=100, cmap='viridis', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Zeit-Overhead (%)')
        ax3.set_ylabel('Funktionswert-Verbesserung (%)')
        ax3.set_title('Effizienzanalyse')
        
        # Füge Farbbalken hinzu
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Gradientengewicht c3')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Gradientennutzung
    ax4 = plt.subplot(2, 4, 4)
    if gradient_only:
        names = [d['short_name'] for d in gradient_only]
        usages = [d['gradient_usage'] for d in gradient_only]
        
        bars = ax4.bar(range(len(gradient_only)), usages, color='purple', alpha=0.7)
        ax4.set_xlabel('Konfiguration')
        ax4.set_ylabel('Gradientennutzung (%)')
        ax4.set_title('Gradientennutzung pro Konfiguration')
        ax4.set_xticks(range(len(gradient_only)))
        ax4.set_xticklabels([f"c3={d['c3']:.1f}\np={d['prob']:.1f}" 
                           for d in gradient_only], rotation=45)
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Erfolgsraten
    ax5 = plt.subplot(2, 4, 5)
    if gradient_only:
        success_rates = [d['gradient_success_rate'] for d in gradient_only]
        
        bars = ax5.bar(range(len(gradient_only)), success_rates, color='orange', alpha=0.7)
        ax5.set_xlabel('Konfiguration')
        ax5.set_ylabel('Gradient-Erfolgsrate (%)')
        ax5.set_title('Gradient-Erfolgsrate pro Konfiguration')
        ax5.set_xticks(range(len(gradient_only)))
        ax5.set_xticklabels([f"c3={d['c3']:.1f}\np={d['prob']:.1f}" 
                           for d in gradient_only], rotation=45)
        ax5.set_ylim(0, 100)
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Balkendiagramm aller Konfigurationen
    ax6 = plt.subplot(2, 4, (6, 7))
    
    config_names = [d['short_name'] for d in data_for_plots]
    f_means = [d['f_opt_mean'] for d in data_for_plots]
    f_stds = [d['f_opt_std'] for d in data_for_plots]
    colors = ['red' if 'Standard' in d['name'] else 'blue' for d in data_for_plots]
    
    bars = ax6.bar(range(len(data_for_plots)), f_means, yerr=f_stds, 
                  capsize=5, color=colors, alpha=0.7)
    ax6.set_xlabel('Konfiguration')
    ax6.set_ylabel('Funktionswert (Mittelwert ± Std)')
    ax6.set_title('Vergleich aller Konfigurationen')
    ax6.set_xticks(range(len(data_for_plots)))
    ax6.set_xticklabels(config_names, rotation=45, ha='right')
    ax6.grid(True, alpha=0.3)
    
    # Füge Werte über Balken hinzu
    for i, (bar, f_mean) in enumerate(zip(bars, f_means)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + f_stds[i],
                f'{f_mean:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 8: Zusammenfassung Tabelle
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Erstelle Tabelle mit Top-Performern (BUG-FIX)
    top_configs = sorted(data_for_plots, key=lambda x: x['f_opt_mean'], reverse=True)[:5]
    
    table_data = []
    headers = ["Rang", "Konfiguration", "f_opt", "Verbesserung", "Overhead"]
    
    for i, config in enumerate(top_configs):
        table_data.append([
            f"{i+1}",
            config['short_name'][:15],  # Kürzen für bessere Darstellung
            f"{config['f_opt_mean']:.4f}",
            f"{config['improvement']:+.1f}%" if config['improvement'] != 0 else "Baseline",
            f"{config['overhead']:+.1f}%" if config['overhead'] != 0 else "Baseline"
        ])
    
    table = ax8.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    ax8.set_title('Top 5 Konfigurationen', fontsize=12, pad=20)
    
    plt.tight_layout()
    plt.show()
    
    # BUG-FIX: Korrekte textuelle Zusammenfassung
    print("\n" + "="*80)
    print("ZUSAMMENFASSUNG DER GRADIENTENPARAMETER-STUDIE")
    print("="*80)
    
    if "Standard PSO" in analysis:
        baseline = analysis["Standard PSO"]
        print(f"Baseline (Standard PSO): f_opt = {baseline['f_opt_mean']:.4f} ± {baseline['f_opt_std']:.4f}")
        print(f"                         Zeit   = {baseline['time_mean']:.3f} ± {baseline['time_std']:.3f} s")
        print()
    
    # BUG-FIX: Finde tatsächlich beste Konfiguration
    best_config_name, best_config_stats = max(analysis.items(), key=lambda x: x[1]['f_opt_mean'])
    print(f"BESTE KONFIGURATION: {best_config_name}")
    print(f"  Funktionswert: {best_config_stats['f_opt_mean']:.4f} ± {best_config_stats['f_opt_std']:.4f}")
    
    if 'vs_baseline' in best_config_stats:
        vs_base = best_config_stats['vs_baseline']
        print(f"  Verbesserung: {vs_base['f_improvement_percent']:+.1f}%")
        print(f"  Zeit-Overhead: {vs_base['time_overhead_percent']:+.1f}%")
        print(f"  Effizienz: {vs_base['efficiency_ratio']:.3f}")
    
    print("\nEMPFEHLUNGEN:")
    
    # Finde optimale Parameter (nur Gradient-Konfigurationen)
    gradient_configs = {k: v for k, v in analysis.items() if k != "Standard PSO"}
    
    if gradient_configs:
        best_gradient_name, best_gradient_stats = max(gradient_configs.items(), 
                                                     key=lambda x: x[1]['f_opt_mean'])
        
        if best_gradient_stats['f_opt_mean'] > baseline['f_opt_mean']:
            print(f"  Beste Gradient-Konfiguration: {best_gradient_name}")
            best_config = best_gradient_stats['config']
            print(f"    → c3={best_config.get('c3', 0):.1f}, p={best_config.get('prob', 0):.1f}")
        else:
            print("  Standard PSO ist allen Gradient-Konfigurationen überlegen")
            print("  Empfehlung: Verwende Standard PSO oder teste andere Parameter")
    
    print("="*80)
    
    return data_for_plots