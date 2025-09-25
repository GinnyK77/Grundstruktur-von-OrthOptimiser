"""Visualisierungsfunktionen für Optimierungsergebnisse."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

def visualize_results(result, total_function, constraint_function, implant_parameters, z_threshold, 
                    x_range=(-3, 3), y_range=(-3, 3)):
    """
    Hauptfunktion für die Visualisierung von Optimierungsergebnissen.
    
    Args:
        result: Optimierungsergebnis
        total_function: Gesamtzielfunktion
        constraint_function: Nebenbedingungsfunktion
        implant_parameters: Liste der Implantatparameter
        z_threshold: Schwellenwert für die Nebenbedingung
        x_range, y_range: Bereich für die Visualisierung
    """
    if not result:
        print("Keine Visualisierung möglich - keine zulässige Lösung gefunden.")
        return
    
    # Extrahiere Ergebnisinformationen
    x_opt = result["x_opt"]
    y_opt = result["y_opt"]
    t_opt = result["t_opt"]
    f_opt = result["f_opt"]
    lambda_opt = result.get("lambda_opt", 0.0)
    path = result.get("path", [])
    
    # Formatiere als Tupel für die Visualisierungsfunktion
    optimal_result = (x_opt, y_opt, t_opt, f_opt, lambda_opt, result["implant_params_opt"], path)
    
    # Rufe die Visualisierungsfunktion auf
    visualize_with_implant_type(
        total_function, constraint_function, t_opt, z_threshold,
        optimal_result, implant_parameters,
        x_range=x_range, y_range=y_range
    )

def visualize_with_implant_type(total_function, constraint_function, t_opt, z_threshold, 
                              optimal_result=None, implant_parameters=None, 
                              x_range=(-3, 3), y_range=(-3, 3)):
    """
    Verbesserte Visualisierung der Optimierungslandschaften mit ganzzahligem t.
    
    Parameters:
    - total_function: Funktion zur Erstellung von total_function(x, y, implant_params)
    - constraint_function: Nebenbedingungsfunktion
    - t_opt: Index des optimalen Implantattyps
    - z_threshold: Höhe der horizontalen Ebene für die Nebenbedingung
    - optimal_result: Tupel (x_opt, y_opt, t_opt, f_opt, lambda_opt, implant_params_opt, path)
    - implant_parameters: Liste der Implantatparameter-Wörterbücher
    - x_range, y_range: Grenzen für die Visualisierung
    """
    print("Erstelle Visualisierungen...")
    
    # Extrahiere optimale Parameter, falls vorhanden
    x_opt, y_opt, t_opt, f_opt, lambda_opt, implant_params_opt, path = None, None, None, None, None, None, None
    if optimal_result:
        x_opt, y_opt, t_opt, f_opt, lambda_opt, implant_params_opt, path = optimal_result
    
    # Erstelle ein Gitter für x- und y-Koordinaten
    x = np.linspace(x_range[0], x_range[1], 50)
    y = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x, y)
    
    # Finde die optimalen Punkte für jeden Implantattyp
    print("\nBerechne optimale Punkte für jeden einzelnen Implantattyp:")
    individual_optima = []
    
    for i, params in enumerate(implant_parameters):
        print(f"Optimierung für {params['name']}...")
        
        # Erstelle eine Funktion für diesen speziellen Implantattyp
        def f_with_implant(x, y):
            return total_function(x, y, params)
        
        # Finde lokale optimale Punkte mit einer einfachen Rastersuche
        best_x, best_y, best_z = 0, 0, float('-inf')
        
        # Definiere Rasterauflösung
        x_grid = np.linspace(x_range[0], x_range[1], 31)
        y_grid = np.linspace(y_range[0], y_range[1], 31)
        
        # Durchsuche alle Punkte im Raster
        for xi in x_grid:
            for yi in y_grid:
                # Berechne Zielfunktionswert
                z = f_with_implant(xi, yi)
                
                # Prüfe, ob Nebenbedingung erfüllt ist
                constraint_value = f_with_implant(xi, yi) - z_threshold
                
                # Falls zulässig und besser als bisherige beste Lösung
                if constraint_value >= -1e-6 and z > best_z:
                    best_x, best_y, best_z = xi, yi, z
        
        individual_optima.append((best_x, best_y, best_z))
        print(f"  Optimaler Punkt für {params['name']}: x={best_x:.4f}, y={best_y:.4f}, f(x,y)={best_z:.4f}")
    
    # ====================================================================
    # 1. Erstelle Fenster: Analytik und Statistik
    # ====================================================================
    fig_analytics = plt.figure(figsize=(16, 10))
    gs_analytics = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Erstelle Formel der Zielfunktion für Titel
    formula = f'f(x,y,t) für Implantattyp {implant_parameters[t_opt]["name"]}'
    
    fig_analytics.suptitle(f'Optimierungsanalyse mit Implantattyp als ganzzahliger Variable\n{formula}\nNebenbedingung: f(x,y,t) ≥ {z_threshold}', fontsize=16)
    
    # ====================================================================
    # 1.1 Bereich: 2D-Konvergenzpfad
    # ====================================================================
    if optimal_result and path:
        ax_convergence = fig_analytics.add_subplot(gs_analytics[0, 0])
        
        # Extrahiere x, y, t aus Pfad
        path_x, path_y, path_t = zip(*path)
        
        # Zeichne 2D-Konvergenzpfad
        sc = ax_convergence.scatter(path_x, path_y, c=path_t, cmap='viridis', 
                                   marker='o', s=30, edgecolors='black', linewidths=0.5)
        
        # Verbinde Punkte mit Linien
        ax_convergence.plot(path_x, path_y, 'k-', alpha=0.3, linewidth=0.5)
        
        # Markiere Start- und Endpunkte
        ax_convergence.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
        ax_convergence.plot(path_x[-1], path_y[-1], 'r*', markersize=15, label='Optimum')
        
        # Zeichne Konturen für optimalen Implantattyp
        implant_params_opt = implant_parameters[t_opt]
        
        # Berechne Zielfunktion für optimalen Implantattyp
        Z_opt = np.zeros_like(X)
        for ix in range(Z_opt.shape[0]):
            for iy in range(Z_opt.shape[1]):
                Z_opt[ix, iy] = total_function(X[ix, iy], Y[ix, iy], implant_params_opt)
        
        # Zeichne Konturen
        levels = np.linspace(np.min(Z_opt), np.max(Z_opt), 10)
        cp = ax_convergence.contour(X, Y, Z_opt, levels=levels, cmap='coolwarm', alpha=0.5)
        
        # Zeichne Kontur für Nebenbedingung
        cs = ax_convergence.contour(X, Y, Z_opt, levels=[z_threshold], colors='green', linestyles='--', linewidths=2)
        ax_convergence.clabel(cs, inline=True, fontsize=8, fmt=f'z = {z_threshold}')
        
        # Farbskala für Implantattyp
        cbar = plt.colorbar(sc, ax=ax_convergence, ticks=np.arange(len(implant_parameters)))
        cbar.set_label('Implantattyp t')
        cbar.ax.set_yticklabels([imp["name"] for imp in implant_parameters])
        
        ax_convergence.set_title('Konvergenzpfad (x,y)', fontsize=12)
        ax_convergence.set_xlabel('x')
        ax_convergence.set_ylabel('y')
        ax_convergence.grid(True, alpha=0.3)
        ax_convergence.legend()
        
        # Setze Visualisierungsgrenzen
        ax_convergence.set_xlim(x_range)
        ax_convergence.set_ylim(y_range)
    
    # ====================================================================
    # 1.2 Bereich: Funktionswert- und t-Wert-Progression
    # ====================================================================
    if optimal_result and path:
        ax_values = fig_analytics.add_subplot(gs_analytics[0, 1])
        
        # Extrahiere t aus Pfad und berechne Funktionswerte
        path_x, path_y, path_t = zip(*path)
        iterations = np.arange(len(path_x))
        
        # Erstelle eine Funktion für jede Iteration basierend auf dem Implantattyp
        path_f = []
        for i in range(len(path)):
            x, y, t = path[i]
            params = implant_parameters[t]
            f_val = total_function(x, y, params)
            path_f.append(f_val)
        
        # Primäre y-Achse für Funktionswerte
        line1 = ax_values.plot(iterations, path_f, 'b-', linewidth=2, label='f(x,y,t)')
        ax_values.set_ylabel('Funktionswert f(x,y,t)', color='blue')
        ax_values.tick_params(axis='y', labelcolor='blue')
        
        # Horizontale Linie für z_threshold
        ax_values.axhline(y=z_threshold, color='g', linestyle='--', alpha=0.7, label=f'z = {z_threshold}')
        
        # Markiere optimalen Funktionswert
        ax_values.plot(iterations[-1], path_f[-1], 'r*', markersize=10, 
                      label=f'Optimum: f = {path_f[-1]:.4f}')
        
        # Sekundäre y-Achse für t-Werte
        ax_t = ax_values.twinx()
        line2 = ax_t.plot(iterations, path_t, 'r-', linewidth=2, alpha=0.6, label='Implantattyp t')
        ax_t.set_ylabel('Implantattyp t', color='red')
        ax_t.tick_params(axis='y', labelcolor='red')
        ax_t.set_yticks(np.arange(len(implant_parameters)))
        ax_t.set_yticklabels([imp["name"] for imp in implant_parameters])
        
        # Kombiniere Legenden
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_values.legend(lines, labels, loc='upper left')
        
        ax_values.set_title('Funktionswert- und Implantattyp-Progression', fontsize=12)
        ax_values.set_xlabel('Iteration')
        ax_values.grid(True, alpha=0.3)
    
    # ====================================================================
    # 1.3 Bereich: Vergleich der Implantattypen (lokale Optima)
    # ====================================================================
    if optimal_result:
        ax_compare = fig_analytics.add_subplot(gs_analytics[1, 0])
        
        # Berechne Funktionswerte für alle Implantattypen
        implant_names = [imp["name"] for imp in implant_parameters]
        
        # Lokale Optimalwerte für jeden Implantattyp
        f_values_local = [opt[2] for opt in individual_optima]
        # Ersetze den Wert für den global optimalen Implantattyp durch den vom Hauptsolver gefundenen Wert
        f_values_local[t_opt] = f_opt
        
        # Erstelle Balkendiagramm mit einem Balken pro Implantattyp
        x_pos = np.arange(len(implant_parameters))
        
        # Standardfarbe für Balken
        colors = ['lightblue' for _ in range(len(implant_parameters))]
        # Hebe optimalen Implantattyp hervor
        colors[t_opt] = 'red'
        
        # Zeichne Balken
        bars = ax_compare.bar(x_pos, f_values_local, color=colors, width=0.6)
        
        # Horizontale Linie für z_threshold
        ax_compare.axhline(y=z_threshold, color='g', linestyle='-', alpha=0.7,
                          label=f'z = {z_threshold}')
        
        # Zeichne Werte über Balken
        for i, bar in enumerate(bars):
            height = bar.get_height()
            color = 'black' if height >= z_threshold else 'gray'
            weight = 'bold' if i == t_opt else 'normal'
            label = f'{height:.4f}' + (" (global)" if i == t_opt else "")
            ax_compare.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        label, ha='center', va='bottom', color=color,
                        fontweight=weight)

        ax_compare.set_title(f'Vergleich der Optima für jeden Implantattyp', fontsize=12)
        ax_compare.set_xlabel('Implantattyp')
        ax_compare.set_ylabel('f(x,y) am lokalen Optimum')
        ax_compare.set_xticks(x_pos)
        ax_compare.set_xticklabels(implant_names, rotation=45, ha='right')
        ax_compare.grid(True, alpha=0.3)
    
    # ====================================================================
    # 1.4 Bereich: Ergebnistabelle und Zusammenfassung
    # ====================================================================
    if optimal_result:
        ax_results = fig_analytics.add_subplot(gs_analytics[1, 1])
        ax_results.set_axis_off()
        
        table_data = []
        headers = ["Typ", "x", "y", "f(x,y)", "f ≥ z?", "Typ"]

        for i, opt in enumerate(individual_optima):
            x_i, y_i, f_i = opt
            # Für global optimalen Implantattyp, verwende Werte vom Hauptsolver
            if i == t_opt:
                x_i, y_i, f_i = x_opt, y_opt, f_opt
                optimum_type = "Global"
            else:
                optimum_type = "Lokal"
                
            is_feasible = f_i >= z_threshold
            
            row = [
                implant_parameters[i]["name"],
                f"{x_i:.4f}",
                f"{y_i:.4f}",
                f"{f_i:.4f}",
                "✓" if is_feasible else "✗",
                optimum_type
            ]
            
            table_data.append(row)
            
        # Erstelle Tabelle
        table = ax_results.table(
            cellText=table_data,
            colLabels=headers,
            loc='center',
            cellLoc='center'
        )
        
        # Formatiere Tabelle
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Hebe optimalen Implantattyp hervor
        for j in range(len(headers)):
            table[(1 + t_opt, j)].set_facecolor("lightcoral")
        
        ax_results.set_title("Zusammenfassung der lokalen Optima für jeden Implantattyp", fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)
    
    # ====================================================================
    # 2. Erstelle Fenster: 3D-Plots für alle Implantattypen
    # ====================================================================
    fig_3d = plt.figure(figsize=(18, 12))
    fig_3d.suptitle(f'3D-Visualisierung aller Implantattypen mit ganzzahligem t', fontsize=16)
    
    # Erstelle Raster für 3D-Plots (2 Zeilen, 3 Spalten)
    gs_3d = gridspec.GridSpec(2, 3)
    
    # Erstelle Plot für jeden Implantattyp
    for i, implant_params in enumerate(implant_parameters):
        # Berechne Position im 2x3-Raster
        row, col = divmod(i, 3)
        
        # Erstelle 3D-Plot
        ax = fig_3d.add_subplot(gs_3d[row, col], projection='3d')
        
        # Berechne Zielfunktion für diesen Implantattyp
        Z = np.zeros_like(X)
        for ix in range(Z.shape[0]):
            for iy in range(Z.shape[1]):
                Z[ix, iy] = total_function(X[ix, iy], Y[ix, iy], implant_params)
        
        # Zeichne Oberfläche
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7)
        
        # Zeichne horizontale Ebene bei z = z_threshold
        x_plane = np.linspace(x_range[0], x_range[1], 10)
        y_plane = np.linspace(y_range[0], y_range[1], 10)
        X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
        Z_plane = np.ones_like(X_plane) * z_threshold
        ax.plot_surface(X_plane, Y_plane, Z_plane, color='green', alpha=0.3)
        
        # Lokale optimale Koordinaten für diesen Implantattyp
        x_i_opt, y_i_opt, z_i_opt = individual_optima[i]
        
        # Klare Unterscheidung - nur ein Punkt pro Plot:
        if optimal_result and t_opt == i:
            # Für global optimalen Implantattyp: Zeige NUR globalen optimalen Punkt
            ax.scatter([x_opt], [y_opt], [f_opt], color='yellow', s=150, marker='*',
                      edgecolor='black', linewidth=1, label='Globaler optimaler Punkt')
            
            # Zeichne vertikale Linie zu z_threshold, falls darüber
            if f_opt >= z_threshold:
                ax.plot([x_opt, x_opt], [y_opt, y_opt], [z_threshold, f_opt], 
                      color='yellow', linestyle='--', linewidth=1.5)
        else:
            # Für alle anderen Implantattypen: Zeige NUR lokalen optimalen Punkt
            ax.scatter([x_i_opt], [y_i_opt], [z_i_opt], color='blue', s=100, marker='o',
                      edgecolor='white', linewidth=1, label='Lokaler optimaler Punkt')
            
            # Zeichne vertikale Linie zu z_threshold, falls darüber
            if z_i_opt >= z_threshold:
                ax.plot([x_i_opt, x_i_opt], [y_i_opt, y_i_opt], [z_threshold, z_i_opt], 
                      color='blue', linestyle='--', linewidth=1.5)
        
        # Beschriftung
        title_color = 'red' if optimal_result and t_opt == i else 'black'
        title_weight = 'bold' if optimal_result and t_opt == i else 'normal'
        
        title = f'{implant_params["name"]}: ' + \
                f'a₁={implant_params["a_1"]}, a₂={implant_params["a_2"]}, ' + \
                f'a₃={implant_params["a_3"]}, a₄={implant_params["a_4"]}\n'
                
        if optimal_result and t_opt == i:
            title += f'GLOBAL OPTIMAL: f({x_opt:.2f}, {y_opt:.2f}) = {f_opt:.2f}'
        else:
            title += f'Optimaler Punkt: f({x_i_opt:.2f}, {y_i_opt:.2f}) = {z_i_opt:.2f}'
        
        ax.set_title(title, fontsize=10, color=title_color, fontweight=title_weight)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')
        
        # Gleicher Betrachtungswinkel für alle Teilplots
        ax.view_init(elev=30, azim=45)
        
        # Gleiche Achsenskalierung für alle Plots
        ax.set_box_aspect([1, 1, 0.8])  # Besseres Verhältnis für 3D-Plots
    
    # Legende für 3D-Plots
    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', markeredgecolor='black',
                  markersize=15, label='Globaler optimaler Punkt'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                  markersize=10, label='Lokaler optimaler Punkt für Implantattyp')
    ]
    
    # Legende in separater Achse unter Plots
    legend_ax = fig_3d.add_axes([0.35, 0.02, 0.3, 0.03])  # Position [links, unten, Breite, Höhe]
    legend_ax.axis('off')
    legend_ax.legend(handles=legend_elements, loc='center', ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    
    # Zeige Plots
    plt.show()