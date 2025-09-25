# OrthOptimiser

Optimierung einer gewichteten Zielfunktion **f(x, y, t)** mit einer **ganzzahligen** Design-Variable **t** (Implantattyp).  
Enthält mehrere Solver (Augmented Lagrangian, PSO, Gradient-Enhanced PSO, optional GEKKO & SciPy) sowie Visualisierung (2D/3D).

---

## Projektstruktur

```bash
.
├─ README.md
├─ main.py
├─ optimization/
│  ├─ config/
│  │  ├─ __init__.py
│  │  ├─ parameters.py
│  │  ├─ functions.py
│  │  └─ constraints.py
│  ├─ solvers/
│  │  ├─ __init__.py
│  │  ├─ base_solver.py
│  │  ├─ augmented_lagrangian.py
│  │  ├─ particle_swarm.py
│  │  ├─ pso_gradient.py
│  │  ├─ scipy_solver.py
│  │  └─ gekko_solver.py
│  ├─ strategies/
│  │  ├─ __init__.py
│  │  └─ starting_points.py
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ numeric.py
│  │  └─ visualization.py
│  └─ comparison/
│     ├─ pso_comparison.py
│     └─ gradient_parameter_comparison.py
└─ requirements.txt  (optional)
```



---

## Installation

Empfohlen: **Python 3.9+** (virtuelle Umgebung).

```bash
# Windows
py -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Grundpakete
pip install -U pip
pip install numpy matplotlib
```
---

Optionale Solver-Abhängigkeiten:

```bash
pip install scipy   # für --solver scipy
pip install gekko   # für --solver gekko
```

---


## Nutzung

Starte das Programm aus dem Projektwurzelordner:

```bash
# Windows
py main.py [OPTIONEN]

# macOS / Linux
python3 main.py [OPTIONEN]
```

### Wichtigste Optionen (CLI):

Solverwahl:
```bash
--solver {augmented_lagrangian, pso, pso_gradient, gekko, scipy}
```
(Standard: pso)

PSO (Standard):
```bash
--particles INT Anzahl Partikel (Standard: 20)
--iterations INT Iterationen (Standard: 50)
```

Gradient-Enhanced PSO:
```bash
--gradient_weight FLOAT Gewicht c3 (Standard: 0.3)
--gradient_prob FLOAT Wahrscheinlichkeit (Standard: 0.5)
--compare_pso PSO vs. Gradient-PSO Kurzvergleich
--comprehensive_gradient_study umfassende Parameterstudie
--gradient_study_runs INT Läufe pro Konfiguration (Standard: 3)
```

SciPy / GEKKO:
```bash
--integer_constraints (ganzzahlige x,y für diese Solver)
```

Gewichte der Teilfunktionen:
```bash
--omega1 FLOAT (f1, Standard 1.0)
--omega2 FLOAT (f2, Standard 1.0)
--omega3 FLOAT (f3, Standard 1.0)
--omega4 FLOAT (f4, Standard 0.0)
```

Nebenbedingung & Bereich:
```bash
--threshold FLOAT z-Schwelle (Standard 0.3)
--x_min FLOAT / --x_max FLOAT (Standard -3.0 / 3.0)
--y_min FLOAT / --y_max FLOAT (Standard -3.0 / 3.0)
```

Sonstiges:
```bash
--no_vis Visualisierung deaktivieren
--verbose, -v, -vv, -vvv mehr Ausgabe
--interactive interaktive Auswahl (Funktionen/Gewichte) im Terminal
```

---

## Quickstart-Beispiele

### 1) Interaktiver Modus
```bash
py main.py --interactive
```

### 2) Standard-PSO
```bash
py main.py --solver pso --particles 30 --iterations 80
```

### 3) Gradient-Enhanced PSO
```bash
py main.py --solver pso_gradient --particles 40 --iterations 80 ^
  --gradient_weight 0.3 --gradient_prob 0.5
```

### 4) PSO-Vergleich (Standard vs. Gradient)
```bash
py main.py --compare_pso --particles 30 --iterations 60
```

### 5) Augmented Lagrangian (kombinierte Startwerte)
```bash
py main.py --solver augmented_lagrangian --strategy combined --num_points 150
```

### 6) SciPy-SLSQP 
```bash
py main.py --solver scipy --integer_constraints
```

### 7) GEKKO MINLP
```bash
py main.py --solver gekko --integer_constraints
```

### 8) Eigene Gewichte & Bereich
```bash
py main.py --solver pso ^
  --omega1 1.0 --omega2 0.8 --omega3 1.2 --omega4 0.0 ^
  --threshold 0.5 ^
  --x_min -4 --x_max 4 --y_min -4 --y_max 4
```

---

## Mathematisches Setup (Kurz)
Registrierte Teilfunktionen (optimization/config/functions.py):
```bash
f1(x,y) = x * exp(-(x² + y²) / a₁)

f2(x,y) = (x² + y²) / a₄

f3(x,y) = a₃ * sin(x * y * a₂)

f4(x,y) = a₅ * cos(x * y / a₆)
```
Die Gesamtfunktion ist die gewichtete Summe aktiver Teilfunktionen.
t ist ganzzahlig und wählt den Implantattyp (A..F) → bestimmt Parameter a₁..a₆ (siehe parameters.py).
Nebenbedingung: f(x,y,t) ≥ z_threshold (horizontale Ebene).

---

## Visualisierung

Wenn nicht ```bash --no_vis``` gesetzt, erzeugt das Programm:

-> 2D-Konvergenzpfad,

-> Verlauf von Funktionswert & t,

-> Balkenvergleich pro Implantattyp,

-> 3D-Flächen je Typ.

Erfordert matplotlib.

