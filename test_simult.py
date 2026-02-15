import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import des modules
from geometry import parse_input_file
from pso import PSO_PathPlanner
from rrt import RRT

def run_comparison_view():
    # Liste des scénarios
    scenarios = ["scenario0.txt", "scenario1.txt", "scenario2.txt", "scenario3.txt", "scenario4.txt"]
    
    # Vérification du dossier
    if not os.path.exists("scenario"):
        print("Erreur : Dossier 'scenario' introuvable.")
        return

    # --- 1. CONFIGURATION & AFFICHAGE TERMINAL ---
    # PSO Params
    pso_particles = 30
    pso_waypoints = 7       
    pso_iter = 1000         
    pso_restart = True      

    # RRT Params
    rrt_delta_s = 50.0
    rrt_delta_r = 150.0
    rrt_iter = 2000
    rrt_sampling = True     
    rrt_opt = True          

    print(f"\n{'='*60}")
    print(f"COMPARATIF GLOBAL : CONFIGURATION")
    print(f"{'='*60}")
    print(f"PSO Settings:")
    print(f"  - Particules    : {pso_particles}")
    print(f"  - Waypoints     : {pso_waypoints}")
    print(f"  - Max Iter      : {pso_iter}")
    print(f"  - Random Restart: {pso_restart}")
    print(f"\nRRT Settings:")
    print(f"  - Delta S       : {rrt_delta_s}")
    print(f"  - Delta R       : {rrt_delta_r}")
    print(f"  - Max Iter      : {rrt_iter}")
    print(f"  - Int. Sampling : {rrt_sampling}")
    print(f"  - Triang. Opt   : {rrt_opt}")
    print(f"{'='*60}\n")

    # --- 2. PRÉPARATION FIGURE ---
    n_rows = len(scenarios)
    n_cols = 2
    
    # constrained_layout gère automatiquement les espacements sans chevauchement
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), constrained_layout=True)
    fig.suptitle("Comparatif Global : PSO (Gauche) vs RRT (Droite)", fontsize=16, fontweight='bold')

    # Cas particulier si un seul scénario
    if n_rows == 1:
        axes = np.array([axes])

    # --- 3. BOUCLE DE TRAITEMENT ---
    for i, filename in enumerate(scenarios):
        filepath = os.path.join("scenario", filename)
        ax_pso = axes[i, 0]
        ax_rrt = axes[i, 1]
        
        # Labels Scénario à gauche
        ax_pso.set_ylabel(filename, rotation=0, size='large', labelpad=60, fontweight='bold', va='center')

        print(f"Traitement de {filename}...", end=" ", flush=True)
        
        if not os.path.exists(filepath):
            ax_pso.text(0.5, 0.5, "Fichier introuvable", ha='center')
            print("Introuvable.")
            continue

        try:
            env, start_pos, goal_pos, *rest = parse_input_file(filepath)
        except Exception:
            print("Erreur Parsing.")
            continue
        
        if env is None: 
            print("Env None.")
            continue

        # --- EXECUTION PSO ---
        pso = PSO_PathPlanner(env, start_pos, goal_pos, 
                              num_particles=pso_particles, 
                              num_waypoints=pso_waypoints, 
                              max_iter=pso_iter)
        
        t0 = time.time()
        path_pso, score_pso = pso.optimize(random_restart=pso_restart)
        dt_pso = time.time() - t0
        
        # Plot PSO
        env.plot(ax_pso, path=path_pso)
        ax_pso.set_title(f"Score: {score_pso:.0f} | Temps: {dt_pso:.3f}s", fontsize=10, backgroundcolor='#e6f2ff')
        ax_pso.scatter(*start_pos, c='green', s=60, label='Start', zorder=5)
        ax_pso.scatter(*goal_pos, c='red', marker='*', s=100, label='Goal', zorder=5)

        # --- EXECUTION RRT ---
        rrt = RRT(env, start_pos, goal_pos, 
                  delta_s=rrt_delta_s, 
                  delta_r=rrt_delta_r, 
                  max_iter=rrt_iter)
        
        t0 = time.time()
        result_rrt = rrt.plan(intelligent_sampling=rrt_sampling, triang_opt=rrt_opt)
        dt_rrt = time.time() - t0
        
        path_rrt = None
        cost_rrt = float('inf')
        status_color = "#ffe6e6" # Rouge pâle
        status_txt = "ECHEC"

        if result_rrt is not None:
            path_rrt, cost_rrt = result_rrt
            status_color = "#e6ffe6" # Vert pâle
            status_txt = "SUCCÈS"

        # Plot RRT
        env.plot(ax_rrt, path=path_rrt, color='red')
        title_str = f"{status_txt} | Coût: {cost_rrt:.1f} | Temps: {dt_rrt:.3f}s"
        ax_rrt.set_title(title_str, fontsize=10, backgroundcolor=status_color)

        ax_rrt.scatter(*start_pos, c='green', s=60, zorder=5)
        ax_rrt.scatter(*goal_pos, c='red', marker='*', s=100, zorder=5)

        # Arbre RRT (léger)
        if hasattr(rrt, 'nodes'):
            count = 0
            for node in rrt.nodes:
                if node.parent:
                    p1, p2 = node.pos, node.parent.pos
                    ax_rrt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', alpha=0.1, linewidth=0.5)
                    count += 1
                    if count > 2000: break

        # Nettoyage visuel
        for ax in [ax_pso, ax_rrt]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel(ax.get_ylabel()) # Garde le label scénario
            
        print("OK.")

    print("\nAffichage terminé.")
    plt.show()

if __name__ == "__main__":
    run_comparison_view()