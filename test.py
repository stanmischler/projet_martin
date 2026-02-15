import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import des modules
from geometry import parse_input_file
from pso import PSO_PathPlanner
from rrt import RRT

<<<<<<< HEAD
def run_comparison_view():
    # Liste des scénarios
=======
def smooth_path(env, path):
    """
    Simplifie le chemin en essayant de relier les points directements.
    Supprime les zigzags inutiles.
    """
    if path is None or len(path) < 3:
        return path
        
    # On convertit en liste pour pouvoir modifier facilement
    smooth_path = [path[0]]
    current_idx = 0
    
    while current_idx < len(path) - 1:
        # On cherche le point le plus loin possible connectable en ligne droite
        next_valid_idx = current_idx + 1
        
        for i in range(len(path) - 1, current_idx, -1):
            # Si on peut relier 'current' à 'i' sans collision...
            if not env.check_collision_segment(path[current_idx], path[i]):
                next_valid_idx = i
                break
        
        # On ajoute ce point et on avance
        smooth_path.append(path[next_valid_idx])
        current_idx = next_valid_idx
        
    return np.array(smooth_path)

def test_all_scenarios():
    # Liste des fichiers de scénarios
    # Assurez-vous que le dossier 'scenario' existe et contient ces fichiers
>>>>>>> 4dc95b06e77a70ee10f9bf111c5347f6555af549
    scenarios = ["scenario0.txt", "scenario1.txt", "scenario2.txt", "scenario3.txt", "scenario4.txt"]
    
    # Vérification du dossier
    if not os.path.exists("scenario"):
        print("Erreur : Dossier 'scenario' introuvable.")
        return

<<<<<<< HEAD
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
=======
    for filename in scenarios[4:5]:  # Test du scénario 4
>>>>>>> 4dc95b06e77a70ee10f9bf111c5347f6555af549
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

<<<<<<< HEAD
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
=======
        print(f"Environnement : {env.width}x{env.height}")
        print(f"Obstacles : {len(env.obstacles)}")

        # --- ALGO 1 : PSO ---
        print("\n--- Lancement PSO ---")
        pso_planner = PSO_PathPlanner(
            env=env, 
            start=start_pos, 
            goal=goal_pos, 
            num_particles=200,
            num_waypoints=20,  # Ajustable selon complexité
            max_iter=200
        )

        t0 = time.time()
        path_pso, score_pso, history = pso_planner.optimize(random_restart=True, simulated_annealing=False, dimensional_learning=True)
        path_pso = smooth_path(env, path_pso)  # On lisse le chemin pour la visualisation
        score_pso = pso_planner.evaluate_position(path_pso)  # Score final du chemin lissé
        dt_pso = time.time() - t0
        print(f"PSO terminé en {dt_pso:.4f}s | Score: {score_pso:.1f}")

        # --- ALGO 2 : RRT ---
        print("\n--- Lancement RRT ---")
        # Paramètres RRT : delta_s (pas), delta_r (rayon rewiring)
        rrt_planner = RRT(
            env=env,
            start=start_pos,
            goal=goal_pos,
            delta_s=50.0, 
            delta_r=150.0,
            max_iter=1
        )
>>>>>>> 4dc95b06e77a70ee10f9bf111c5347f6555af549

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

<<<<<<< HEAD
        # Plot RRT
        env.plot(ax_rrt, path=path_rrt, color='red')
        title_str = f"{status_txt} | Coût: {cost_rrt:.1f} | Temps: {dt_rrt:.3f}s"
        ax_rrt.set_title(title_str, fontsize=10, backgroundcolor=status_color)
=======
        # --- VISUALISATION CÔTE À CÔTE ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: PSO
        env.plot(ax1, path=path_pso, title=f"PSO ({dt_pso:.2f}s) - Score: {score_pso:.0f}")
        # Tracer l'historique des chemins explorés (en vert clair)
        for i in range(0, len(history), 5):
            raw_waypoints = history[i]
            
            # 1. On remet en forme (N points, 2 coordonnées)
            waypoints = raw_waypoints.reshape((-1, 2))
            
            # 2. On colle le Départ au début et l'Arrivée à la fin
            # (Exactement comme dans la méthode get_full_path de la particule)
            full_path_history = np.vstack([start_pos, waypoints, goal_pos])
            
            # 3. On trace
            # alpha=0.1 est très important : cela rend les traits transparents
            # pour voir les zones où l'algorithme a beaucoup cherché.
            ax1.plot(full_path_history[:, 0], full_path_history[:, 1], 
                     color='green', linewidth=1, alpha=0.1)

        # Affichage du chemin FINAL (en bleu ou noir, bien visible) par dessus
        if path_pso is not None:
             ax1.plot(path_pso[:, 0], path_pso[:, 1], color='blue', linewidth=3, label='Best Path')

        ax1.scatter(start_pos[0], start_pos[1], c='green', s=100, label='Start')
        ax1.scatter(goal_pos[0], goal_pos[1], c='red', marker='*', s=150, label='Goal')
        ax1.legend()
>>>>>>> 4dc95b06e77a70ee10f9bf111c5347f6555af549

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