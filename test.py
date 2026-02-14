import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import des modules
from geometry import parse_input_file
from pso import PSO_PathPlanner
from rrt import RRT

def test_all_scenarios():
    # Liste des fichiers de scénarios
    # Assurez-vous que le dossier 'scenario' existe et contient ces fichiers
    scenarios = ["scenario0.txt", "scenario1.txt", "scenario2.txt", "scenario3.txt", "scenario4.txt"]
    
    # Vérification du dossier
    if not os.path.exists("scenario"):
        print("Erreur : Le dossier 'scenario' n'existe pas.")
        # On essaie de lister le dossier courant pour aider
        print(f"Dossier courant : {os.getcwd()}")
        return

    for filename in scenarios:
        filepath = os.path.join("scenario", filename)
        
        print(f"\n{'='*60}")
        print(f"TEST DU SCENARIO : {filename}")
        print(f"{'='*60}")
        
        if not os.path.exists(filepath):
            print(f"Fichier {filepath} introuvable. Passage au suivant.")
            continue

        # 1. Chargement du scénario
        # parse_input_file retourne : env, start1, goal1, start2, goal2, Radius
        # On utilise *rest pour ignorer les données du robot 2 pour l'instant
        try:
            env, start_pos, goal_pos, *rest = parse_input_file(filepath)
        except Exception as e:
            print(f"Erreur lors du parsing de {filename}: {e}")
            continue

        print(f"Environnement : {env.width}x{env.height}")
        print(f"Obstacles : {len(env.obstacles)}")

        # --- ALGO 1 : PSO ---
        print("\n--- Lancement PSO ---")
        pso_planner = PSO_PathPlanner(
            env=env, 
            start=start_pos, 
            goal=goal_pos, 
            num_particles=100,
            num_waypoints=10,  # Ajustable selon complexité
            max_iter=150
        )

        t0 = time.time()
        path_pso, score_pso = pso_planner.optimize(random_restart=True)
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
            max_iter=3000
        )

        t0 = time.time()
        path_rrt = rrt_planner.plan(intelligent_sampling=True, triang_opt=True) # On active le sampling intelligent
        dt_rrt = time.time() - t0
        
        if path_rrt is not None:
            print(f"RRT terminé en {dt_rrt:.4f}s | Chemin trouvé (longueur: {len(path_rrt)} noeuds)")
        else:
            print(f"RRT terminé en {dt_rrt:.4f}s | ECHEC (Pas de chemin)")


        # --- VISUALISATION CÔTE À CÔTE ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: PSO
        env.plot(ax1, path=path_pso, title=f"PSO ({dt_pso:.2f}s) - Score: {score_pso:.0f}")
        ax1.scatter(start_pos[0], start_pos[1], c='green', s=100, label='Start')
        ax1.scatter(goal_pos[0], goal_pos[1], c='red', marker='*', s=150, label='Goal')
        ax1.legend()

        # Plot 2: RRT
        title_rrt = f"RRT ({dt_rrt:.2f}s) - " + ("Succès" if path_rrt is not None else "Échec")
        env.plot(ax2, path=path_rrt, color='red', title=title_rrt)
        
        # Dessiner l'arbre RRT (optionnel mais instructif)
        # On parcourt tous les noeuds explorés pour voir l'arbre
        if hasattr(rrt_planner, 'nodes'):
            for node in rrt_planner.nodes:
                if node.parent is not None:
                    p1 = node.pos
                    p2 = node.parent.pos
                    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', alpha=0.15, linewidth=1)

        ax2.scatter(start_pos[0], start_pos[1], c='green', s=100)
        ax2.scatter(goal_pos[0], goal_pos[1], c='red', marker='*', s=150)
        
        plt.suptitle(f"Comparaison Scénario : {filename}", fontsize=16)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    test_all_scenarios()