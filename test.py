import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import des modules
from geometry import parse_input_file
from pso import PSO_PathPlanner

def test_all_scenarios():
    # Liste des fichiers de scénarios (assurez-vous qu'ils sont dans le même dossier)
    scenarios = ["scenario0.txt", "scenario1.txt", "scenario2.txt", "scenario3.txt", "scenario4.txt"]
    
    for filename in scenarios:
        print(f"\n{'='*40}")
        print(f"TEST DU SCENARIO : {filename}")
        print(f"{'='*40}")
        
        if not os.path.exists("scenario/" + filename):
            print(f"Fichier {filename} introuvable. Passage au suivant.")
            continue

        # 1. Chargement du scénario via le parser mis à jour
        env, start_pos, goal_pos = parse_input_file("scenario/" + filename)
        
        if env is None:
            continue

        print(f"Environnement : {env.width}x{env.height}")
        print(f"Départ : {start_pos}")
        print(f"Arrivée : {goal_pos}")
        print(f"Nombre d'obstacles : {len(env.obstacles)}")

        # 2. Configuration du PSO
        # On augmente un peu les itérations car certains scénarios sont complexes
        num_particles = 100
        num_waypoints = 6   # Assez de flexibilité pour contourner
        max_iter = 200

        planner = PSO_PathPlanner(
            env=env, 
            start=start_pos, 
            goal=goal_pos, 
            num_particles=num_particles,
            num_waypoints=num_waypoints, 
            max_iter=max_iter
        )

        # 3. Exécution
        t0 = time.time()
        best_path, best_score = planner.optimize()
        dt = time.time() - t0

        print(f"Terminé en {dt:.4f} secondes.")
        print(f"Score final : {best_score:.4f}")
        
        if best_score > 10000:
            print(">>> ECHEC : Collision détectée ou chemin non trouvé.")
        else:
            print(">>> SUCCES : Chemin valide trouvé.")

        # 4. Affichage
        fig, ax = plt.subplots(figsize=(8, 8))
        env.plot(ax, path=best_path, title=f"{filename} - Score: {best_score:.1f}")
        
        # Marqueurs Start/Goal
        ax.scatter(start_pos[0], start_pos[1], c='green', s=100, label='Start', zorder=5)
        ax.scatter(goal_pos[0], goal_pos[1], c='red', marker='*', s=150, label='Goal', zorder=5)
        ax.legend()
        
        plt.show() # Bloque l'exécution jusqu'à la fermeture de la fenêtre

if __name__ == "__main__":
    test_all_scenarios()