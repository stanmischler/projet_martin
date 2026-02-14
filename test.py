import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import des modules
from geometry import parse_input_file
from pso import PSO_PathPlanner
from rrt import RRT

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
    scenarios = ["scenario0.txt", "scenario1.txt", "scenario2.txt", "scenario3.txt", "scenario4.txt"]
    
    # Vérification du dossier
    if not os.path.exists("scenario"):
        print("Erreur : Le dossier 'scenario' n'existe pas.")
        # On essaie de lister le dossier courant pour aider
        print(f"Dossier courant : {os.getcwd()}")
        return

    for filename in scenarios[4:5]:  # Test du scénario 4
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