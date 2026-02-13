import numpy as np
import matplotlib.pyplot as plt
import time

# Import des modules créés précédemment
from geometry import Environment
from pso import PSO_PathPlanner

def run_pso_test():
    print("=== TEST UNITAIRE : PSO PATH PLANNING ===")

    # 1. Configuration de l'environnement (similaire au projet : rectangle + obstacles)
    # On crée une carte 100x100 avec des obstacles simples
    width, height = 100, 100
    
    # [cite_start]Obstacles format: (x, y, largeur, hauteur) [cite: 14]
    # On place un mur au milieu pour forcer le contournement
    obstacles = [
        (40, 0, 20, 70),   # Un long mur vertical qui bloque le passage direct
        (20, 80, 60, 10),  # Un bloc en haut
    ]
    
    env = Environment(width, height, obstacles)
    
    # 2. Définition Départ et Arrivée
    start_pos = np.array([10.0, 50.0])  # Gauche
    goal_pos = np.array([90.0, 50.0])   # Droite (derrière le mur)

    print(f"Environnement : {width}x{height}")
    print(f"Départ : {start_pos}, Arrivée : {goal_pos}")
    print(f"Nombre d'obstacles : {len(obstacles)}")

    # [cite_start]3. Paramètres du PSO [cite: 43]
    # S (particules), c1, c2, w sont définis par défaut dans la classe, 
    # mais on peut ajuster le nombre de waypoints et d'itérations.
    num_particles = 100
    num_waypoints = 5   # Nombre de points intermédiaires (complexité du chemin)
    max_iter = 150

    print(f"\nLancement du PSO avec {num_particles} particules et {max_iter} itérations...")
    
    planner = PSO_PathPlanner(
        env=env, 
        start=start_pos, 
        goal=goal_pos, 
        num_particles=num_particles,
        num_waypoints=num_waypoints, 
        max_iter=max_iter
    )

    # [cite_start]4. Exécution et Mesure du temps [cite: 49]
    t0 = time.time()
    best_path, best_score = planner.optimize()
    dt = time.time() - t0

    print(f"Terminé en {dt:.4f} secondes.")
    print(f"Meilleur score (Fitness) : {best_score:.4f}")
    
    # Si le score est très élevé, c'est qu'il y a collision (pénalité dans pso.py)
    if best_score > 1000: 
        print("/!\\ ATTENTION : Le chemin final contient des collisions ou est invalide.")
    else:
        print("Succès : Chemin valide trouvé.")

    # [cite_start]5. Visualisation [cite: 25]
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # On utilise la méthode plot de l'environnement
    env.plot(ax, path=best_path, title=f"Résultat PSO (Fitness: {best_score:.1f})")
    
    # Ajout des marqueurs spécifiques pour Start/Goal
    ax.scatter(start_pos[0], start_pos[1], c='green', s=150, marker='o', label='Départ', zorder=5)
    ax.scatter(goal_pos[0], goal_pos[1], c='red', s=150, marker='*', label='Arrivée', zorder=5)
    ax.legend()
    
    plt.grid(True, linestyle='--', alpha=0.5)
    print("Affichage du graphique...")
    plt.show()

if __name__ == "__main__":
    run_pso_test()