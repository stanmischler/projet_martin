import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Import de vos modules
from geometry import Environment
from joint_rrt import JointRRT  # Assurez-vous que le fichier s'appelle bien joint_rrt.py

def parse_scenario(filepath):
    """
    Fonction utilitaire pour lire les fichiers scénarios.
    Retourne l'environnement et les données des deux robots.
    """
    if not os.path.exists(filepath):
        print(f"Erreur : Le fichier {filepath} n'existe pas.")
        return None

    try:
        with open(filepath, 'r') as f:
            content = f.read().replace('\n', ' ').split()
            data = [float(x) for x in content]
        
        idx = 0
        # Dimensions
        xmax, ymax = data[idx], data[idx+1]; idx += 2
        
        # Robot 1
        s1 = np.array([data[idx], data[idx+1]]); idx += 2
        g1 = np.array([data[idx], data[idx+1]]); idx += 2
        
        # Robot 2
        s2 = np.array([data[idx], data[idx+1]]); idx += 2
        g2 = np.array([data[idx], data[idx+1]]); idx += 2
        
        # Rayon de sécurité
        R = data[idx]; idx += 1
        
        # Obstacles
        obstacles = []
        while idx < len(data) - 3:
            ox, oy, lx, ly = data[idx:idx+4]
            obstacles.append((ox, oy, lx, ly))
            idx += 4
            
        env = Environment(xmax, ymax, obstacles)
        return env, s1, g1, s2, g2, R

    except Exception as e:
        print(f"Erreur de parsing : {e}")
        return None

def test_joint_planner():
    # 1. Configuration
    filename = "scenario/scenario1.txt" # Changez ceci selon vos fichiers
    
    # Création d'un environnement de secours si le fichier n'existe pas
    if not os.path.exists(filename):
        print(f"Attention: {filename} introuvable. Utilisation d'un environnement par défaut.")
        env = Environment(100, 100, [(40, 0, 20, 60)]) # Un mur au milieu
        s1, g1 = np.array([10, 50]), np.array([90, 50])
        s2, g2 = np.array([50, 10]), np.array([50, 90])
        R = 10.0
    else:
        data = parse_scenario(filename)
        if not data: return
        env, s1, g1, s2, g2, R = data

    print(f"--- Lancement du Test JointRRT ---")
    print(f"Robot 1: {s1} -> {g1}")
    print(f"Robot 2: {s2} -> {g2}")
    print(f"Distance de sécurité: {R}")

    # 2. Instanciation de JointRRT
    # Vitesse arbitraire de 10.0 unités/s
    joint_planner = JointRRT(
        speed=10.0,
        start_1=s1, goal_1=g1,
        start_2=s2, goal_2=g2,
        safe_distance=R,
        env=env,
        delta_s=10,   # Pas RRT
        max_iter=2000
    )

    # 3. Exécution de la planification
    t0 = time.time()
    try:
        # Note: votre méthode joint_plan appelle RRT.plan()
        result = joint_planner.joint_plan(h=0.1)
    except Exception as e:
        print(f"Erreur lors de l'exécution : {e}")
        import traceback
        traceback.print_exc()
        return

    dt = time.time() - t0

    # 4. Analyse des résultats
    if result is None:
        print("ÉCHEC : Aucun chemin trouvé par les RRTs individuels.")
        return
    
    # Extraction des données retournées par votre méthode joint_plan
    # Votre signature retourne : path1, path2, t_2
    path1, path2, t_2 = result

    # Petite vérification car np.array(None) n'est pas None (voir note en bas)
    if path1.shape == () or path2.shape == ():
         print("ÉCHEC : L'un des chemins est vide (None).")
         return

    print(f"\n--- RÉSULTATS ({dt:.4f}s) ---")
    print(f"Chemin Robot 1 : {len(path1)} points")
    print(f"Chemin Robot 2 : {len(path2)} points")
    print(f"Délai calculé (t_2) : {t_2:.2f} secondes")
    
    if t_2 > 0:
        print("=> CONFLIT DÉTECTÉ : Le Robot 2 doit attendre pour éviter le Robot 1.")
    else:
        print("=> TRAJET LIBRE : Les robots peuvent partir en même temps.")

    # 5. Visualisation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Utilisation de la méthode plot de votre classe Environment
    env.plot(ax, title=f"Planification Conjointe (Retard R2 = {t_2:.2f}s)")
    
    # Tracé des chemins
    ax.plot(path1[:, 0], path1[:, 1], 'r-', linewidth=2, label='Robot 1 (Prioritaire)')
    ax.plot(path2[:, 0], path2[:, 1], 'b--', linewidth=2, label=f'Robot 2 (Attend {t_2}s)')
    
    # Points de départ et d'arrivée
    ax.scatter(s1[0], s1[1], c='red', marker='o', s=100, zorder=5)
    ax.scatter(g1[0], g1[1], c='red', marker='x', s=100, zorder=5)
    ax.text(s1[0], s1[1], " S1", color='red', fontweight='bold')
    
    ax.scatter(s2[0], s2[1], c='blue', marker='o', s=100, zorder=5)
    ax.scatter(g2[0], g2[1], c='blue', marker='x', s=100, zorder=5)
    ax.text(s2[0], s2[1], " S2", color='blue', fontweight='bold')

    # Visualisation de la zone de sécurité au croisement (optionnel, juste au départ)
    start_circle1 = plt.Circle(s1, R, color='r', fill=False, alpha=0.3, linestyle=':')
    ax.add_patch(start_circle1)

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_joint_planner()