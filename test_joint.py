import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import os
import time

# Import des modules existants
from geometry import Environment
from joint_rrt import JointRRT

def parse_scenario(filepath):
    """Lecture du scénario (identique à test2.py)"""
    if not os.path.exists(filepath):
        print(f"Erreur : Le fichier {filepath} n'existe pas.")
        return None

    try:
        with open(filepath, 'r') as f:
            content = f.read().replace('\n', ' ').split()
            data = [float(x) for x in content]
        
        idx = 0
        xmax, ymax = data[idx], data[idx+1]; idx += 2
        s1 = np.array([data[idx], data[idx+1]]); idx += 2
        g1 = np.array([data[idx], data[idx+1]]); idx += 2
        s2 = np.array([data[idx], data[idx+1]]); idx += 2
        g2 = np.array([data[idx], data[idx+1]]); idx += 2
        R = data[idx]; idx += 1
        
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

def get_position_at_time(t, path, cum_dist, speed, start_delay=0.0):
    """
    Interpole la position d'un robot sur son chemin à un instant t donné.
    """
    # Si le temps est inférieur au délai de départ, le robot est au début
    if t < start_delay:
        return path[0]
    
    # Distance théorique parcourue
    dist_travelled = (t - start_delay) * speed
    
    # Si on a fini le chemin, on reste à la fin (np.interp le fait naturellement, 
    # mais on s'assure que c'est bien géré)
    if dist_travelled > cum_dist[-1]:
        return path[-1]
    
    x = np.interp(dist_travelled, cum_dist, path[:, 0])
    y = np.interp(dist_travelled, cum_dist, path[:, 1])
    return np.array([x, y])

def run_animation():
    # --- 1. CONFIGURATION ---
    filename = "scenario/scenario2.txt"  # CHOISISSEZ VOTRE SCENARIO ICI
    SPEED = 50.0  # Vitesse d'animation (unités/sec)
    
    # Fallback si fichier manquant
    if not os.path.exists(filename):
        print(f"Fichier {filename} introuvable. Création env par défaut.")
        env = Environment(100, 100, [(40, 20, 20, 60)])
        s1, g1 = np.array([10, 50]), np.array([90, 50])
        s2, g2 = np.array([50, 10]), np.array([50, 90])
        R = 15.0
    else:
        data = parse_scenario(filename)
        if not data: return
        env, s1, g1, s2, g2, R = data

    print(f"--- Planification Conjointe ---")
    planner = JointRRT(SPEED, s1, g1, s2, g2, R, env, max_iter=2000)
    
    # Calcul des chemins et du délai
    t0_calc = time.time()
    try:
        res = planner.joint_plan(h=0.1)
    except Exception as e:
        print(f"Erreur code : {e}")
        return

    if res is None:
        print("Échec de la planification.")
        return

    path1, path2, t_2 = res
    print(f"Calcul terminé en {time.time()-t0_calc:.2f}s")
    print(f"Délai imposé au Robot 2 : {t_2:.2f} secondes")

    # --- 2. PRÉPARATION DES DONNÉES POUR L'ANIMATION ---
    
    # Pré-calcul des distances cumulées pour l'interpolation rapide
    dists1 = np.linalg.norm(path1[1:] - path1[:-1], axis=1)
    cum_dist1 = np.insert(np.cumsum(dists1), 0, 0.0)
    
    dists2 = np.linalg.norm(path2[1:] - path2[:-1], axis=1)
    cum_dist2 = np.insert(np.cumsum(dists2), 0, 0.0)
    
    # Durée totale de l'animation
    total_time = max(cum_dist1[-1]/SPEED, t_2 + cum_dist2[-1]/SPEED) + 1.0
    
    # --- 3. CONFIGURATION GRAPHIQUE ---
    fig, ax = plt.subplots(figsize=(8, 8))
    env.plot(ax, title=f"Animation Joint RRT (R2 Delay: {t_2:.1f}s)")
    
    # Tracer les chemins complets en pointillés (pour référence)
    ax.plot(path1[:, 0], path1[:, 1], 'r--', alpha=0.3, label='Trajet R1')
    ax.plot(path2[:, 0], path2[:, 1], 'b--', alpha=0.3, label='Trajet R2')
    
    # Objets mobiles (Robots)
    # Robot 1 (Rouge)
    robot1_dot, = ax.plot([], [], 'ro', ms=8, zorder=5)
    robot1_safe = plt.Circle((0, 0), R, color='red', fill=True, alpha=0.1)
    ax.add_patch(robot1_safe)
    
    # Robot 2 (Bleu)
    robot2_dot, = ax.plot([], [], 'bo', ms=8, zorder=5)
    robot2_safe = plt.Circle((0, 0), R, color='blue', fill=True, alpha=0.1)
    ax.add_patch(robot2_safe)
    
    # Texte d'information (Chronomètre)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.8))

    # --- 4. FONCTION D'ANIMATION ---
    dt = 0.05 # Pas de temps de l'animation (fluidité)
    
    def init():
        robot1_dot.set_data([], [])
        robot2_dot.set_data([], [])
        robot1_safe.center = (-100, -100) # Hors champ
        robot2_safe.center = (-100, -100)
        time_text.set_text('')
        return robot1_dot, robot2_dot, robot1_safe, robot2_safe, time_text

    def update(frame):
        current_time = frame * dt
        
        # Position Robot 1 (Départ t=0)
        pos1 = get_position_at_time(current_time, path1, cum_dist1, SPEED, start_delay=0.0)
        robot1_dot.set_data([pos1[0]], [pos1[1]])
        robot1_safe.center = (pos1[0], pos1[1])
        
        # Position Robot 2 (Départ t=t_2)
        pos2 = get_position_at_time(current_time, path2, cum_dist2, SPEED, start_delay=t_2)
        robot2_dot.set_data([pos2[0]], [pos2[1]])
        robot2_safe.center = (pos2[0], pos2[1])
        
        # Détection collision visuelle (Feedback couleur)
        dist = np.linalg.norm(pos1 - pos2)
        if dist < R:
            time_text.set_color('red')
            time_text.set_text(f"Time: {current_time:.2f}s | COLLISION! ({dist:.1f} < {R})")
            # Rendre les cercles rouges vifs pour montrer l'erreur
            robot1_safe.set_color('red')
            robot2_safe.set_color('red')
            robot1_safe.set_alpha(0.5)
            robot2_safe.set_alpha(0.5)
        else:
            time_text.set_color('black')
            status = "Waiting..." if current_time < t_2 else "Moving"
            time_text.set_text(f"Time: {current_time:.2f}s | R2: {status}")
            # Rétablir couleurs normales
            robot1_safe.set_color('red')
            robot2_safe.set_color('blue')
            robot1_safe.set_alpha(0.1)
            robot2_safe.set_alpha(0.1)
            
        return robot1_dot, robot2_dot, robot1_safe, robot2_safe, time_text

    # Création de l'animation
    frames = int(total_time / dt)
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, 
                         interval=dt*1000, blit=True, repeat=True)
    
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    run_animation()