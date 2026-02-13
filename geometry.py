import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def lines_intersect(p1, p2, p3, p4):
    """
    Vérifie si le segment [p1, p2] coupe le segment [p3, p4].
    Retourne True si intersection, False sinon.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return False  # Parallèles

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    # On utilise une petite tolérance pour les cas limites
    return 0 <= ua <= 1 and 0 <= ub <= 1

class Environment:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles # List of (x, y, w, h)

    def is_inside(self, point):
        x, y = point
        return 0 <= x <= self.width and 0 <= y <= self.height

    def check_collision_point(self, point):
        if not self.is_inside(point):
            return True
        x, y = point
        for (ox, oy, w, h) in self.obstacles:
            if ox <= x <= ox + w and oy <= y <= oy + h:
                return True
        return False

    def check_collision_segment(self, p1, p2):
        """
        Vérification continue (mathématique) de collision.
        Teste l'intersection du mouvement avec les 4 bords de chaque obstacle.
        """
        # 1. Vérifier si les points extrêmes sont dans un obstacle
        if self.check_collision_point(p1) or self.check_collision_point(p2):
            return True

        # 2. Vérifier l'intersection avec les bords de chaque obstacle
        for (ox, oy, w, h) in self.obstacles:
            # Les 4 coins du rectangle
            c1 = np.array([ox, oy])
            c2 = np.array([ox + w, oy])
            c3 = np.array([ox + w, oy + h])
            c4 = np.array([ox, oy + h])

            # Les 4 segments du rectangle (bas, droite, haut, gauche)
            edges = [
                (c1, c2),
                (c2, c3),
                (c3, c4),
                (c4, c1)
            ]

            for edge_start, edge_end in edges:
                if lines_intersect(p1, p2, edge_start, edge_end):
                    return True
        
        return False

    def plot(self, ax, path=None, color='blue', title="Environment"):
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        # Dessin des obstacles
        for (ox, oy, w, h) in self.obstacles:
            rect = patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='gray')
            ax.add_patch(rect)
        
        # Dessin du chemin
        if path is not None and len(path) > 0:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2, label='Path')
        
        ax.set_title(title)
        ax.set_aspect('equal')

def parse_input_file(filepath):
    """
    Lit le format spécifique des fichiers scénarios INF421.
    Gère les retours à la ligne et la notation scientifique.
    """
    with open(filepath, 'r') as f:
        # Lecture de tout le contenu, remplacement des retours ligne par des espaces
        # split() découpe ensuite par espaces
        content = f.read().replace('\n', ' ').split()
        # Conversion en float (gère automatiquement le format 1.00e+03)
        data = [float(x) for x in content]
        
    idx = 0
    try:
        # Dimensions
        xmax = data[idx]; idx += 1
        ymax = data[idx]; idx += 1
        
        # Robot 1
        start_x = data[idx]; idx += 1
        start_y = data[idx]; idx += 1
        goal_x = data[idx]; idx += 1
        goal_y = data[idx]; idx += 1
        
        # Robot 2 (on le lit mais on ne l'utilise pas forcément pour le PSO simple)
        start2_x = data[idx]; idx += 1
        start2_y = data[idx]; idx += 1
        goal2_x = data[idx]; idx += 1
        goal2_y = data[idx]; idx += 1
        
        # Rayon de sécurité
        R = data[idx]; idx += 1
        
        # Obstacles (par groupes de 4 : x, y, w, h)
        obstacles = []
        while idx < len(data):
            # Sécurité pour ne pas planter si le fichier a des lignes vides à la fin
            if idx + 3 >= len(data): 
                break
            ox = data[idx]; idx += 1
            oy = data[idx]; idx += 1
            lx = data[idx]; idx += 1
            ly = data[idx]; idx += 1
            obstacles.append((ox, oy, lx, ly))
            
        env = Environment(xmax, ymax, obstacles)
        
        # Retourne des tableaux numpy pour être compatible avec pso.py
        return env, np.array([start_x, start_y]), np.array([goal_x, goal_y])
        
    except IndexError:
        print(f"Erreur de lecture du fichier {filepath}: format incorrect ou incomplet.")
        return None, None, None