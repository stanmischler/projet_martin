import numpy as np
from geometry import Environment

class Node:
    def __init__(self, pos, parent=None):
        self.pos = pos
        self.parent = parent
        self.cost = 0.0 # Cost from root

class RRT:
    def __init__(self, env, start, goal, delta_s=50, delta_r=100, max_iter=3000):
        self.env = env
        self.start = Node(start)
        self.goal = goal
        self.nodes = [self.start]
        self.delta_s = delta_s # Step size
        self.delta_r = delta_r # Rewiring radius
        self.max_iter = max_iter

    def get_nearest_node(self, point):
        # Linear search (simple implementation)
        # KD-Tree would be O(log n)
        dists = [np.linalg.norm(node.pos - point) for node in self.nodes]
        min_idx = np.argmin(dists)
        return self.nodes[min_idx]

    def steer(self, from_node, to_point):
        """Moves from from_node towards to_point by max delta_s"""
        vec = to_point - from_node.pos
        dist = np.linalg.norm(vec)
        if dist <= self.delta_s:
            return Node(to_point, from_node)
        
        # Normalize and scale
        vec = vec / dist * self.delta_s
        new_pos = from_node.pos + vec
        return Node(new_pos, from_node)

    def triang_opt(self,node):
        parent = node.parent
        grand_parent = parent.parent
        if grand_parent is not None:
            if not self.env.check_collision_segment(grand_parent.pos, node.pos):
                node.parent = grand_parent
                node.cost = grand_parent.cost + np.linalg.norm(grand_parent.pos - node.pos)

    def plan(self, intelligent_sampling=False, triang_opt=False):
        for i in range(self.max_iter):
            # Sampling
            if intelligent_sampling and np.random.rand() < 0.3:
                # Echantillonnage proche des obstacles (Corridors uniquement)
                obs = self.env.obstacles[np.random.randint(len(self.env.obstacles))]
                ox, oy, w, h = obs
                margin = 20.0  # Largeur du couloir autour de l'obstacle
                
                # On choisit aléatoirement un des 4 côtés (0:Gauche, 1:Droite, 2:Bas, 3:Haut)
                side = np.random.randint(0, 4)
                
                if side == 0:   # Gauche
                    # x entre [ox - margin, ox]
                    rand_x = ox - np.random.rand() * margin
                    # y étendu pour couvrir les coins
                    rand_y = (oy - margin) + np.random.rand() * (h + 2*margin)
                    
                elif side == 1: # Droite
                    # x entre [ox + w, ox + w + margin]
                    rand_x = ox + w + np.random.rand() * margin
                    rand_y = (oy - margin) + np.random.rand() * (h + 2*margin)
                    
                elif side == 2: # Bas
                    # x étendu
                    rand_x = (ox - margin) + np.random.rand() * (w + 2*margin)
                    # y entre [oy - margin, oy]
                    rand_y = oy - np.random.rand() * margin
                    
                else:           # Haut (side == 3)
                    # x étendu
                    rand_x = (ox - margin) + np.random.rand() * (w + 2*margin)
                    # y entre [oy + h, oy + h + margin]
                    rand_y = oy + h + np.random.rand() * margin

                rand_pt = np.array([rand_x, rand_y])
            else:
                rand_pt = np.array([np.random.rand()*self.env.width, np.random.rand()*self.env.height])
                
            nearest = self.get_nearest_node(rand_pt)
            new_node = self.steer(nearest, rand_pt)
            
            if not self.env.check_collision_segment(nearest.pos, new_node.pos):
                new_node.cost = nearest.cost + np.linalg.norm(new_node.pos - nearest.pos)
                
                # REWIRING (RRT*)
                # Find neighbors within delta_r
                neighbors = []
                for node in self.nodes:
                    if np.linalg.norm(node.pos - new_node.pos) < self.delta_r:
                        neighbors.append(node)
                
                # 1. Choose best parent
                for nb in neighbors:
                    if not self.env.check_collision_segment(nb.pos, new_node.pos):
                        cost = nb.cost + np.linalg.norm(nb.pos - new_node.pos)
                        if cost < new_node.cost:
                            new_node.parent = nb
                            new_node.cost = cost
                
                self.nodes.append(new_node)
                
                # 2. Rewire neighbors
                for nb in neighbors:
                     if nb == new_node.parent: continue
                     dist = np.linalg.norm(new_node.pos - nb.pos)
                     if new_node.cost + dist < nb.cost:
                         if not self.env.check_collision_segment(new_node.pos, nb.pos):
                             nb.parent = new_node
                             nb.cost = new_node.cost + dist

                if triang_opt:
                    self.triang_opt(new_node)
                             
                # Check goal
                if np.linalg.norm(new_node.pos - self.goal) < 200:
                    if not self.env.check_collision_segment(new_node.pos, self.goal):
                        goal_node = Node(self.goal, new_node)
                        goal_node.cost = new_node.cost + np.linalg.norm(new_node.pos - self.goal)

                        if triang_opt:
                            return self.triang_opt_path(self.extract_path(goal_node))
                        return self.extract_path(goal_node)
                    
        return None # No path found

    def extract_path(self, node):
        path = []
        curr = node
        while curr is not None:
            path.append(curr.pos)
            curr = curr.parent
        return path[::-1] # Reverse

    def length(self, path):
        return sum(np.linalg.norm(path[i] - path[i+1]) for i in range(len(path)-1))

    def triang_opt_path(self, path):
        if len(path) < 3:
            return path
        
        optimized_path = list(path)
        i = 0
        while i < len(optimized_path) - 2:
            if not self.env.check_collision_segment(optimized_path[i], optimized_path[i+2]):
                optimized_path.pop(i+1)
            else:
                i += 1
                
        return optimized_path
        