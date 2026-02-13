import numpy as np
from geometry import Environment

class Node:
    def __init__(self, pos, parent=None):
        self.pos = pos
        self.parent = parent
        self.cost = 0.0 # Cost from root

class RRT:
    def __init__(self, env, start, goal, delta_s=50, delta_r=100, max_iter=1000):
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

    def plan(self, intelligent_sampling=False):
        for i in range(self.max_iter):
            # Sampling
            if intelligent_sampling and np.random.rand() < 0.3:
                # Sample near obstacles (simplified implementation)
                obs = self.env.obstacles[np.random.randint(len(self.env.obstacles))]
                ox, oy, w, h = obs
                rand_pt = np.array([ox - 10 + np.random.rand()*(w+20), oy - 10 + np.random.rand()*(h+20)])
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
                    if self.env.check_collision_segment(nb.pos, new_node.pos):
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
                             
            # Check goal
            if np.linalg.norm(new_node.pos - self.goal) < self.delta_s:
                if not self.env.check_collision_segment(new_node.pos, self.goal):
                    goal_node = Node(self.goal, new_node)
                    goal_node.cost = new_node.cost + np.linalg.norm(new_node.pos - self.goal)
                    return self.extract_path(goal_node)
                    
        return None # No path found

    def extract_path(self, node):
        path = []
        curr = node
        while curr is not None:
            path.append(curr.pos)
            curr = curr.parent
        return path[::-1] # Reverse