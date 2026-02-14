import numpy as np
from geometry import Environment

class Particle:
    def __init__(self, num_waypoints, start, goal, env: Environment):
        self.start = start
        self.goal = goal
        self.num_waypoints = num_waypoints
        
        # Position: flatten array of waypoints [x1, y1, x2, y2, ...]
        # Initialize randomly within bounds
        self.position = np.random.rand(num_waypoints * 2)
        for i in range(num_waypoints):
            self.position[2*i] *= env.width
            self.position[2*i+1] *= env.height
            
        self.velocity = np.zeros_like(self.position)
        self.p_best = self.position.copy()
        self.p_best_score = float('inf')
        self.env = env

    def evaluate(self):
        """Fitness function: Length of path + Penalty for collision"""
        path = self.get_full_path()
        length = 0
        penalty = 0
        
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            dist = np.linalg.norm(p2 - p1)
            length += dist
            if self.env.check_collision_segment(p1, p2):
                penalty += 10000 # Large penalty per collision
                
        return length + penalty

    def get_full_path(self):
        waypoints = self.position.reshape((-1, 2))
        return np.vstack([self.start, waypoints, self.goal])



class PSO_PathPlanner:
    def __init__(self, env, start, goal, num_particles=50, num_waypoints=5, max_iter=100):
        self.env = env
        self.start = start
        self.goal = goal
        self.num_particles = num_particles
        self.num_waypoints = num_waypoints
        self.max_iter = max_iter
        self.particles = [Particle(num_waypoints, start, goal, env) for _ in range(num_particles)]
        self.g_best = self.particles[0].position.copy()
        self.g_best_score = float('inf')
        
        # Hyperparameters
        self.w = 0.5  # Inertia
        self.c1 = 0.5 # Cognitive (local)
        self.c2 = 2 # Social (global)
    
    def evaluate_position(self, position):
        waypoints = position.reshape((-1, 2))
        path = np.vstack([self.start, waypoints, self.goal])
        length = 0
        penalty = 0
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            dist = np.linalg.norm(p2 - p1)
            length += dist
            if self.env.check_collision_segment(p1, p2):
                penalty += 10000 # Large penalty per collision
        return length + penalty

    def optimize(self, random_restart=False):
        for k in range(self.max_iter):
            for p in self.particles:
                score = p.evaluate()
                
                # Update local best
                if score < p.p_best_score:
                    p.p_best_score = score
                    p.p_best = p.position.copy()
                
                # Update global best
                if score < self.g_best_score:
                    self.g_best_score = score
                    self.g_best = p.position.copy()
            
            # Move particles
            for p in self.particles:
                if random_restart and np.random.rand() < 0.1:
                    positions=np.random.rand(self.num_waypoints * 2)
                    for i in range(self.num_waypoints):
                        positions[2*i] *= self.env.width
                        positions[2*i+1] *= self.env.height
                    p.position = positions
                    p.velocity = np.zeros_like(positions)

                else:
                    r1 = np.random.rand(len(p.position))
                    r2 = np.random.rand(len(p.position))
                    
                    p.velocity = (self.w * p.velocity) + \
                                (self.c1 * r1 * (p.p_best - p.position)) + \
                                (self.c2 * r2 * (self.g_best - p.position))
                    p.position = p.position + p.velocity
        
        # Reconstruct best path
        best_p = Particle(self.num_waypoints, self.start, self.goal, self.env)
        best_p.position = self.g_best
        return best_p.get_full_path(), self.g_best_score

    def optimize_annealing(self,T_0,beta, random_restart=False):
        for p in self.particles:
                score = p.evaluate()
                
                # Update local best
                if score < p.p_best_score:
                    p.p_best_score = score
                    p.p_best = p.position.copy()
                
                # Update global best
                if score < self.g_best_score:
                    self.g_best_score = score
                    self.g_best = p.position.copy()
        
        for k in range(self.max_iter):
            T = T_0 * (beta ** k)
            for p in self.particles:

                if random_restart and np.random.rand() < 0.1:
                    positions=np.random.rand(self.num_waypoints * 2)
                    for i in range(self.num_waypoints):
                        positions[2*i] *= self.env.width
                        positions[2*i+1] *= self.env.height
                    p.position = positions
                    p.velocity = np.zeros_like(positions)

                    if p.evaluate() < p.p_best_score:
                        p.p_best_score = p.evaluate()
                        p.p_best = p.position.copy()

                    if p.evaluate() < self.g_best_score:
                        self.g_best_score = p.evaluate()
                        self.g_best = p.position.copy()

                else:

                    r1 = np.random.rand(len(p.position))
                    r2 = np.random.rand(len(p.position))
                        
                    velocity = (self.w * p.velocity) + \
                                (self.c1 * r1 * (p.p_best - p.position)) + \
                                (self.c2 * r2 * (self.g_best - p.position))
                    position = p.position + p.velocity

                    fitness=self.evaluate_position(position)

                    delta=fitness-p.p_best_score

                    q=min(1,np.exp(-delta/T))

                    if np.random.rand() < q:
                        self.g_best_score = fitness
                        self.g_best = position.copy()

                    p.position = position
                    p.velocity = velocity

                    if fitness < p.p_best_score:
                        p.p_best_score = fitness
                    p.p_best = position.copy()

        best_p = Particle(self.num_waypoints, self.start, self.goal, self.env)
        personal_records=[p.p_best_score for p in self.particles]
        index_best=np.argmin(personal_records)
        record=self.particles[index_best].p_best.copy()
        best_p.position=record
        return best_p.get_full_path(), personal_records[index_best]





        

        
        

