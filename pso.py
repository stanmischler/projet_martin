import numpy as np
from geometry import Environment

class Particle:
    def __init__(self, num_waypoints, start, goal, env: Environment):
        self.start = start
        self.goal = goal
        self.num_waypoints = num_waypoints
        self.env = env
        self.reset()

    def reset(self):
        """Réinitialise la particule à une position et vitesse aléatoire."""
        # Position: vecteur plat [x1, y1, x2, y2, ...]
        self.position = np.random.rand(self.num_waypoints * 2)
        for i in range(self.num_waypoints):
            self.position[2*i] *= self.env.width
            self.position[2*i+1] *= self.env.height
            
        self.velocity = np.zeros_like(self.position)
        self.p_best = self.position.copy()
        self.p_best_score = float('inf')

    def evaluate(self):
        """Fitness function: Longueur du chemin + Pénalité collision"""
        path = self.get_full_path()
        length = 0
        penalty = 0
        
        # ... boucle sur les segments ...
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            
            # Distance euclidienne (on veut minimiser la longueur)
            dist = np.linalg.norm(p2 - p1)
            length += dist
            
            # --- NOUVELLE LOGIQUE ---
            # On appelle la fonction hybride qui gère Exactitude + Gradient
            penalty_score = self.env.compute_collision_penalty(p1, p2, steps=20)
            penalty += penalty_score

        penalty += self.env.penalize_self_intersection(path) # Pénalité très forte pour les auto-intersections (boucles)
                
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
        
        # Initialisation des particules
        self.particles = [Particle(num_waypoints, start, goal, env) for _ in range(num_particles)]
        self.g_best = self.particles[0].position.copy()
        self.g_best_score = float('inf')
        
        # Hyperparamètres standards PSO
        self.w = 0.5   # Inertie
        self.c1 = 1.5  # Cognitif (attraction vers p_best)
        self.c2 = 1.5  # Social (attraction vers g_best)

    def evaluate_position(self, position):
        """
        Helper pour évaluer une position arbitraire (utilisé par le Dimensional Learning)
        """
        waypoints = position.reshape((-1, 2))
        path = np.vstack([self.start, waypoints, self.goal])
        length = 0
        penalty = 0
        # ... boucle sur les segments ...
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            
            # Distance euclidienne (on veut minimiser la longueur)
            dist = np.linalg.norm(p2 - p1)
            length += dist
            
            # --- NOUVELLE LOGIQUE ---
            # On appelle la fonction hybride qui gère Exactitude + Gradient
            penalty_score = self.env.compute_collision_penalty(p1, p2, steps=20)
            penalty += penalty_score

        penalty += self.env.penalize_self_intersection(path) # Pénalité très forte pour les auto-intersections (boucles)
                
        return length + penalty

    def optimize(self, random_restart=False, simulated_annealing=False, dimensional_learning=False):
        """
        Version optimisée et sécurisée (Clamping + Best Memory).
        """
        # Paramètres Recuit Simulé
        T = 2000.0 if simulated_annealing else 0  # Température initiale plus haute pour Scénario 3
        beta = 0.97 # Refroidissement plus lent (pour explorer plus longtemps)

        # Sauvegarde de la meilleure solution ABSOLUE trouvée (indépendamment du mouvement de l'essaim)
        best_ever_position = self.g_best.copy()
        best_ever_score = self.g_best_score

        history = []
        
        # Limitation de vitesse (pour éviter que les particules traversent la carte en 1 itération)
        v_max = 0.2 * max(self.env.width, self.env.height)

        for k in range(self.max_iter):
            # 1. Évaluation et Mise à jour
            for p in self.particles:
                score = p.evaluate()
                
                # Update Local Best
                if score < p.p_best_score:
                    p.p_best_score = score
                    p.p_best = p.position.copy()
                
                # Update Global Best (Attracteur de l'essaim)
                delta = score - self.g_best_score
                
                accepted = False
                if delta < 0:
                    accepted = True
                elif simulated_annealing and T > 1e-3:
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        accepted = True
                
                if accepted:
                    self.g_best_score = score
                    self.g_best = p.position.copy()
                    
                    # Si c'est une amélioration REELLE, on met à jour le record historique
                    if score < best_ever_score:
                        best_ever_score = score
                        best_ever_position = p.position.copy()

            # 2. Dimensional Learning (Amélioré pour Scénario 3)
            # On ne l'applique que si on n'est pas en train de faire du "bruit" avec le recuit
            if dimensional_learning:
                dims = len(self.g_best)
                # On teste un pas plus grand pour débloquer les situations difficiles
                steps_to_test = [5.0, 1.0] 
                
                for i in range(dims):
                    original_val = self.g_best[i]
                    best_val = original_val
                    current_dim_score = self.g_best_score
                    
                    for step in steps_to_test:
                        # Test +step
                        self.g_best[i] = original_val + step
                        score = self.evaluate_position(self.g_best)
                        if score < current_dim_score:
                            current_dim_score = score
                            best_val = self.g_best[i]
                            # Si on améliore, on met aussi à jour le record historique
                            if score < best_ever_score:
                                best_ever_score = score
                                best_ever_position = self.g_best.copy()
                            break # On a trouvé mieux, on passe à la dimension suivante

                        # Test -step
                        self.g_best[i] = original_val - step
                        score = self.evaluate_position(self.g_best)
                        if score < current_dim_score:
                            current_dim_score = score
                            best_val = self.g_best[i]
                            if score < best_ever_score:
                                best_ever_score = score
                                best_ever_position = self.g_best.copy()
                            break
                    
                    self.g_best[i] = best_val # On valide la meilleure modif
                    self.g_best_score = current_dim_score

            # 3. Mouvement des particules (Avec Clamping)
            for p in self.particles:
                if random_restart and np.random.rand() < 0.005: # 0.5% suffit avec 200 particules
                    p.reset()
                else:
                    r1 = np.random.rand(len(p.position))
                    r2 = np.random.rand(len(p.position))
                    
                    # Mise à jour vitesse
                    new_vel = (self.w * p.velocity) + \
                              (self.c1 * r1 * (p.p_best - p.position)) + \
                              (self.c2 * r2 * (self.g_best - p.position))
                    
                    # Clamping de la vitesse (Limitation)
                    # Empêche la particule de devenir folle
                    norm_v = np.linalg.norm(new_vel)
                    if norm_v > v_max:
                        new_vel = (new_vel / norm_v) * v_max
                    
                    p.velocity = new_vel
                    p.position = p.position + p.velocity
                    
                    # CLAMPING DE LA POSITION (Crucial !)
                    # On ramène les points qui sortent de la carte à l'intérieur
                    for i in range(self.num_waypoints):
                        p.position[2*i] = np.clip(p.position[2*i], 0, self.env.width)
                        p.position[2*i+1] = np.clip(p.position[2*i+1], 0, self.env.height)

            if simulated_annealing:
                T *= beta
            
            history.append(self.g_best) # On plot le score réel, pas celui du recuit qui oscille
        
        # On retourne le MEILLEUR historique, pas forcément celui de la dernière itération
        final_p = Particle(self.num_waypoints, self.start, self.goal, self.env)
        final_p.position = best_ever_position
        return final_p.get_full_path(), best_ever_score, history