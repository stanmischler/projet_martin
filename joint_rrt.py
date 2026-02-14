import numpy as np
from geometry import Environment
from rrt import RRT

class JointRRT:
    def __init__(self,speed,start_1,goal_1,start_2,goal_2,safe_distance,env, delta_s=50, delta_r=100, max_iter=3000):
        self.speed=speed
        self.start_1=start_1
        self.goal_1=goal_1
        self.start_2=start_2
        self.goal_2=goal_2
        self.safe_distance=safe_distance
        self.env=env
        self.delta_s=delta_s
        self.delta_r=delta_r
        self.max_iter=max_iter
        self.RRT_1=RRT(env,start_1,goal_1,delta_s,delta_r,max_iter)
        self.RRT_2=RRT(env,start_2,goal_2,delta_s,delta_r,max_iter)

    def not_collide(self,path1, path2, t_2, h=0.01):

         # Vitesse constante des robots (unités/sec)
        
        # --- 1. Prétraitement : Calcul des distances cumulées le long des chemins ---
        # Cela sert d'axe "temporel" pour l'interpolation
        
        # Pour path1

        SAFE_ZONE=self.safe_distance

        speed=self.speed
        dists1 = np.linalg.norm(path1[1:] - path1[:-1], axis=1) # Longueur de chaque segment
        cum_dist1 = np.insert(np.cumsum(dists1), 0, 0.0)        # [0, d1, d1+d2, ...]
        total_dist1 = cum_dist1[-1]
        
        # Pour path2
        dists2 = np.linalg.norm(path2[1:] - path2[:-1], axis=1)
        cum_dist2 = np.insert(np.cumsum(dists2), 0, 0.0)
        total_dist2 = cum_dist2[-1]
        
        # --- 2. Définition de la plage de temps ---
        # Temps de fin = max(fin robot 1, départ robot 2 + fin robot 2)
        t_end = max(total_dist1 / speed, t_2 + total_dist2 / speed)
        
        # On génère tous les instants à vérifier
        times = np.arange(0, t_end + h, h)
        
        for t in times:
            # --- Position Robot 1 ---
            dist_travelled_1 = t * speed
            
            # Interpolation (numpy gère automatiquement si on dépasse la fin -> il reste au dernier point)
            # On interpole X et Y séparément en fonction de la distance parcourue
            r1_x = np.interp(dist_travelled_1, cum_dist1, path1[:, 0])
            r1_y = np.interp(dist_travelled_1, cum_dist1, path1[:, 1])
            pos1 = np.array([r1_x, r1_y])
            
            # --- Position Robot 2 ---
            # Le robot 2 ne bouge que si t >= t_2
            if t < t_2:
                pos2 = path2[0] # Il attend au départ
            else:
                dist_travelled_2 = (t - t_2) * speed
                r2_x = np.interp(dist_travelled_2, cum_dist2, path2[:, 0])
                r2_y = np.interp(dist_travelled_2, cum_dist2, path2[:, 1])
                pos2 = np.array([r2_x, r2_y])
                
            # --- Vérification Collision ---
            if np.linalg.norm(pos1 - pos2) < SAFE_ZONE:
                return False # COLLISION DÉTECTÉE
                
        return True # CHEMIN SÛR

    def joint_plan(self, h=0.01):
        path1=np.array(self.RRT_1.plan(intelligent_sampling=True, triang_opt=False))
        path2=np.array(self.RRT_2.plan(intelligent_sampling=True, triang_opt=False))
        if path1 is None or path2 is None:
            return None
        t_2=0
        while not self.not_collide(path1, path2, t_2):
            t_2+=h
        return path1, path2, t_2

    



