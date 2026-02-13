import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

    def check_collision_segment(self, p1, p2, num_checks=20):
        # Discretization approach for simplicity
        # A more robust approach would use line-box intersection math
        vector = p2 - p1
        dist = np.linalg.norm(vector)
        if dist == 0: return self.check_collision_point(p1)
        
        steps = int(dist / 5.0) + 2 # Check every 5 units approx
        for i in range(steps + 1):
            p = p1 + (vector * (i / steps))
            if self.check_collision_point(p):
                return True
        return False

    def plot(self, ax, path=None, color='blue', title="Environment"):
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        for (ox, oy, w, h) in self.obstacles:
            rect = patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)
        
        if path is not None and len(path) > 0:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2, label='Path')
        
        ax.set_title(title)
        ax.set_aspect('equal')

def parse_input_file(filepath):
    """Parses the specific file format described in the project."""
    with open(filepath, 'r') as f:
        # Assuming all data is on one line or space separated based on typical competitive programming formats
        data = f.read().replace(',', ' ').split()
        data = [float(x) for x in data]
        
    idx = 0
    xmax, ymax = data[idx], data[idx+1]; idx += 2
    start_x, start_y = data[idx], data[idx+1]; idx += 2
    goal_x, goal_y = data[idx], data[idx+1]; idx += 2
    start2_x, start2_y = data[idx], data[idx+1]; idx += 2
    goal2_x, goal2_y = data[idx], data[idx+1]; idx += 2
    R = data[idx]; idx += 1
    
    obstacles = []
    while idx < len(data):
        ox, oy, lx, ly = data[idx], data[idx+1], data[idx+2], data[idx+3]
        obstacles.append((ox, oy, lx, ly))
        idx += 4
        
    env = Environment(xmax, ymax, obstacles)
    return env, np.array([start_x, start_y]), np.array([goal_x, goal_y]), np.array([start2_x, start2_y]), np.array([goal2_x, goal2_y]), R