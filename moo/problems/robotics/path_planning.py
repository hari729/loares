
"""
path_planning_final.py

Final integrated module for multi-objective grid-based path planning.
- Hybrid initialization (A* + RRT) with fixed seed and safe RRT.
- Path shortening (Algorithm 4) applied during initialization.
- Vectorized objective evaluation: length, safety, smoothness.
- Fast variable modifier (round + clip). Optional in-modifier shortening.
- PathPlanningProblem class wrapping the evaluation for use with
  an optimizer that expects: function(population) -> (objectives, constraints).

Globals expected to be set externally before creating the Problem:
  grid_map : 2D numpy array (occupancy: 0 free, >=1 obstacle)
  start    : (x, y) tuple
  target   : (x, y) tuple
  n_points : number of intermediate waypoints (int) - may be set by hybrid_initialization

Note: This file is self-contained except for opti.core.problem import; if that
is unavailable, PathPlanningProblem instantiation will raise as expected.
"""

import numpy as np
import heapq
import random

# Optional import from user framework.
# If missing here, the import error will be raised only when the class is used.
try:
    from opti.core.problem import Problem
except Exception:
    # Provide a harmless placeholder to allow running standalone tests in environments without the framework.
    class Problem:
        def __init__(self, *args, **kwargs):
            pass


# ----------------------
# Global variables
# ----------------------
grid_map = None        # 2D numpy array: occupancy probability [0,1]
start = None           # tuple (x, y)
target = None          # tuple (x, y)
n_points = 10          # number of intermediate points (will often be set by hybrid_initialization)
GLOBAL_SEED = 42       # deterministic seed
APPLY_SHORTEN_IN_MODIFIER = False  # set True if you want shortening in every variable_modifier call


# ----------------------
# Utility: random seed
# ----------------------
def set_seed(seed=GLOBAL_SEED):
    """Set deterministic RNG state for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


# ----------------------
# CollisionFree (Bresenham-like)
# ----------------------
def CollisionFree(p1, p2, grid):
    """
    Return True if the straight segment between p1 and p2 does not intersect obstacle cells.
    p1, p2: (x, y) coordinates (can be floats; will be rounded to nearest cell).
    grid: 2D numpy occupancy grid where >= 1.0 indicates obstacle.
    """
    x1, y1 = np.round(p1).astype(int)
    x2, y2 = np.round(p2).astype(int)

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        # out-of-bounds => treat as collision (not free)
        if not (0 <= x1 < grid.shape[0] and 0 <= y1 < grid.shape[1]):
            return False
        if grid[x1, y1] >= 1.0:
            return False
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return True


# ----------------------
# A* search (4-connected grid)
# ----------------------
def run_astar(grid, start_pt, goal_pt):
    """
    Run A* on a 4-connected grid. Returns an array of (x,y) tuples path including start and goal,
    or None if unreachable.
    start_pt, goal_pt are tuples (x,y).
    """
    sx, sy = start_pt
    gx, gy = goal_pt
    # simple Manhattan heuristic
    h = lambda x, y: abs(x - gx) + abs(y - gy)

    open_set = [(0 + h(sx, sy), 0, (sx, sy), None)]
    came_from = {}
    g_score = { (sx, sy): 0 }
    visited = set()

    while open_set:
        _, cost, current, parent = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        came_from[current] = parent

        if current == goal_pt:
            break

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]):
                continue
            if grid[nx, ny] >= 1.0:
                continue
            new_cost = cost + 1
            if (nx, ny) not in g_score or new_cost < g_score[(nx, ny)]:
                g_score[(nx, ny)] = new_cost
                heapq.heappush(open_set, (new_cost + h(nx, ny), new_cost, (nx, ny), current))

    # reconstruct
    node = goal_pt
    if node not in came_from:
        return None
    path = []
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()
    return np.array(path)


# ----------------------
# Safer RRT (tuple-based nodes, max_nodes cap)
# ----------------------
def run_rrt(grid, start_pt, goal_pt, max_iters=1000, step=3, goal_prob=0.1, max_nodes=5000):
    """
    Simple RRT variant. Returns array of (x,y) nodes (including start and goal) or None.
    Uses tuple nodes for hashing, caps the number of nodes to avoid runaway memory.
    """
    h, w = grid.shape
    nodes = [tuple(start_pt)]
    parents = { tuple(start_pt): None }

    for _ in range(max_iters):
        if random.random() < goal_prob:
            sample = np.array(goal_pt)
        else:
            sample = np.array([random.randint(0, h-1), random.randint(0, w-1)])

        # find nearest node (linear search; OK for modest node counts)
        dists = [np.linalg.norm(np.array(n) - sample) for n in nodes]
        nearest_t = nodes[int(np.argmin(dists))]
        nearest = np.array(nearest_t)

        direction = sample - nearest
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction = (direction / norm) * min(step, norm)
        new_arr = np.round(nearest + direction).astype(int)
        new_node = (int(new_arr[0]), int(new_arr[1]))

        # validity checks
        if not (0 <= new_node[0] < h and 0 <= new_node[1] < w):
            continue
        if not CollisionFree(np.array(nearest_t), np.array(new_node), grid):
            continue
        if new_node in parents:
            continue

        nodes.append(new_node)
        parents[new_node] = nearest_t

        # goal reached?
        if np.linalg.norm(np.array(new_node) - np.array(goal_pt)) < step:
            parents[tuple(goal_pt)] = new_node
            break

        if len(nodes) > max_nodes:
            return None

    if tuple(goal_pt) not in parents:
        return None

    # reconstruct path safely (guard against unexpected loops)
    path = []
    node = tuple(goal_pt)
    visited = set()
    while node is not None:
        if node in visited:
            break
        visited.add(node)
        path.append(node)
        node = parents.get(node)
    path.reverse()
    return np.array(path)


# ----------------------
# Path Shortening (Algorithm 4)
# ----------------------
def path_shortening(path, grid):
    """
    Remove redundant waypoints: if segment (i -> i+2) is collision-free,
    drop i+1. Returns shortened path (numpy array).
    """
    shortened = [path[0]]
    i = 0
    while i < len(path) - 2:
        if CollisionFree(path[i], path[i + 2], grid):
            i += 1
        else:
            shortened.append(path[i + 1])
            i += 1
    shortened.append(path[-1])
    return np.array(shortened)


# ----------------------
# Hybrid Initialization (A* + RRT) - sets n_points adaptively
# ----------------------
def hybrid_initialization(grid, start_pt, goal_pt, pop_size=100, max_rrt_tries_factor=5):
    """
    Build initial population using A* (half) + RRT (half), as in the paper.
    Applies path_shortening() to each found path, then determines n_points:
      n_points = max(1, (len(longest_path) - 2) * 2)
    Returns population as numpy array shape (pop_size, n_points*2).
    """
    set_seed(GLOBAL_SEED)
    init_paths = []

    astar_path = run_astar(grid, start_pt, goal_pt)
    if astar_path is None:
        raise RuntimeError("A* failed to find a feasible path.")
    astar_path = path_shortening(astar_path, grid)

    # half population are clones of A*
    for _ in range(pop_size // 2):
        init_paths.append(astar_path)

    # remainder from RRT
    tries = 0
    max_tries = pop_size * max_rrt_tries_factor
    while len(init_paths) < pop_size and tries < max_tries:
        rrt_path = run_rrt(grid, start_pt, goal_pt)
        tries += 1
        if rrt_path is not None:
            rrt_path = path_shortening(rrt_path, grid)
            init_paths.append(rrt_path)

    if len(init_paths) == 0:
        raise RuntimeError("Hybrid initialization failed to produce any feasible paths.")

    # pick longest and set n_points (twice intermediate points for safety)
    longest = max(init_paths, key=lambda p: len(p))
    max_len = len(longest)
    global n_points
    n_points = max(1, (max_len - 2) * 2)

    # convert to flat population array (pad or truncate cores to n_points)
    pop = np.zeros((len(init_paths), n_points * 2), dtype=float)
    for i, pth in enumerate(init_paths):
        core = np.array(pth[1:-1], dtype=float)  # intermediate waypoints
        if core.size == 0:
            core = np.array([start_pt], dtype=float)
        if len(core) < n_points:
            pad = np.repeat(core[-1][np.newaxis, :], n_points - len(core), axis=0)
            core = np.vstack((core, pad))
        elif len(core) > n_points:
            core = core[:n_points]
        pop[i, :] = core.flatten()
    return pop


# ----------------------
# Variable modifier (fast, vectorized)
# ----------------------
def path_var_modifier(solutions):
    """
    Round variables to integer grid indices and clip to bounds.
    Optional: apply path_shortening per individual if APPLY_SHORTEN_IN_MODIFIER=True.
    Input:
      solutions: (pop_size, n_points*2) float numpy array
    Returns same shaped numpy array (float, but integer values).
    """
    global grid_map, start, target, n_points

    if grid_map is None:
        raise ValueError("grid_map must be defined before calling path_var_modifier")

    x_max, y_max = grid_map.shape

    # round & clip vectorized
    sols = np.rint(solutions)
    sols[:, 0::2] = np.clip(sols[:, 0::2], 0, x_max - 1)
    sols[:, 1::2] = np.clip(sols[:, 1::2], 0, y_max - 1)

    if APPLY_SHORTEN_IN_MODIFIER:
        # Apply shortening per individual but keep fixed-length representation by padding/truncating
        new_sols = np.zeros_like(sols)
        for idx, sol in enumerate(sols):
            path = sol.reshape(n_points, 2)
            full_path = np.vstack(([start], path, [target]))
            shortened = path_shortening(full_path, grid_map)
            core = shortened[1:-1]
            if core.size == 0:
                core = np.array([start])
            if len(core) < n_points:
                pad = np.repeat(core[-1][np.newaxis, :], n_points - len(core), axis=0)
                core = np.vstack((core, pad))
            elif len(core) > n_points:
                core = core[:n_points]
            new_sols[idx, :] = core.flatten()
        return new_sols

    return sols


# ----------------------
# Objective evaluator (vectorized where possible)
# ----------------------
def path_planning_objectives(population):
    """
    Evaluate objectives and constraints for a population.
    Input:
      population: (pop_size, n_points*2) numpy array
    Outputs:
      objectives: (pop_size, 3) -> [length, safety, smoothness]
      constraints: (pop_size, 1) -> number of colliding segments (0 means feasible)
    Note: length & smoothness are vectorized; collision checks remain per segment.
    """
    global grid_map, start, target, n_points

    if grid_map is None:
        raise ValueError("grid_map must be defined before calling path_planning_objectives")

    pop = np.asarray(population)
    n = pop.shape[0]
    if pop.shape[1] != n_points * 2:
        raise ValueError(f"population columns ({pop.shape[1]}) != n_points*2 ({n_points*2})")

    paths = pop.reshape(n, n_points, 2).astype(float)

    # construct full paths (start + intermediates + target)
    start_tile = np.tile(np.array(start, dtype=float), (n, 1))[:, np.newaxis, :]
    target_tile = np.tile(np.array(target, dtype=float), (n, 1))[:, np.newaxis, :]
    full_paths = np.concatenate([start_tile, paths, target_tile], axis=1)  # shape (n, n_points+2, 2)

    # 1) Path lengths (vectorized)
    diffs = np.diff(full_paths, axis=1)  # shape (n, n_segments, 2)
    lengths = np.sum(np.linalg.norm(diffs, axis=2), axis=1)

    # 2) Smoothness: sum of (pi - angle) over internal points (vectorized)
    if diffs.shape[1] >= 2:
        v1 = diffs[:, :-1, :]  # segments 0..m-1
        v2 = diffs[:, 1:, :]   # segments 1..m
        dot = np.einsum('nij,nij->ni', v1, v2)
        norm = np.linalg.norm(v1, axis=2) * np.linalg.norm(v2, axis=2)
        cos_ang = np.clip(dot / (norm + 1e-12), -1.0, 1.0)
        angles = np.arccos(cos_ang)
        smoothness = np.sum(np.pi - angles, axis=1)
    else:
        smoothness = np.zeros(n)

    # 3) Safety: sample occupancy at rounded coordinates of waypoints + endpoints (vectorized)
    coords = np.rint(full_paths).astype(int)
    coords[..., 0] = np.clip(coords[..., 0], 0, grid_map.shape[0]-1)
    coords[..., 1] = np.clip(coords[..., 1], 0, grid_map.shape[1]-1)
    # sum occupancy values along the path nodes (approximation used in paper)
    safety = np.sum(grid_map[coords[..., 0], coords[..., 1]], axis=1)

    # 4) Constraints: number of colliding segments per individual (needs per-segment check)
    constraints = np.zeros(n, dtype=float)
    for i in range(n):
        fp = full_paths[i]
        collisions = 0
        # iterate segments
        for j in range(fp.shape[0] - 1):
            if not CollisionFree(fp[j], fp[j+1], grid_map):
                collisions += 1
        constraints[i] = collisions

    objectives = np.column_stack([lengths, safety, smoothness])
    return objectives, constraints.reshape(-1, 1)


# ----------------------
# Problem wrapper class (framework-compatible)
# ----------------------
class PathPlanningProblem(Problem):
    """
    Wraps path_planning_objectives into a Problem object expected by your optimization framework.
    Must set globals grid_map, start, target, and let hybrid_initialization set n_points if used.
    """
    def __init__(self, psize=100, max_evals=50000, init_pop=None):
        global grid_map, n_points, start, target

        if grid_map is None or start is None or target is None:
            raise ValueError("Define 'grid_map', 'start', and 'target' globals before creating the problem.")

        # Construct rectangular bounds per variable: x uses grid_map.shape[0], y uses grid_map.shape[1]
        bounds = np.zeros((n_points * 2, 2))
        bounds[0::2, :] = [0, grid_map.shape[0]]  # x bounds
        bounds[1::2, :] = [0, grid_map.shape[1]]  # y bounds

        super().__init__(
            function=path_planning_objectives,
            n_vars=n_points * 2,
            n_obj=3,
            n_constr=1,
            psize=psize,
            max_evals=max_evals,
            bounds=bounds,
            minmax=np.array([1, 1, 1]),
            variable_modifier=path_var_modifier
        )

        # Optionally set initial population (framework may expect a certain attribute name)
        if init_pop is not None:
            try:
                self.initial_population = init_pop
            except Exception:
                # If framework uses different attribute or setter, user can assign manually after creation.
                pass


# ----------------------
# End of module
# ----------------------
