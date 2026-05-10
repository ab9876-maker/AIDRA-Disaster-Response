"""
Search & Planning Module
Implements: BFS, DFS, Greedy Best-First, A*, Hill Climbing, Simulated Annealing
All algorithms return (path, stats_dict) for comparison.
"""

import heapq
import math
import random
import time
from collections import deque
from typing import List, Tuple, Dict, Optional, Callable

from core.environment import DisasterEnvironment, CellType


def reconstruct_path(came_from: Dict, start: Tuple, goal: Tuple) -> List[Tuple]:
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def path_cost(env: DisasterEnvironment, path: List[Tuple], risk_weight: float = 1.0) -> float:
    """Cost = distance + risk_weight * risk_exposure"""
    cost = 0
    risk = 0
    for cell in path:
        cost += 1
        if env.is_high_risk(cell):
            risk += 1
    return cost + risk_weight * risk


def risk_cells_in_path(env: DisasterEnvironment, path: List[Tuple]) -> int:
    return sum(1 for p in path if env.is_high_risk(p))


# ─── BFS ──────────────────────────────────────────────────────────────────────

def bfs(env: DisasterEnvironment, start: Tuple, goal: Tuple,
        allow_risk: bool = True) -> Tuple[Optional[List[Tuple]], Dict]:
    t0 = time.perf_counter()
    frontier = deque([start])
    came_from = {start: None}
    nodes_expanded = 0
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        current = frontier.popleft()
        nodes_expanded += 1

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            elapsed = (time.perf_counter() - t0) * 1000
            return path, {
                "algorithm": "BFS",
                "nodes_expanded": nodes_expanded,
                "max_frontier": max_frontier,
                "path_length": len(path),
                "path_cost": path_cost(env, path),
                "risk_cells": risk_cells_in_path(env, path),
                "time_ms": round(elapsed, 3),
                "complete": True,
                "optimal": True,
            }

        for neighbor in env.get_neighbors(current, allow_risk=allow_risk):
            if neighbor not in came_from:
                came_from[neighbor] = current
                frontier.append(neighbor)

    elapsed = (time.perf_counter() - t0) * 1000
    return None, {"algorithm": "BFS", "nodes_expanded": nodes_expanded,
                  "complete": False, "time_ms": round(elapsed, 3)}


# ─── DFS ──────────────────────────────────────────────────────────────────────

def dfs(env: DisasterEnvironment, start: Tuple, goal: Tuple,
        allow_risk: bool = True) -> Tuple[Optional[List[Tuple]], Dict]:
    t0 = time.perf_counter()
    stack = [start]
    came_from = {start: None}
    nodes_expanded = 0
    max_frontier = 1

    while stack:
        max_frontier = max(max_frontier, len(stack))
        current = stack.pop()
        nodes_expanded += 1

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            elapsed = (time.perf_counter() - t0) * 1000
            return path, {
                "algorithm": "DFS",
                "nodes_expanded": nodes_expanded,
                "max_frontier": max_frontier,
                "path_length": len(path),
                "path_cost": path_cost(env, path),
                "risk_cells": risk_cells_in_path(env, path),
                "time_ms": round(elapsed, 3),
                "complete": True,
                "optimal": False,
            }

        for neighbor in env.get_neighbors(current, allow_risk=allow_risk):
            if neighbor not in came_from:
                came_from[neighbor] = current
                stack.append(neighbor)

    elapsed = (time.perf_counter() - t0) * 1000
    return None, {"algorithm": "DFS", "nodes_expanded": nodes_expanded,
                  "complete": False, "time_ms": round(elapsed, 3)}


# ─── Greedy Best-First ────────────────────────────────────────────────────────

def greedy_best_first(env: DisasterEnvironment, start: Tuple, goal: Tuple,
                       allow_risk: bool = True) -> Tuple[Optional[List[Tuple]], Dict]:
    t0 = time.perf_counter()
    h = lambda pos: env.manhattan_distance(pos, goal)
    frontier = [(h(start), start)]
    came_from = {start: None}
    nodes_expanded = 0
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        _, current = heapq.heappop(frontier)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            elapsed = (time.perf_counter() - t0) * 1000
            return path, {
                "algorithm": "Greedy-BF",
                "nodes_expanded": nodes_expanded,
                "max_frontier": max_frontier,
                "path_length": len(path),
                "path_cost": path_cost(env, path),
                "risk_cells": risk_cells_in_path(env, path),
                "time_ms": round(elapsed, 3),
                "complete": True,
                "optimal": False,
            }

        for neighbor in env.get_neighbors(current, allow_risk=allow_risk):
            if neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(frontier, (h(neighbor), neighbor))

    elapsed = (time.perf_counter() - t0) * 1000
    return None, {"algorithm": "Greedy-BF", "nodes_expanded": nodes_expanded,
                  "complete": False, "time_ms": round(elapsed, 3)}


# ─── A* ───────────────────────────────────────────────────────────────────────

def astar(env: DisasterEnvironment, start: Tuple, goal: Tuple,
          allow_risk: bool = True, risk_penalty: float = 3.0) -> Tuple[Optional[List[Tuple]], Dict]:
    """
    A* with composite heuristic: h(n) = manhattan_distance + risk_bonus
    risk_penalty: extra cost per high-risk cell traversed (trades off time vs. risk).
    Justification: higher risk_penalty → agent prefers safer but longer routes.
    """
    t0 = time.perf_counter()

    def h(pos):
        return env.manhattan_distance(pos, goal)

    def step_cost(pos):
        base = 1
        if env.is_high_risk(pos):
            base += risk_penalty
        return base

    g_cost = {start: 0}
    f_cost = {start: h(start)}
    came_from = {start: None}
    frontier = [(f_cost[start], start)]
    nodes_expanded = 0
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        _, current = heapq.heappop(frontier)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            elapsed = (time.perf_counter() - t0) * 1000
            return path, {
                "algorithm": "A*",
                "nodes_expanded": nodes_expanded,
                "max_frontier": max_frontier,
                "path_length": len(path),
                "path_cost": round(g_cost[goal], 2),
                "risk_cells": risk_cells_in_path(env, path),
                "time_ms": round(elapsed, 3),
                "complete": True,
                "optimal": True,
                "risk_penalty_used": risk_penalty,
            }

        for neighbor in env.get_neighbors(current, allow_risk=allow_risk):
            tentative_g = g_cost[current] + step_cost(neighbor)
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g
                f_cost[neighbor] = tentative_g + h(neighbor)
                heapq.heappush(frontier, (f_cost[neighbor], neighbor))

    elapsed = (time.perf_counter() - t0) * 1000
    return None, {"algorithm": "A*", "nodes_expanded": nodes_expanded,
                  "complete": False, "time_ms": round(elapsed, 3)}


# ─── Hill Climbing (Local Search for route optimization) ─────────────────────

def hill_climbing_route(env: DisasterEnvironment, victims_order: List,
                         start: Tuple) -> Tuple[List, Dict]:
    """
    Hill climbing to optimize victim visitation order.
    Objective: minimize total estimated Manhattan distance.
    """
    t0 = time.perf_counter()

    def total_distance(order):
        dist = 0
        pos = start
        for v in order:
            dist += env.manhattan_distance(pos, v.position)
            pos = v.position
        dist += min(env.manhattan_distance(pos, mc) for mc in env.medical_centers)
        return dist

    current_order = victims_order[:]
    current_cost = total_distance(current_order)
    iterations = 0
    improvements = 0

    while True:
        iterations += 1
        best_neighbor = None
        best_cost = current_cost

        for i in range(len(current_order)):
            for j in range(i + 1, len(current_order)):
                neighbor = current_order[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                cost = total_distance(neighbor)
                if cost < best_cost:
                    best_cost = cost
                    best_neighbor = neighbor

        if best_neighbor is None:
            break
        current_order = best_neighbor
        current_cost = best_cost
        improvements += 1

    elapsed = (time.perf_counter() - t0) * 1000
    return current_order, {
        "algorithm": "Hill Climbing",
        "iterations": iterations,
        "improvements": improvements,
        "final_cost": round(current_cost, 2),
        "time_ms": round(elapsed, 3),
    }


# ─── Simulated Annealing ──────────────────────────────────────────────────────

def simulated_annealing_route(env: DisasterEnvironment, victims_order: List,
                               start: Tuple, T_start: float = 100.0,
                               T_end: float = 0.1, cooling: float = 0.95) -> Tuple[List, Dict]:
    """
    Simulated Annealing for victim visitation order optimization.
    Can escape local optima unlike Hill Climbing.
    """
    t0 = time.perf_counter()

    def total_distance(order):
        dist = 0
        pos = start
        for v in order:
            dist += env.manhattan_distance(pos, v.position)
            pos = v.position
        dist += min(env.manhattan_distance(pos, mc) for mc in env.medical_centers)
        return dist

    current = victims_order[:]
    current_cost = total_distance(current)
    best = current[:]
    best_cost = current_cost
    T = T_start
    iterations = 0
    accepted = 0

    while T > T_end and len(current) > 1:
        iterations += 1
        i, j = random.sample(range(len(current)), 2)
        neighbor = current[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbor_cost = total_distance(neighbor)
        delta = neighbor_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor
            current_cost = neighbor_cost
            accepted += 1
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        T *= cooling

    elapsed = (time.perf_counter() - t0) * 1000
    return best, {
        "algorithm": "Simulated Annealing",
        "iterations": iterations,
        "accepted_moves": accepted,
        "final_cost": round(best_cost, 2),
        "final_temperature": round(T, 4),
        "time_ms": round(elapsed, 3),
    }


# ─── Multi-algorithm Planner ─────────────────────────────────────────────────

class RoutePlanner:
    """
    Runs all search algorithms and selects the best based on a trade-off policy.
    """

    def __init__(self, env: DisasterEnvironment):
        self.env = env
        self.comparison_log = []

    def plan_route(self, start: Tuple, goal: Tuple,
                   policy: str = "balanced") -> Tuple[Optional[List[Tuple]], Dict]:
        """
        policy options:
          'fast'     → minimize path length (use Greedy-BF)
          'safe'     → avoid risk zones (A* with high risk_penalty, no risk allowed)
          'balanced' → A* with moderate risk_penalty
          'compare'  → run all, return best by A* balanced
        """
        results = {}

        bfs_path, bfs_stats = bfs(self.env, start, goal, allow_risk=True)
        dfs_path, dfs_stats = dfs(self.env, start, goal, allow_risk=True)
        greedy_path, greedy_stats = greedy_best_first(self.env, start, goal, allow_risk=True)
        astar_balanced_path, astar_balanced_stats = astar(self.env, start, goal, allow_risk=True, risk_penalty=3.0)
        astar_safe_path, astar_safe_stats = astar(self.env, start, goal, allow_risk=False, risk_penalty=10.0)

        results = {
            "BFS": (bfs_path, bfs_stats),
            "DFS": (dfs_path, dfs_stats),
            "Greedy-BF": (greedy_path, greedy_stats),
            "A*_balanced": (astar_balanced_path, astar_balanced_stats),
            "A*_safe": (astar_safe_path, astar_safe_stats),
        }

        self.comparison_log.append({
            "start": start, "goal": goal, "policy": policy,
            "comparisons": {k: v[1] for k, v in results.items()}
        })

        if policy == "fast":
            chosen = results["Greedy-BF"]
            reason = "Greedy Best-First selected: minimizes estimated distance (fastest route, higher risk)"
        elif policy == "safe":
            chosen = results["A*_safe"] if results["A*_safe"][0] else results["A*_balanced"]
            reason = "A* (safe) selected: avoids all high-risk zones (longer route, zero risk)"
        else:  # balanced
            # Prefer A*_balanced; fallback if no path
            chosen = results["A*_balanced"] if results["A*_balanced"][0] else results["A*_safe"]
            reason = "A* (balanced, risk_penalty=3.0) selected: optimal trade-off between time and risk"

        if chosen[0]:
            chosen[1]["selection_reason"] = reason
            chosen[1]["policy"] = policy

        return chosen[0], chosen[1], results
