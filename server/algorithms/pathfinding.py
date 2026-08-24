"""
server/algorithms/pathfinding.py
A* and Dijkstra Graph Pathfinding with Dynamic Ocean Current Vector Weighting.
"""

from __future__ import annotations
import math
import heapq
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass

from .kdtree import haversine_distance


@dataclass
class OceanCell:
    row: int
    col: int
    lat: float
    lon: float
    u_current: float  # Eastward current velocity in knots
    v_current: float  # Northward current velocity in knots
    is_land: bool = False
    bathymetry_depth: float = 100.0  # Meters


class OceanGridGraph:
    """
    Discretized ocean surface mesh supporting dynamic velocity fields.
    """

    def __init__(self, rows: int, cols: int, top_left: Tuple[float, float], res_deg: float = 0.1):
        self.rows = rows
        self.cols = cols
        self.res_deg = res_deg
        self.grid: List[List[OceanCell]] = []
        
        start_lat, start_lon = top_left
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                lat = start_lat - (r * res_deg)
                lon = start_lon + (c * res_deg)
                row_cells.append(OceanCell(row=r, col=c, lat=lat, lon=lon, u_current=0.0, v_current=0.0))
            self.grid.append(row_cells)

    def set_current(self, r: int, c: int, u: float, v: float):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid[r][c].u_current = u
            self.grid[r][c].v_current = v

    def set_obstacle(self, r: int, c: int, is_land: bool = True):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid[r][c].is_land = is_land

    def get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """8-directional navigation on ocean grid."""
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        valid_neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if not self.grid[nr][nc].is_land:
                    valid_neighbors.append((nr, nc))
        return valid_neighbors

    def transition_cost(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        vessel_speed_knots: float = 12.0
    ) -> float:
        """
        Calculates travel time (hours) considering vector dot-product of current and heading.
        """
        cell_from = self.grid[from_pos[0]][from_pos[1]]
        cell_to = self.grid[to_pos[0]][to_pos[1]]

        # Distance in nautical miles (1 km ~ 0.539957 NM)
        dist_km = haversine_distance((cell_from.lat, cell_from.lon), (cell_to.lat, cell_to.lon))
        dist_nm = dist_km * 0.539957

        # Unit direction vector
        d_lat = cell_to.lat - cell_from.lat
        d_lon = cell_to.lon - cell_from.lon
        mag = math.hypot(d_lat, d_lon)
        if mag == 0:
            return 0.0

        dir_n = d_lat / mag  # North component
        dir_e = d_lon / mag  # East component

        # Average current between cells
        avg_u = (cell_from.u_current + cell_to.u_current) / 2.0  # East
        avg_v = (cell_from.v_current + cell_to.v_current) / 2.0  # North

        # Effective ground speed = Vessel Speed + Projected Current Vector
        current_along_track = (avg_u * dir_e) + (avg_v * dir_n)
        effective_speed = vessel_speed_knots + current_along_track

        # Minimum ground speed cutoff to avoid zero/negative travel speed
        min_feasible_speed = max(0.5, vessel_speed_knots * 0.1)
        if effective_speed < min_feasible_speed:
            effective_speed = min_feasible_speed

        # Travel time in hours (cost metric)
        return dist_nm / effective_speed


def heuristic(pos: Tuple[int, int], goal: Tuple[int, int], graph: OceanGridGraph, max_speed_knots: float) -> float:
    """
    Admissible A* heuristic: Great-circle distance divided by maximum possible ground speed.
    """
    c1 = graph.grid[pos[0]][pos[1]]
    c2 = graph.grid[goal[0]][goal[1]]
    dist_nm = haversine_distance((c1.lat, c1.lon), (c2.lat, c2.lon)) * 0.539957
    return dist_nm / max_speed_knots


def a_star_pathfinding(
    graph: OceanGridGraph,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    vessel_speed_knots: float = 12.0,
    max_current_knots: float = 4.0
) -> Dict[str, Any]:
    """
    A* algorithm optimized for ocean routing with current assistance.
    """
    if graph.grid[start[0]][start[1]].is_land or graph.grid[goal[0]][goal[1]].is_land:
        return {"success": False, "error": "Start or Goal point is on land."}

    max_possible_speed = vessel_speed_knots + max_current_knots

    # Priority queue stores (f_score, counter, current_node)
    counter = 0
    open_set: List[Tuple[float, int, Tuple[int, int]]] = []
    heapq.heappush(open_set, (0.0, counter, start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    f_score: Dict[Tuple[int, int], float] = {start: heuristic(start, goal, graph, max_possible_speed)}
    closed_set: Set[Tuple[int, int]] = set()

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            curr_step = current
            while curr_step in came_from:
                c_node = graph.grid[curr_step[0]][curr_step[1]]
                path.append({
                    "grid_pos": curr_step,
                    "lat": c_node.lat,
                    "lon": c_node.lon,
                    "u_current": c_node.u_current,
                    "v_current": c_node.v_current
                })
                curr_step = came_from[curr_step]
            c_start = graph.grid[start[0]][start[1]]
            path.append({
                "grid_pos": start,
                "lat": c_start.lat,
                "lon": c_start.lon,
                "u_current": c_start.u_current,
                "v_current": c_start.v_current
            })
            path.reverse()
            return {
                "success": True,
                "algorithm": "A*",
                "total_time_hours": g_score[goal],
                "waypoint_count": len(path),
                "waypoints": path
            }

        if current in closed_set:
            continue
        closed_set.add(current)

        for neighbor in graph.get_neighbors(current[0], current[1]):
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + graph.transition_cost(current, neighbor, vessel_speed_knots)

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = heuristic(neighbor, goal, graph, max_possible_speed)
                f = tentative_g + h
                f_score[neighbor] = f
                counter += 1
                heapq.heappush(open_set, (f, counter, neighbor))

    return {"success": False, "error": "No viable ocean path found"}


def dijkstra_pathfinding(
    graph: OceanGridGraph,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    vessel_speed_knots: float = 12.0
) -> Dict[str, Any]:
    """
    Uniform-cost Dijkstra search baseline without heuristic guidance.
    """
    if graph.grid[start[0]][start[1]].is_land or graph.grid[goal[0]][goal[1]].is_land:
        return {"success": False, "error": "Start or Goal point is on land."}

    counter = 0
    pq: List[Tuple[float, int, Tuple[int, int]]] = []
    heapq.heappush(pq, (0.0, counter, start))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}

    while pq:
        current_cost, _, current = heapq.heappop(pq)

        if current == goal:
            path = []
            curr_step = current
            while curr_step in came_from:
                c_node = graph.grid[curr_step[0]][curr_step[1]]
                path.append({
                    "grid_pos": curr_step,
                    "lat": c_node.lat,
                    "lon": c_node.lon
                })
                curr_step = came_from[curr_step]
            c_start = graph.grid[start[0]][start[1]]
            path.append({"grid_pos": start, "lat": c_start.lat, "lon": c_start.lon})
            path.reverse()
            return {
                "success": True,
                "algorithm": "Dijkstra",
                "total_time_hours": current_cost,
                "waypoint_count": len(path),
                "waypoints": path
            }

        if current_cost > cost_so_far.get(current, float('inf')):
            continue

        for neighbor in graph.get_neighbors(current[0], current[1]):
            new_cost = current_cost + graph.transition_cost(current, neighbor, vessel_speed_knots)
            if new_cost < cost_so_far.get(neighbor, float('inf')):
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(pq, (new_cost, counter, neighbor))

    return {"success": False, "error": "No viable path found"}