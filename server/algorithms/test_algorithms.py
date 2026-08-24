"""
server/algorithms/test_algorithms.py
Unit tests verifying KD-Tree performance & Current-Aware Pathfinding.
"""

import pytest
import math
from server.algorithms.kdtree import KDTree, haversine_distance
from server.algorithms.clustering import FishCluster, FishClusterSpatialIndex
from server.algorithms.pathfinding import OceanGridGraph, a_star_pathfinding, dijkstra_pathfinding


def test_haversine_accuracy():
    # Mumbai to Goa ~ 430-440 km
    mumbai = (18.9220, 72.8347)
    goa = (15.4909, 73.8278)
    d = haversine_distance(mumbai, goa)
    assert 380.0 < d < 450.0


def test_kdtree_exact_nearest_lookup():
    clusters = [
        FishCluster("C1", 15.0, 72.0, "Tuna", 45.2, 30.0, 28.5, 35.0, 0.92),
        FishCluster("C2", 16.0, 73.0, "Mackerel", 80.0, 15.0, 29.1, 34.5, 0.88),
        FishCluster("C3", 18.0, 72.5, "Sardine", 120.5, 10.0, 27.8, 35.2, 0.95),
        FishCluster("C4", 12.0, 74.0, "Anchovy", 30.0, 50.0, 26.5, 36.0, 0.85)
    ]
    
    spatial_index = FishClusterSpatialIndex(clusters)
    
    # Query very close to C3 (18.0, 72.5)
    res = spatial_index.find_nearest_cluster(18.01, 72.51)
    assert res["nearest_cluster"]["cluster_id"] == "C3"
    assert res["distance_km"] < 5.0


def test_kdtree_k_nearest():
    points = [((float(i), float(i)), f"P{i}") for i in range(20)]
    tree = KDTree(points, dimensions=2, is_geospatial=False)
    
    knn = tree.k_nearest_neighbors((5.1, 5.1), k=3)
    assert len(knn) == 3
    assert knn[0]["payload"] == "P5"
    assert knn[1]["payload"] in ["P4", "P6"]


def test_kdtree_radius_search():
    clusters = [
        FishCluster("C1", 15.0, 72.0, "Tuna", 50.0, 20.0, 28.0, 35.0, 0.9),
        FishCluster("C2", 15.05, 72.05, "Tuna", 60.0, 25.0, 28.0, 35.0, 0.9),
        FishCluster("C3", 20.0, 80.0, "Sardine", 10.0, 10.0, 26.0, 34.0, 0.7)
    ]
    spatial_index = FishClusterSpatialIndex(clusters)
    
    # Radius search of 20 km around (15.0, 72.0)
    in_radius = spatial_index.find_clusters_in_zone(15.0, 72.0, radius_km=20.0)
    cluster_ids = [c["payload"]["cluster_id"] for c in in_radius]
    assert "C1" in cluster_ids
    assert "C2" in cluster_ids
    assert "C3" not in cluster_ids


def test_ocean_pathfinding_current_assistance():
    # 10x10 grid starting at lat 20.0, lon 70.0, spacing 0.1 deg (~6 NM / ~11 km)
    graph = OceanGridGraph(rows=10, cols=10, top_left=(20.0, 70.0), res_deg=0.1)

    start = (0, 0)
    goal = (0, 9)

    # Route with zero current
    base_res = a_star_pathfinding(graph, start, goal, vessel_speed_knots=10.0)
    assert base_res["success"] is True
    base_time = base_res["total_time_hours"]

    # Inject strong favorable tailwind/current (u = +5 knots Eastward) across the path
    for c in range(10):
        graph.set_current(0, c, u=5.0, v=0.0)

    assisted_res = a_star_pathfinding(graph, start, goal, vessel_speed_knots=10.0)
    assert assisted_res["success"] is True
    assisted_time = assisted_res["total_time_hours"]

    # Assisted time must be significantly less than baseline time
    assert assisted_time < base_time


def test_pathfinding_obstacle_avoidance():
    graph = OceanGridGraph(rows=5, cols=5, top_left=(15.0, 70.0), res_deg=0.1)
    start = (2, 0)
    goal = (2, 4)

    # Place a vertical wall of land blocks blocking direct path at col 2
    for r in range(1, 4):
        graph.set_obstacle(r, 2, is_land=True)

    res = a_star_pathfinding(graph, start, goal, vessel_speed_knots=12.0)
    assert res["success"] is True
    
    # Assert path stepped around obstacle
    for wp in res["waypoints"]:
        pos = wp["grid_pos"]
        assert not (1 <= pos[0] <= 3 and pos[1] == 2)