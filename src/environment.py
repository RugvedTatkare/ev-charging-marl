"""Shared EV charging environment for multi-agent reinforcement learning."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class EV:
    ev_id: int
    x: float
    y: float
    battery: float
    state: str = "idle"  # idle, traveling, queued, charging
    target_station: Optional[int] = None
    wait_steps: int = 0
    travel_distance: float = 0.0
    total_cost: float = 0.0
    completed_charges: int = 0
    route: list[tuple[float, float]] = field(default_factory=list)
    route_index: int = 0


@dataclass
class ChargingStation:
    station_id: int
    x: float
    y: float
    price: float
    charge_rate: float = 12.0
    queue: list[int] = field(default_factory=list)
    charging_ev: Optional[int] = None


class EVChargingEnvironment:
    def __init__(self, num_evs=12, seed=42, max_steps=250):
        self.num_evs = num_evs
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.width, self.height = 100.0, 100.0
        self.max_steps = max_steps
        self.time = 0
        # The map is a small road grid. Stations sit directly on intersections,
        # so route planning and the rendered streets use the same geometry.
        self.road_x = [10.0, 30.0, 50.0, 70.0, 90.0]
        self.road_y = [10.0, 30.0, 50.0, 70.0, 90.0]
        self.road_nodes = [(x, y) for y in self.road_y for x in self.road_x]
        self.road_graph = {node: [] for node in self.road_nodes}
        for y in self.road_y:
            for left, right in zip(self.road_x[:-1], self.road_x[1:]):
                self.road_graph[(left, y)].append(((right, y), right - left))
                self.road_graph[(right, y)].append(((left, y), right - left))
        for x in self.road_x:
            for lower, upper in zip(self.road_y[:-1], self.road_y[1:]):
                self.road_graph[(x, lower)].append(((x, upper), upper - lower))
                self.road_graph[(x, upper)].append(((x, lower), upper - lower))

        self.stations = [
            ChargingStation(0, 30, 30, 1.0),
            ChargingStation(1, 70, 30, 1.5),
            ChargingStation(2, 50, 90, 0.8),
        ]
        self.evs = []
        road_edges = []
        for node, edges in self.road_graph.items():
            for neighbor, _ in edges:
                if node < neighbor:
                    road_edges.append((node, neighbor))
        for i in range(num_evs):
            a, b = road_edges[int(self.rng.integers(0, len(road_edges)))]
            fraction = float(self.rng.uniform(0.12, 0.88))
            x = a[0] + (b[0] - a[0]) * fraction
            y = a[1] + (b[1] - a[1]) * fraction
            self.evs.append(EV(i, x, y, float(self.rng.uniform(15, 65))))
        self.average_battery_history = []
        self.total_cost_history = []
        self.queue_history = [[] for _ in self.stations]
        self.wait_history = []
        self.rewards = []

    @staticmethod
    def distance_xy(x, y, station):
        return float(np.hypot(x - station.x, y - station.y))

    def _nearest_road_node(self, x, y):
        return min(self.road_nodes, key=lambda node: np.hypot(node[0] - x, node[1] - y))

    def _shortest_road_path(self, start, goal):
        # Dijkstra over the small road graph; returns ordered intersections.
        distances = {node: float("inf") for node in self.road_nodes}
        previous = {}
        distances[start] = 0.0
        pending = set(self.road_nodes)
        while pending:
            current = min(pending, key=lambda node: distances[node])
            pending.remove(current)
            if current == goal or distances[current] == float("inf"):
                break
            for neighbor, weight in self.road_graph[current]:
                candidate = distances[current] + weight
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    previous[neighbor] = current
        path = [goal]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path.reverse()
        return path

    def _assign_road_route(self, ev, station):
        start = self._nearest_road_node(ev.x, ev.y)
        goal = (float(station.x), float(station.y))
        path = self._shortest_road_path(start, goal)
        ev.route = [(float(ev.x), float(ev.y))] + path
        ev.route_index = 1
        ev.travel_distance += float(np.hypot(start[0] - ev.x, start[1] - ev.y))
        ev.travel_distance += sum(
            float(np.hypot(b[0] - a[0], b[1] - a[1]))
            for a, b in zip(path[:-1], path[1:])
        )

    def station_queue_size(self, station):
        return len(station.queue) + int(station.charging_ev is not None)

    def observe(self, ev_id):
        ev = self.evs[ev_id]
        nearest = min(self.distance_xy(ev.x, ev.y, s) for s in self.stations)
        battery_bin = min(4, int(ev.battery // 20))
        distance_bin = min(3, int(nearest // 25))
        queues = [min(3, self.station_queue_size(s)) for s in self.stations]
        return (battery_bin, distance_bin, queues[0], queues[1], queues[2])

    def step(self, actions, reward_mode="individual"):
        """Advance one time tick; actions maps idle EV ids to station indices."""
        self.time += 1
        rewards = np.zeros(self.num_evs, dtype=float)

        # Non-charging EVs consume battery.
        for ev in self.evs:
            if ev.state != "charging":
                ev.battery = max(0.0, ev.battery - 0.25)

        # Charge EVs already connected.
        for station in self.stations:
            if station.charging_ev is None:
                continue
            ev = self.evs[station.charging_ev]
            amount = min(station.charge_rate, 100.0 - ev.battery)
            ev.battery += amount
            ev.total_cost += amount * station.price
            rewards[ev.ev_id] += 2.0 - 0.03 * amount * station.price
            if ev.battery >= 80.0:
                ev.state = "idle"
                ev.target_station = None
                ev.completed_charges += 1
                station.charging_ev = None
                # After charging, spawn the EV on a road segment again.
                road_edges = [(node, neighbor) for node, edges in self.road_graph.items()
                              for neighbor, _ in edges if node < neighbor]
                a, b = road_edges[int(self.rng.integers(0, len(road_edges)))]
                fraction = float(self.rng.uniform(0.12, 0.88))
                ev.x = a[0] + (b[0] - a[0]) * fraction
                ev.y = a[1] + (b[1] - a[1]) * fraction
                ev.route = []
                ev.route_index = 0

        # Fill free chargers from their queues.
        for station in self.stations:
            if station.charging_ev is None and station.queue:
                ev_id = station.queue.pop(0)
                station.charging_ev = ev_id
                self.evs[ev_id].state = "charging"

        # Apply decisions for EVs that need charging.
        for ev_id, action in actions.items():
            ev = self.evs[ev_id]
            if ev.state != "idle" or ev.battery > 40 or not 0 <= action < len(self.stations):
                continue
            station = self.stations[action]
            distance = self.distance_xy(ev.x, ev.y, station)
            queue_size = self.station_queue_size(station)
            ev.target_station = action
            ev.state = "traveling"
            self._assign_road_route(ev, station)
            rewards[ev_id] -= 0.015 * distance + 0.35 * queue_size + 0.15 * station.price

        # Follow the same road graph drawn in the Streamlit map.
        speed = 5.0
        for ev in self.evs:
            if ev.state != "traveling" or ev.target_station is None:
                continue
            remaining = speed
            while remaining > 0 and ev.route_index < len(ev.route):
                target_x, target_y = ev.route[ev.route_index]
                dx, dy = target_x - ev.x, target_y - ev.y
                distance = float(np.hypot(dx, dy))
                if distance <= remaining + 1e-9:
                    ev.x, ev.y = target_x, target_y
                    remaining -= distance
                    ev.route_index += 1
                elif distance > 0:
                    ev.x += remaining * dx / distance
                    ev.y += remaining * dy / distance
                    remaining = 0
            if ev.route_index >= len(ev.route):
                station = self.stations[ev.target_station]
                ev.x, ev.y = station.x, station.y
                ev.state = "queued"
                ev.route = []
                ev.route_index = 0
                station.queue.append(ev.ev_id)

        # Queue penalty.
        for station in self.stations:
            for ev_id in station.queue:
                self.evs[ev_id].wait_steps += 1
                rewards[ev_id] -= 0.2

        self.average_battery_history.append(float(np.mean([e.battery for e in self.evs])))
        self.total_cost_history.append(float(sum(e.total_cost for e in self.evs)))
        self.wait_history.append(float(np.mean([e.wait_steps for e in self.evs])))
        self.rewards.append(float(np.mean(rewards)))
        for i, station in enumerate(self.stations):
            self.queue_history[i].append(self.station_queue_size(station))

        if reward_mode == "shared":
            rewards[:] = float(np.mean(rewards))

        next_states = [self.observe(i) for i in range(self.num_evs)]
        done = self.time >= self.max_steps
        info = {
            "average_battery": self.average_battery_history[-1],
            "total_cost": self.total_cost_history[-1],
            "average_wait_steps": self.wait_history[-1],
            "completed_charges": sum(e.completed_charges for e in self.evs),
            "queue_lengths": [self.station_queue_size(s) for s in self.stations],
        }
        return next_states, rewards, done, info

    def summary(self):
        return {
            "steps": self.time,
            "average_battery": round(float(np.mean([e.battery for e in self.evs])), 2),
            "total_cost": round(float(sum(e.total_cost for e in self.evs)), 2),
            "average_wait_steps": round(float(np.mean([e.wait_steps for e in self.evs])), 2),
            "completed_charges": sum(e.completed_charges for e in self.evs),
            "queue_lengths": [self.station_queue_size(s) for s in self.stations],
        }
