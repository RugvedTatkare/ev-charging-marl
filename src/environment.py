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
        self.stations = [
            ChargingStation(0, 20, 25, 1.0),
            ChargingStation(1, 75, 30, 1.5),
            ChargingStation(2, 50, 80, 0.8),
        ]
        self.evs = [
            EV(i, float(self.rng.uniform(5, 95)),
               float(self.rng.uniform(5, 95)),
               float(self.rng.uniform(15, 65)))
            for i in range(num_evs)
        ]
        self.average_battery_history = []
        self.total_cost_history = []
        self.queue_history = [[] for _ in self.stations]
        self.wait_history = []
        self.rewards = []

    @staticmethod
    def distance_xy(x, y, station):
        return float(np.hypot(x - station.x, y - station.y))

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
                ev.x, ev.y = float(self.rng.uniform(5, 95)), float(self.rng.uniform(5, 95))

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
            ev.travel_distance += distance
            rewards[ev_id] -= 0.015 * distance + 0.35 * queue_size + 0.15 * station.price

        # Travel to the selected station.
        speed = 5.0
        for ev in self.evs:
            if ev.state != "traveling" or ev.target_station is None:
                continue
            station = self.stations[ev.target_station]
            dx, dy = station.x - ev.x, station.y - ev.y
            distance = float(np.hypot(dx, dy))
            if distance <= speed:
                ev.x, ev.y = station.x, station.y
                ev.state = "queued"
                station.queue.append(ev.ev_id)
            elif distance > 0:
                ev.x += speed * dx / distance
                ev.y += speed * dy / distance

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
