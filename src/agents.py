"""Tabular Q-Learning agents and rule-based baseline."""
import numpy as np


class QLearningAgent:
    def __init__(self, n_actions=3, alpha=0.15, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.985, seed=0):
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min = epsilon, epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = {}

    def values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=float)
        return self.q_table[state]

    def choose_action(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.values(state)))

    def update(self, state, action, reward, next_state, done):
        old_value = self.values(state)[action]
        future = 0.0 if done else float(np.max(self.values(next_state)))
        target = reward + self.gamma * future
        self.q_table[state][action] += self.alpha * (target - old_value)

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def baseline_action(env, ev_id):
    """Select a station using distance, queue size and electricity price."""
    ev = env.evs[ev_id]
    scores = []
    for station in env.stations:
        distance = env.distance_xy(ev.x, ev.y, station)
        queue = env.station_queue_size(station)
        scores.append(distance + 8 * queue + 5 * station.price)
    return int(np.argmin(scores))
