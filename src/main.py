"""Launch the animated baseline simulation."""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from environment import EVChargingEnvironment
from agents import baseline_action

NUM_EVS, MAX_STEPS = 12, 250
env = EVChargingEnvironment(num_evs=NUM_EVS, seed=42, max_steps=MAX_STEPS)
fig, (city_ax, graph_ax) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("EV Charging Multi-Agent Simulation — Baseline")

city_ax.set(xlim=(0, env.width), ylim=(0, env.height), aspect="equal",
            title="Shared city environment", xlabel="X position", ylabel="Y position")
city_ax.grid(alpha=0.2)
city_ax.scatter([s.x for s in env.stations], [s.y for s in env.stations],
                marker="s", s=220, label="Charging stations")
for s in env.stations:
    city_ax.annotate(f"S{s.station_id}\n${s.price:.1f}/unit", (s.x, s.y),
                     xytext=(6, 8), textcoords="offset points", fontsize=8)
ev_scatter = city_ax.scatter([e.x for e in env.evs], [e.y for e in env.evs], s=65)
status_text = city_ax.text(0.02, 0.98, "", transform=city_ax.transAxes, va="top", fontsize=9)
city_ax.legend(loc="lower left")

graph_ax.set(title="Live metrics", xlabel="Simulation step", ylabel="Average battery (%)", ylim=(0, 105))
graph_ax.grid(alpha=0.3)
battery_line, = graph_ax.plot([], [], label="Average battery")
queue_ax = graph_ax.twinx()
queue_ax.set_ylabel("EVs queued or charging")
queue_ax.set_ylim(0, NUM_EVS)
queue_lines = []
for s in env.stations:
    line, = queue_ax.plot([], [], linestyle="--", label=f"Station {s.station_id} queue")
    queue_lines.append(line)
graph_ax.legend(loc="upper left")
queue_ax.legend(loc="upper right")
colors = {"idle": "tab:blue", "traveling": "tab:orange", "queued": "tab:red", "charging": "tab:green"}


def update(_frame):
    actions = {e.ev_id: baseline_action(env, e.ev_id)
               for e in env.evs if e.state == "idle" and e.battery <= 40}
    env.step(actions)
    ev_scatter.set_offsets([[e.x, e.y] for e in env.evs])
    ev_scatter.set_color([colors[e.state] for e in env.evs])
    info = env.summary()
    status_text.set_text(f"Step: {info['steps']}\nAvg battery: {info['average_battery']}%\n"
                         f"Total cost: {info['total_cost']:.1f}\nQueues: {info['queue_lengths']}")
    steps = range(1, env.time + 1)
    battery_line.set_data(steps, env.average_battery_history)
    for i, line in enumerate(queue_lines):
        line.set_data(steps, env.queue_history[i])
    graph_ax.set_xlim(0, max(30, env.time + 1))
    queue_ax.set_xlim(0, max(30, env.time + 1))
    return [ev_scatter, status_text, battery_line, *queue_lines]


ani = FuncAnimation(fig, update, frames=MAX_STEPS, interval=350,
                    blit=False, repeat=False, cache_frame_data=False)
plt.tight_layout()
plt.show()
print("\nFinal simulation summary:")
print(env.summary())
