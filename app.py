"""Streamlit interface for the EV Charging MARL simulation.

Run from the repository root:
    streamlit run app.py
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agents import QLearningAgent, baseline_action
from src.environment import EVChargingEnvironment

st.set_page_config(
    page_title="EV Charging MARL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A restrained dashboard theme; Streamlit still handles responsive layout.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1600px;}
      div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 14px 16px;
      }
      .eyebrow {font-size: .78rem; letter-spacing: .13em; text-transform: uppercase;
                opacity: .7; font-weight: 650;}
      .subtle {opacity: .72;}
      section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
    </style>
    """,
    unsafe_allow_html=True,
)

STRATEGIES = {
    "Rule-based baseline": "baseline",
    "Individual Q-Learning": "q_learning",
    "Shared-reward Q-Learning": "shared_reward",
}
STATUS_COLORS = {
    "idle": "#64748b",
    "traveling": "#f59e0b",
    "queued": "#ef4444",
    "charging": "#10b981",
}


@st.cache_resource(show_spinner=False)
def train_agents(
    mode: str,
    episodes: int,
    num_evs: int,
    max_steps: int,
    seed: int,
) -> list[QLearningAgent]:
    """Train independent tabular agents using the project's existing environment."""
    agents = [
        QLearningAgent(seed=seed + i, epsilon_decay=0.985)
        for i in range(num_evs)
    ]
    reward_mode = "shared" if mode == "shared_reward" else "individual"

    for episode in range(episodes):
        env = EVChargingEnvironment(
            num_evs=num_evs,
            seed=seed + episode,
            max_steps=max_steps,
        )
        states = [env.observe(i) for i in range(num_evs)]

        for _ in range(max_steps):
            actions: dict[int, int] = {}
            chosen: dict[int, tuple[Any, int]] = {}
            for i, ev in enumerate(env.evs):
                if ev.state == "idle" and ev.battery <= 40:
                    state = states[i]
                    action = agents[i].choose_action(state, explore=True)
                    actions[i] = action
                    chosen[i] = (state, action)

            next_states, rewards, done, _ = env.step(
                actions, reward_mode=reward_mode
            )
            for i, (state, action) in chosen.items():
                agents[i].update(
                    state, action, float(rewards[i]), next_states[i], done
                )
            states = next_states
            if done:
                break

        for agent in agents:
            agent.decay_exploration()

    # Evaluation is greedy after training.
    for agent in agents:
        agent.epsilon = 0.0
    return agents


def initialise_simulation(
    strategy_label: str,
    num_evs: int,
    max_steps: int,
    seed: int,
    training_episodes: int,
) -> None:
    mode = STRATEGIES[strategy_label]
    learned_agents = None

    if mode != "baseline":
        with st.spinner(
            f"Training {strategy_label} for {training_episodes} episodes…"
        ):
            learned_agents = train_agents(
                mode, training_episodes, num_evs, max_steps, seed
            )
            # Each simulation gets its own mutable Q tables.
            learned_agents = copy.deepcopy(learned_agents)

    env = EVChargingEnvironment(
        num_evs=num_evs, seed=seed, max_steps=max_steps
    )
    st.session_state.env = env
    st.session_state.agents = learned_agents
    st.session_state.strategy_label = strategy_label
    st.session_state.strategy_mode = mode
    st.session_state.playing = False
    st.session_state.reward_history = []
    st.session_state.metric_history = []
    st.session_state.has_simulation = True


def advance_one_step() -> None:
    env: EVChargingEnvironment = st.session_state.env
    mode: str = st.session_state.strategy_mode
    agents = st.session_state.agents
    states = [env.observe(i) for i in range(len(env.evs))]
    actions: dict[int, int] = {}

    for i, ev in enumerate(env.evs):
        if ev.state == "idle" and ev.battery <= 40:
            if mode == "baseline":
                actions[i] = baseline_action(env, i)
            else:
                actions[i] = agents[i].choose_action(states[i], explore=False)

    reward_mode = "shared" if mode == "shared_reward" else "individual"
    _, rewards, done, _ = env.step(actions, reward_mode=reward_mode)
    st.session_state.reward_history.append(float(np.mean(rewards)))
    summary = env.summary()
    st.session_state.metric_history.append(
        {
            "Step": summary["steps"],
            "Average battery (%)": summary["average_battery"],
            "Total cost": summary["total_cost"],
            "Average wait (steps/EV)": summary["average_wait_steps"],
            "Completed charges": summary["completed_charges"],
            "Mean reward": float(np.mean(rewards)),
            "Total queued or charging": int(sum(summary["queue_lengths"])),
        }
    )
    if done:
        st.session_state.playing = False


def build_city_figure(env: EVChargingEnvironment) -> go.Figure:
    """Render a stylised top-down city whose streets match the environment road graph."""
    fig = go.Figure()

    # City blocks: alternate green spaces and low-rise urban blocks.
    block_colors = ["#e7efe7", "#e8edf0", "#edf0e5", "#e8ece7"]
    xs, ys = env.road_x, env.road_y
    for row in range(len(ys) - 1):
        for col in range(len(xs) - 1):
            x0, x1 = xs[col] + 1.8, xs[col + 1] - 1.8
            y0, y1 = ys[row] + 1.8, ys[row + 1] - 1.8
            color = block_colors[(row * 3 + col) % len(block_colors)]
            # Building/park blocks remain below the street layers.
            fig.add_shape(
                type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="rgba(71,85,105,0.12)", width=1),
                fillcolor=color, layer="below",
            )
            if (row + col) % 3 == 0:
                # Small building footprints add a little visual scale to the map.
                bx0, bx1 = x0 + 3, x1 - 3
                by0, by1 = y0 + 3, y1 - 3
                fig.add_shape(
                    type="rect", x0=bx0, y0=by0, x1=bx1, y1=by1,
                    line=dict(color="rgba(100,116,139,0.12)", width=1),
                    fillcolor="#d7e0e2", layer="below",
                )

    # Road surfaces and lane markings follow the exact road graph used by the environment.
    for x in env.road_x:
        fig.add_shape(
            type="line", x0=x, y0=5, x1=x, y1=95,
            line=dict(color="#94a3a8", width=11), layer="below",
        )
        fig.add_shape(
            type="line", x0=x, y0=5, x1=x, y1=95,
            line=dict(color="#f8fafc", width=1.2, dash="dash"), layer="below",
        )
    for y in env.road_y:
        fig.add_shape(
            type="line", x0=5, y0=y, x1=95, y1=y,
            line=dict(color="#94a3a8", width=11), layer="below",
        )
        fig.add_shape(
            type="line", x0=5, y0=y, x1=95, y1=y,
            line=dict(color="#f8fafc", width=1.2, dash="dash"), layer="below",
        )

    # Highlight each active route as a clean solid route overlay on top of the roads.
    for ev in env.evs:
        if ev.state == "traveling" and ev.route:
            remaining_route = [(ev.x, ev.y)] + ev.route[ev.route_index:]
            if len(remaining_route) >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=[p[0] for p in remaining_route],
                        y=[p[1] for p in remaining_route],
                        mode="lines",
                        line=dict(color="rgba(245,158,11,0.8)", width=3),
                        hoverinfo="skip", showlegend=False,
                    )
                )

    # Charging station symbols and labels.
    station_x = [s.x for s in env.stations]
    station_y = [s.y for s in env.stations]
    station_text = [
        f"Station {s.station_id + 1}<br>Price: {s.price:.2f} / unit"
        f"<br>Queue: {len(s.queue)}"
        f"<br>Charging: {'Yes' if s.charging_ev is not None else 'No'}"
        for s in env.stations
    ]
    fig.add_trace(
        go.Scatter(
            x=station_x, y=station_y, mode="markers+text",
            text=[f"CHARGE {s.station_id + 1}" for s in env.stations],
            textposition="top center",
            textfont=dict(size=10, color="#134e4a", family="Inter, sans-serif"),
            marker=dict(
                symbol="square", size=28, color="#0f766e",
                line=dict(color="#ccfbf1", width=3),
            ),
            customdata=station_text,
            hovertemplate="%{customdata}<extra>Charging station</extra>",
            name="Charging stations",
        )
    )

    ev_colors = [STATUS_COLORS.get(e.state, "#64748b") for e in env.evs]
    customdata = [
        [
            f"EV {e.ev_id + 1}", e.state.title(), f"{e.battery:.1f}%",
            f"Station {e.target_station + 1}" if e.target_station is not None else "None",
            e.wait_steps, f"{e.total_cost:.2f}",
        ]
        for e in env.evs
    ]
    fig.add_trace(
        go.Scatter(
            x=[e.x for e in env.evs], y=[e.y for e in env.evs],
            mode="markers+text",
            text=[f"⚡{e.ev_id + 1}" for e in env.evs],
            textposition="top center",
            textfont=dict(size=10, color="#0f172a", family="Inter, sans-serif"),
            marker=dict(
                size=15, color=ev_colors, symbol="circle",
                line=dict(color="#ffffff", width=2),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Status: %{customdata[1]}"
                "<br>Battery: %{customdata[2]}<br>Target: %{customdata[3]}"
                "<br>Wait steps: %{customdata[4]}<br>Accumulated cost: %{customdata[5]}"
                "<extra></extra>"
            ),
            name="Electric vehicles",
        )
    )

    fig.update_layout(
        height=650,
        margin=dict(l=8, r=8, t=18, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f1f5f2",
        font=dict(family="Inter, sans-serif", size=11, color="#334155"),
        xaxis=dict(range=[3, 97], visible=False, showgrid=False, zeroline=False),
        yaxis=dict(
            range=[3, 97], visible=False, showgrid=False, zeroline=False,
            scaleanchor="x", scaleratio=1,
        ),
        showlegend=False,
        hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#f8fafc")),
    )
    return fig


def build_history_figure(history: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    if history:
        frame = pd.DataFrame(history)
        fig.add_trace(
            go.Scatter(
                x=frame["Step"], y=frame["Average battery (%)"],
                mode="lines", name="Average battery",
                line=dict(color="#0d9488", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["Step"], y=frame["Total queued or charging"],
                mode="lines", name="Queue + active",
                yaxis="y2", line=dict(color="#f59e0b", width=2, dash="dot"),
            )
        )
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(128,128,128,0.035)",
        xaxis_title="Simulation step",
        yaxis=dict(title="Average battery (%)", rangemode="tozero"),
        yaxis2=dict(title="EVs queued / charging", overlaying="y", side="right", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


st.markdown('<div class="eyebrow">Multi-agent reinforcement learning · Interactive lab</div>', unsafe_allow_html=True)
st.title("EV Charging Intelligence")
st.markdown(
    "Explore how electric vehicles choose shared charging stations, "
    "trade off distance and congestion, and learn from individual or shared rewards."
)

with st.sidebar:
    st.header("Simulation setup")
    strategy_label = st.selectbox("Decision strategy", list(STRATEGIES.keys()))
    num_evs = st.slider("Number of EVs", min_value=4, max_value=24, value=12, step=2)
    max_steps = st.slider("Episode length (steps)", min_value=60, max_value=300, value=180, step=30)
    seed = st.number_input("Random seed", min_value=0, max_value=100_000, value=42, step=1)
    training_episodes = st.slider(
        "Training episodes (RL only)", min_value=10, max_value=150, value=50, step=10,
        help="More episodes take longer to train but give the Q-learning agents more experience.",
    )
    start_clicked = st.button("Start / reset simulation", type="primary", use_container_width=True)
    st.caption("The locations, prices, and battery dynamics are synthetic demonstration settings.")

if start_clicked:
    initialise_simulation(
        strategy_label, num_evs, max_steps, int(seed), training_episodes
    )

if not st.session_state.get("has_simulation", False):
    st.markdown("### Get started")
    st.write(
        "Choose a strategy in the sidebar, then select **Start / reset simulation**. "
        "The baseline runs immediately; Q-Learning strategies train agents first."
    )
    st.info("The city view will show individual EVs moving, queuing, and charging as the simulation advances.")
    st.stop()

env: EVChargingEnvironment = st.session_state.env
summary = env.summary()
finished = summary["steps"] >= env.max_steps
status_label = "Completed" if finished else ("Running" if st.session_state.playing else "Paused")

top_cols = st.columns([2.3, 1, 1, 1])
with top_cols[0]:
    st.subheader(st.session_state.strategy_label)
    st.caption(f"Simulation status: **{status_label}** · Seed {env.seed} · {env.num_evs} EVs")
with top_cols[1]:
    st.metric("Simulation step", f"{summary['steps']} / {env.max_steps}")
with top_cols[2]:
    st.metric("Completed charges", summary["completed_charges"])
with top_cols[3]:
    st.metric("Total energy cost", f"{summary['total_cost']:.2f}")

st.divider()


@st.fragment(run_every=0.5)
def live_simulation() -> None:
    env_local: EVChargingEnvironment = st.session_state.env
    summary_local = env_local.summary()
    done_local = summary_local["steps"] >= env_local.max_steps
    controls = st.columns([1, 1, 1, 3])
    with controls[0]:
        if st.button(
            "Pause" if st.session_state.playing else "Play",
            type="primary", use_container_width=True,
            disabled=done_local,
            key="play_pause",
        ):
            st.session_state.playing = not st.session_state.playing
    with controls[1]:
        step_clicked = st.button("Step once", use_container_width=True, disabled=done_local, key="step_once")
    with controls[2]:
        run_to_end = st.button("Run to end", use_container_width=True, disabled=done_local, key="run_end")
    if run_to_end:
        st.session_state.playing = False
        # Run this episode to completion synchronously; the current state is retained.
        for _ in range(env_local.max_steps - env_local.time):
            advance_one_step()
    elif step_clicked:
        st.session_state.playing = False
        advance_one_step()
    elif st.session_state.playing and not done_local:
        advance_one_step()

    env_local = st.session_state.env
    summary_local = env_local.summary()
    st_cols = st.columns(4)
    st_cols[0].metric("Average battery", f"{summary_local['average_battery']:.1f}%")
    st_cols[1].metric("Average wait / EV", f"{summary_local['average_wait_steps']:.1f} steps")
    st_cols[2].metric("Vehicles travelling", sum(e.state == "traveling" for e in env_local.evs))
    st_cols[3].metric("Vehicles charging", sum(e.state == "charging" for e in env_local.evs))

    city_col, chart_col = st.columns([1.35, 1])
    with city_col:
        st.markdown("#### City simulation")
        st.plotly_chart(build_city_figure(env_local), use_container_width=True, key="city_map")
        legend_cols = st.columns(4)
        for col, (state, color) in zip(legend_cols, STATUS_COLORS.items()):
            col.markdown(
                f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
                f'background:{color};margin-right:6px"></span>{state.title()}',
                unsafe_allow_html=True,
            )
    with chart_col:
        st.markdown("#### Live system metrics")
        st.plotly_chart(
            build_history_figure(st.session_state.metric_history),
            use_container_width=True, key="live_metrics",
        )
        st.markdown("#### Charging stations")
        rows = []
        for station in env_local.stations:
            rows.append({
                "Station": f"S{station.station_id}",
                "Price / unit": f"{station.price:.2f}",
                "Waiting": len(station.queue),
                "Charging": "Yes" if station.charging_ev is not None else "No",
                "Total demand": env_local.station_queue_size(station),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("#### EV status")
    ev_rows = []
    for ev in env_local.evs:
        ev_rows.append({
            "EV": f"EV {ev.ev_id}",
            "Status": ev.state.title(),
            "Battery (%)": round(ev.battery, 1),
            "Target station": f"S{ev.target_station}" if ev.target_station is not None else "—",
            "Wait steps": ev.wait_steps,
            "Completed charges": ev.completed_charges,
            "Cost": round(ev.total_cost, 2),
        })
    st.dataframe(pd.DataFrame(ev_rows), hide_index=True, use_container_width=True)

    if summary_local["steps"] >= env_local.max_steps:
        st.success("Episode complete. Use Start / reset simulation in the sidebar to run another episode.")
    elif st.session_state.playing:
        st.caption("Simulation is playing · updates approximately twice per second.")
    else:
        st.caption("Simulation is paused. Play, step through one tick, or run to the end.")

    if st.session_state.metric_history:
        history_csv = pd.DataFrame(st.session_state.metric_history).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download this run's metrics (CSV)",
            data=history_csv,
            file_name="ev_charging_simulation_metrics.csv",
            mime="text/csv",
            key="download_metrics",
        )


live_simulation()

st.divider()
with st.expander("About the model and interpretation"):
    st.markdown(
        """
        - **Rule-based baseline:** selects a station using distance, queue / active demand, and price.
        - **Individual Q-Learning:** each EV learns from its own reward signal.
        - **Shared-reward Q-Learning:** each EV receives the mean reward across EVs for that step.
        - **State:** discretised battery level, nearest-station distance, and station demand.
        - **Action:** choose one of the three charging stations when an idle EV's battery is at or below 40%.
        - **Reward trade-offs:** the environment penalises travel, queueing, and charging cost, while charging makes progress toward a battery target.
        
        Results depend on random seeds and training duration. These are synthetic simulation settings, not real-world charging predictions or a guarantee that a learned strategy is optimal.
        """
    )
