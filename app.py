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
    fig = go.Figure()

    # A light coordinate grid makes motion and relative station placement clear.
    for axis_value in range(0, 101, 10):
        fig.add_shape(
            type="line", x0=axis_value, y0=0, x1=axis_value, y1=100,
            line=dict(color="rgba(148,163,184,0.13)", width=1), layer="below",
        )
        fig.add_shape(
            type="line", x0=0, y0=axis_value, x1=100, y1=axis_value,
            line=dict(color="rgba(148,163,184,0.13)", width=1), layer="below",
        )

    # Show travel intent as faint lines from travelling EVs to their target.
    for ev in env.evs:
        if ev.state == "traveling" and ev.target_station is not None:
            station = env.stations[ev.target_station]
            fig.add_trace(
                go.Scatter(
                    x=[ev.x, station.x], y=[ev.y, station.y],
                    mode="lines", line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    hoverinfo="skip", showlegend=False,
                )
            )

    station_x = [s.x for s in env.stations]
    station_y = [s.y for s in env.stations]
    station_text = [
        f"S{s.station_id} · ${s.price:.2f}/unit<br>"
        f"Queue + charging: {env.station_queue_size(s)}"
        for s in env.stations
    ]
    fig.add_trace(
        go.Scatter(
            x=station_x, y=station_y, mode="markers+text",
            text=[f"S{s.station_id}" for s in env.stations],
            textposition="top center",
            marker=dict(
                symbol="square", size=24, color="#0f766e",
                line=dict(color="#99f6e4", width=2),
            ),
            customdata=station_text,
            hovertemplate="%{customdata}<extra>Charging station</extra>",
            name="Charging stations",
        )
    )

    ev_x = [e.x for e in env.evs]
    ev_y = [e.y for e in env.evs]
    ev_status = [e.state for e in env.evs]
    ev_colors = [STATUS_COLORS.get(e.state, "#64748b") for e in env.evs]
    customdata = [
        [
            f"EV {e.ev_id}", e.state.title(), f"{e.battery:.1f}%",
            f"S{e.target_station}" if e.target_station is not None else "—",
            e.wait_steps, f"{e.total_cost:.2f}",
        ]
        for e in env.evs
    ]
    fig.add_trace(
        go.Scatter(
            x=ev_x, y=ev_y, mode="markers+text",
            text=[str(e.ev_id) for e in env.evs],
            textposition="middle center",
            textfont=dict(size=9, color="white"),
            marker=dict(
                size=17, color=ev_colors,
                line=dict(color="rgba(255,255,255,0.85)", width=1.2),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Status: %{customdata[1]}"
                "<br>Battery: %{customdata[2]}<br>Target: %{customdata[3]}"
                "<br>Waiting steps: %{customdata[4]}"
                "<br>Accumulated cost: %{customdata[5]}"
                "<extra></extra>"
            ),
            name="Electric vehicles",
        )
    )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(128,128,128,0.035)",
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(range=[0, 100], title="City X", showgrid=False, zeroline=False),
        yaxis=dict(
            range=[0, 100], title="City Y", showgrid=False, zeroline=False,
            scaleanchor="x", scaleratio=1,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        showlegend=False,
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
        if st.button("Step once", use_container_width=True, disabled=done_local, key="step_once"):
            advance_one_step()
    with controls[2]:
        if st.button("Run to end", use_container_width=True, disabled=done_local, key="run_end"):
            st.session_state.playing = False
            # Run this episode to completion synchronously; the current state is retained.
            for _ in range(env_local.max_steps - env_local.time):
                advance_one_step()
    with controls[3]:
        st.caption("Play advances the city automatically. Step once is useful for observing a single decision cycle.")

    if st.session_state.playing and not done_local:
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
