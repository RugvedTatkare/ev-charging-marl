"""Train Q-Learning variants and compare them with the baseline."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import EVChargingEnvironment
from agents import QLearningAgent, baseline_action

NUM_EVS, EPISODES, MAX_STEPS = 12, 100, 180


def run_episode(mode, seed, agents=None, train=True):
    env = EVChargingEnvironment(NUM_EVS, seed=seed, max_steps=MAX_STEPS)
    states = [env.observe(i) for i in range(NUM_EVS)]
    rewards_by_step = []
    for _ in range(MAX_STEPS):
        actions, chosen = {}, {}
        for i, ev in enumerate(env.evs):
            if ev.state == "idle" and ev.battery <= 40:
                action = baseline_action(env, i) if mode == "baseline" else agents[i].choose_action(states[i], explore=train)
                actions[i] = action
                chosen[i] = action
        next_states, rewards, done, _ = env.step(
            actions, reward_mode="shared" if mode == "shared_reward" else "individual"
        )
        if mode != "baseline" and train:
            for i, action in chosen.items():
                agents[i].update(states[i], action, float(rewards[i]), next_states[i], done)
        states = next_states
        rewards_by_step.append(float(np.mean(rewards)))
        if done:
            break
    return {
        "mode": mode, "seed": seed,
        "total_reward": float(np.sum(rewards_by_step)),
        "average_reward_per_step": float(np.mean(rewards_by_step)),
        "total_cost": float(sum(e.total_cost for e in env.evs)),
        "average_wait_steps_per_ev": float(np.mean([e.wait_steps for e in env.evs])),
        "completed_charges": int(sum(e.completed_charges for e in env.evs)),
        "unique_q_states": int(sum(len(a.q_table) for a in agents)) if agents else 0,
    }


def train_variant(mode, base_seed):
    agents = [QLearningAgent(seed=base_seed + i, epsilon_decay=0.985) for i in range(NUM_EVS)]
    for episode in range(EPISODES):
        run_episode(mode, base_seed + episode, agents=agents, train=True)
        for agent in agents:
            agent.decay_exploration()
    evaluation = [run_episode(mode, base_seed + 10_000 + i, agents=agents, train=False) for i in range(10)]
    return evaluation


def main():
    os.makedirs("outputs", exist_ok=True)
    results = []
    print(f"Training {EPISODES} episodes per learning configuration...")
    for mode in ("baseline", "q_learning", "shared_reward"):
        if mode == "baseline":
            evaluation = [run_episode(mode, 20_000 + i, train=False) for i in range(10)]
        else:
            evaluation = train_variant(mode, 100 if mode == "q_learning" else 500)
        results.extend(evaluation)
        print(f"Finished: {mode}")
    df = pd.DataFrame(results)
    df.to_csv("outputs/comparison.csv", index=False)
    summary = df.groupby("mode").agg(
        total_cost_mean=("total_cost", "mean"),
        wait_steps_mean=("average_wait_steps_per_ev", "mean"),
        completed_charges_mean=("completed_charges", "mean"),
        reward_mean=("average_reward_per_step", "mean"),
    ).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, col, title in zip(
        axes,
        ["total_cost_mean", "wait_steps_mean", "completed_charges_mean"],
        ["Mean total charging cost", "Mean wait steps per EV", "Mean completed charges"],
    ):
        ax.bar(summary["mode"], summary[col])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("EV charging strategy comparison (10 evaluation seeds)")
    fig.tight_layout()
    fig.savefig("outputs/comparison.png", dpi=160, bbox_inches="tight")
    print("\nEvaluation summary:")
    print(summary.to_string(index=False))
    print("\nSaved outputs/comparison.csv and outputs/comparison.png")


if __name__ == "__main__":
    main()
