# EV Charging Multi-Agent Reinforcement Learning

An interactive simulation of electric vehicles selecting shared charging stations using a rule-based baseline and tabular Q-Learning variants.

## Features

- **Road-aware Streamlit city simulation** with a shared road graph: EVs route through intersections instead of cutting diagonally across city blocks. The road network in the visualization matches the movement logic in the environment.
- **Refined city visuals** with road surfaces, lane markings, urban blocks, station markers, route highlights, status colours, and EV hover details.
- **Three strategies:** rule-based baseline, individual Q-Learning, and shared-reward Q-Learning.
- **Play, pause, single-step, and run-to-end controls** for inspecting the simulation.
- **Live metrics:** battery levels, charging cost, completed charges, waiting time, station demand, and per-EV status.
- **Downloadable run metrics** in CSV format.
- Existing Matplotlib animation and experiment-comparison workflow remain available.

## Setup on macOS

Open this folder in VS Code and use Terminal → New Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, press `Cmd+Shift+P`, choose **Python: Select Interpreter**, and select `.venv/bin/python`.

## Launch the Streamlit app

Run from the repository root:

```bash
streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`. Open it in your browser.

Choose a strategy in the sidebar, set the number of EVs and episode length, then click **Start / reset simulation**. Q-Learning strategies train agents first; the training episode count controls how much experience they receive. Use Play or Step once to view the simulation, or Run to end to finish the current episode. The run's metrics can be downloaded as a CSV.

## Optional: existing Matplotlib animation

```bash
python src/main.py
```

## Run strategy experiments

```bash
python src/experiments.py
```

Experiment outputs are written to `outputs/comparison.csv` and `outputs/comparison.png`.

## Model overview

- EV agents share three charging stations.
- Each EV has a battery, position, state, waiting time, accumulated charging cost, and completed-charge count.
- State is discretised into a battery bin, nearest-station distance bin, and station demand bins.
- Available actions select one of the charging stations when an idle EV's battery is at or below the decision threshold.
- The baseline considers distance, queue / active demand, and electricity price.
- Individual Q-Learning uses each EV's reward; shared-reward Q-Learning provides each agent the mean reward for that time step.
- Rewards reflect travel, station demand, charging price, and queueing; charging provides positive reward.

## Notes and limitations

Locations, prices, and battery dynamics are synthetic demonstration settings, not real infrastructure data. Q-Learning results depend on the random seed and training duration; these experiments do not guarantee an optimal policy or real-world performance.
