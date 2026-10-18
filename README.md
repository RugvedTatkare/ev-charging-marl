# EV Charging Multi-Agent Reinforcement Learning

A Python simulation of electric vehicles choosing shared charging stations. It includes a rule-based baseline, independent tabular Q-Learning, a shared-reward variant, an animated city, and experiment graphs.

## Setup on macOS
Open this folder in VS Code and use Terminal → New Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, press `Cmd+Shift+P`, choose **Python: Select Interpreter**, and select `.venv/bin/python`.

## Run animation
From the project root:

```bash
python src/main.py
```

## Train and compare strategies
```bash
python src/experiments.py
```

Outputs are written to `outputs/comparison.csv` and `outputs/comparison.png`.

## Model
- 12 EV agents share 3 stations.
- EVs have battery, location, charging state, waiting time, and charging cost.
- State is discretized into battery bin, nearest-station distance bin, and each station's queue-size bin.
- Actions select one of three stations.
- Reward penalizes travel distance, congestion, and electricity cost; charging yields positive reward and queueing incurs a waiting penalty.
- The learning update is Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)].
- The shared-reward experiment gives all agents the mean reward from the current step.

All locations, prices, and battery dynamics are synthetic demonstration settings, not real infrastructure data. Independent learners in a shared environment are a practical baseline and do not guarantee optimal coordination.
