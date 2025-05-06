# Continuous Control in Reinforcement Learning

This repository contains an implementation and comparison of various Deep Reinforcement Learning algorithms for continuous control tasks, specifically using the **BipedalWalker-v3** environment from OpenAI Gym.

## Project Description

This project implements and evaluates three prominent algorithms for continuous control in Reinforcement Learning:

- **Deep Deterministic Policy Gradient (DDPG)**
- **Soft Actor-Critic (SAC)**
- **Proximal Policy Optimization (PPO)**

These algorithms are tested in the **BipedalWalker-v3** environment, a complex continuous control task where a bipedal robot must learn to walk efficiently without falling.

## Environment: BipedalWalker-v3

![BipedalWalker-v3](Latex/BipdalWalker-1.jpg)

**Features:**

- **State space:** 24 dimensions (Position, joint angles, velocities, ground contact)
- **Action space:** 4 continuous dimensions \[-1, 1\] (Torques on hips & knees)
- **Reward system:**
  - + for forward movement
  - - for energy consumption and unnatural movements
  - +300 for reaching the goal
  - Penalty for falling

**Challenges:**

- Dynamic balance
- Sparse rewards
- Long-term planning
- High dimensionality

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/LeonardKreil/continuous-control.git
cd continuous-control

# Create Python environment
conda create --name continuous-control python=3.8
conda activate continuous-control

# Install dependencies
pip install -r requirements.txt
pip install gym[box2d] 
 
```

## 📦 Contents of requirements.txt

The following core packages are included in the requirements.txt:
- gymnasium==0.28.1
- numpy==1.24.3
- torch==2.0.1
- matplotlib==3.7.1
- pandas==2.0.2
- stable-baselines3==2.1.0
- seaborn==0.12.2

## 📁 Project Structure
```bash
bipedal_control_experiment/
├── algorithms/
│   ├── ppo.py             # PPO implementation
│   ├── sac.py             # Soft Actor-Critic
│   ├── ddpg.py            # Deep Deterministic Policy Gradient
│   └── td3.py             # Twin Delayed DDPG (optional)
├── utils/
│   ├── logger.py          # Logging
│   ├── plotting.py        # Visualization
│   └── evaluation.py      # Evaluation
├── config/
│   ├── hyperparameters.py # Configuration
│   ├── callbacks.py
│   ├── evaluation.py
│   ├── logger.py
│   ├── multi_plot.py
│   ├── plot.py
│   └── plotting.py
├── experiments/
│   ├── experiment.py      # Experiment classes
│   └── compare.py         # Comparison functions
├── results/               # Results
├── models/                # Trained models
├── main.py                # Main script
└── requirements.txt       # Dependencies
```

## ▶️ Usage
```bash
python main.py
```
### Features:
- Selection of DDPG, SAC, PPO
- Configuration via config/
- Training and evaluation
- Visualization of results

## ⚖️ Algorithm Comparison

### 🧠 DDPG (Deep Deterministic Policy Gradient)
- Deterministic policy
- Actor-Critic architecture
- Ornstein-Uhlenbeck process for exploration
- Stabilized via replay buffer & target networks
- ❗️ Sensitive to hyperparameters

### 🔥 SAC (Soft Actor-Critic)
- Maximum-entropy framework
- Stochastic policy (mean & standard deviation)
- Two Q-networks to avoid overestimation
- Adaptive entropy regularization
- ✅ High stability & efficiency

### 📈 PPO (Proximal Policy Optimization)
- Clipping mechanism for stable updates
- On-policy learning
- Generalized Advantage Estimation
- ✅ Simple & efficient
- ❗️ Less efficient than off-policy methods

## 📊 Experimental Results

| Algorithm | Convergence Speed       | Max Reward | Training Time |
|-----------|--------------------------|------------|---------------|
| SAC       | High (~200k steps)       | ~280       | 5.50 hours    |
| PPO       | Medium (~300k steps)     | ~225       | 0.35 hours    |
| DDPG      | Low (>400k steps)        | ~0         | 1.25 hours    |

### Key Insights:

- SAC delivers the best performance and convergence
- PPO is efficient with low computational cost
- DDPG requires careful hyperparameter tuning

## 🔍 Theoretical Explanations

- SAC vs. DDPG: Entropy encourages exploration and prevents early convergence
- SAC vs. PPO: Off-policy learning enables data reuse
- DDPG vs. PPO: PPO provides more stable updates via clipping

## 🔭 Future Research Directions

- Model-based RL (e.g., dynamics learning)
- Hierarchical RL (task decomposition)
- Multi-task & transfer learning
- Incorporation of prior knowledge (e.g., physical models, demonstrations)
