# continuous-control

bipedal_control_experiment/
├── algorithms/
│   ├── __init__.py
│   ├── ppo.py             # PPO Implementierung
│   ├── sac.py             # Soft Actor-Critic
│   ├── ddpg.py            # Deep Deterministic Policy Gradient
│   ├── td3.py             # Twin Delayed DDPG
│   └── dqn.py             # Q-Learning (diskretisierte Version)
├── utils/
│   ├── __init__.py
│   ├── logger.py          # Logging-Funktionalität
│   ├── plotting.py        # Visualisierung der Ergebnisse
│   └── evaluation.py      # Evaluierungsfunktionen
├── config/
│   ├── __init__.py
│   └── hyperparameters.py # Hyperparameter-Konfigurationen
├── experiments/
│   ├── __init__.py
│   ├── experiment.py      # Experiment-Klasse
│   └── compare.py         # Vergleichsfunktionen
├── results/               # Ergebnisse speichern
├── models/                # Trainierte Modelle speichern
├── main.py                # Hauptskript zum Starten des Experiments
└── requirements.txt       # Abhängigkeiten