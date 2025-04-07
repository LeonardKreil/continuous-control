import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config.hyperparameters import get_all_configs
from experiments.experiment import Experiment
from utils.plotting import plot_comparison

def compare_algorithms(env_id="BipedalWalker-v3", total_timesteps=1000000, 
                      eval_freq=10000, seed=42, n_eval_episodes=10):
    """
    Trainiert und vergleicht alle Algorithmen auf der angegebenen Umgebung.
    """
    # Hole alle Konfigurationen
    configs = get_all_configs()
    results = {}
    
    # Trainiere jeden Algorithmus
    for algo_name, config in configs.items():
        print(f"=== Training {algo_name} ===")
        experiment = Experiment(
            algo_name=algo_name,
            env_id=env_id,
            config=config,
            seed=seed
        )
        
        # Training
        model_path = experiment.train(total_timesteps=total_timesteps, eval_freq=eval_freq)
        
        # Evaluation
        mean_reward, std_reward = experiment.evaluate(n_eval_episodes=n_eval_episodes)
        
        results[algo_name] = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "model_path": model_path
        }
    
    # Ergebnisse als DataFrame
    results_df = pd.DataFrame({
        algo: [results[algo]["mean_reward"]] for algo in results.keys()
    })
    
    # Speichere Ergebnisse
    results_path = f"./results/comparison_{env_id}_{seed}.csv"
    results_df.to_csv(results_path, index=False)
    
    # Plotte Vergleich
    plot_comparison(results, env_id=env_id, seed=seed)
    
    return results