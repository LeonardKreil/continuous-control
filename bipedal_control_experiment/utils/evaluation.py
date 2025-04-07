import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy

def evaluate_and_save(model, env, n_eval_episodes=10, deterministic=True, render=False,
                     algo_name=None, env_id=None, seed=None):
    """
    Evaluiert ein Modell und speichert die Ergebnisse.
    """
    # Evaluierung durchführen
    episode_rewards, episode_lengths = [], []
    
    for i in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    # Berechne Statistiken
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    # Speichere Ergebnisse
    if algo_name and env_id:
        os.makedirs("./results", exist_ok=True)
        results_path = f"./results/eval_{algo_name}_{env_id}"
        if seed is not None:
            results_path += f"_{seed}"
        results_path += ".csv"
        
        df = pd.DataFrame({
            "episode": range(1, n_eval_episodes + 1),
            "reward": episode_rewards,
            "length": episode_lengths
        })
        df.to_csv(results_path, index=False)
        
        # Zusammenfassung
        summary_df = pd.DataFrame({
            "metric": ["mean_reward", "std_reward", "mean_length", "std_length"],
            "value": [mean_reward, std_reward, mean_length, std_length]
        })
        summary_path = results_path.replace(".csv", "_summary.csv")
        summary_df.to_csv(summary_path, index=False)
    
    return mean_reward, std_reward