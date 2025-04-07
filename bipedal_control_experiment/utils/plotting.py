import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

def plot_learning_curve(log_folder, title, x_axis="Timesteps"):
    """
    Zeichnet eine Lernkurve basierend auf den Monitor-Dateien.
    """
    plt.figure(figsize=(10, 5))
    plt.title(title)
    
    # Lade Trainingsdaten
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = np.array(moving_average(y, window=50))
    
    # Zeichne Lernkurve
    plt.plot(x, y)
    plt.xlabel(x_axis)
    plt.ylabel("Reward")
    plt.grid()
    
    # Speichere Diagramm
    os.makedirs("./results/plots", exist_ok=True)
    plt.savefig(f"./results/plots/{title.replace(' ', '_')}.png")
    plt.close()

def moving_average(values, window):
    """
    Berechnet den gleitenden Durchschnitt mit der angegebenen Fenstergröße.
    """
    if len(values) < window:
        return values
    
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_comparison(results, env_id, seed):
    """
    Zeichnet einen Vergleich zwischen verschiedenen Algorithmen.
    """
    plt.figure(figsize=(12, 6))
    plt.title(f"Algorithmen-Vergleich auf {env_id}")
    
    # Daten für das Diagramm vorbereiten
    algo_names = list(results.keys())
    mean_rewards = [results[algo]["mean_reward"] for algo in algo_names]
    std_rewards = [results[algo]["std_reward"] for algo in algo_names]
    
    # Balkendiagramm mit Fehlerbalken
    bars = plt.bar(algo_names, mean_rewards, yerr=std_rewards, capsize=10)
    
    # Beschriftungen und Grid
    plt.xlabel("Algorithmus")
    plt.ylabel("Mittlere Belohnung")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Beschriftungen über den Balken
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}', ha='center', va='bottom')
    
    # Speichere Diagramm
    os.makedirs("./results/plots", exist_ok=True)
    plt.savefig(f"./results/plots/comparison_{env_id}_{seed}.png")
    plt.close()
    
    # Erstelle auch eine Tabelle mit den Ergebnissen
    df = pd.DataFrame({
        "Algorithmus": algo_names,
        "Mittlere Belohnung": mean_rewards,
        "Standardabweichung": std_rewards
    })
    df.to_csv(f"./results/comparison_{env_id}_{seed}_table.csv", index=False)