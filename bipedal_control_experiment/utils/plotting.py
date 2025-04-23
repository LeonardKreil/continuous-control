import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

def plot_learning_curve(log_dirs, labels=None, title="Lernkurven", window=5):
    """
    Plottet Lernkurven aus mehreren Trainingsläufen zum Vergleich.
    
    Args:
        log_dirs: Liste von Verzeichnissen, die Training-Log-Dateien enthalten
        labels: Liste von Labels für die Legende
        title: Titel des Plots
        window: Fenstergröße für gleitenden Durchschnitt
    """
    plt.figure(figsize=(10, 6))
    
    if labels is None:
        labels = [os.path.basename(log_dir) for log_dir in log_dirs]
    
    for i, log_dir in enumerate(log_dirs):
        log_file = os.path.join(log_dir, "training_log.csv")
        
        if not os.path.exists(log_file):
            print(f"Warnung: Log-Datei {log_file} nicht gefunden.")
            continue
        
        # Log-Daten laden
        data = pd.read_csv(log_file)
        
        # Gleitender Durchschnitt für die Rewards berechnen
        rewards = data['reward'].values
        timesteps = data['timestep'].values
        
        if window > 1:
            smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            # Passe die x-Achse entsprechend an
            smooth_timesteps = timesteps[window-1:]
        else:
            smooth_rewards = rewards
            smooth_timesteps = timesteps
        
        # Plot
        plt.plot(smooth_timesteps, smooth_rewards, label=labels[i])
    
    plt.xlabel('Timesteps')
    plt.ylabel('Durchschnittliche Belohnung')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
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