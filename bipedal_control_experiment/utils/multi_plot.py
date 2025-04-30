import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_training_csv(file_path):
    """
    Lädt eine CSV-Datei mit 'timestep' und 'reward'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
    
    df = pd.read_csv(file_path)
    if 'timestep' not in df.columns or 'reward' not in df.columns:
        raise ValueError(f"CSV-Datei {file_path} muss 'timestep' und 'reward' enthalten.")
    
    return df

def plot_multiple_reward_curves(csv_files, labels=None, title="Reward über Timesteps", window=10, save_path=None):
    """
    Plottet mehrere Reward-Kurven in einem Diagramm.
    
    Args:
        csv_files: Liste von Pfaden zu CSV-Dateien
        labels: Optional Liste von Labels (muss gleiche Länge wie csv_files haben)
        title: Titel des Plots
        window: Fenstergröße für gleitenden Durchschnitt
        save_path: Wenn angegeben, wird der Plot gespeichert
    """
    plt.figure(figsize=(10, 6))

    if labels is None:
        labels = [os.path.splitext(os.path.basename(path))[0] for path in csv_files]
    
    for i, file_path in enumerate(csv_files):
        df = load_training_csv(file_path)
        timesteps = df['timestep'].values
        rewards = df['reward'].values

        # Gemeinsamen Startpunkt setzen
        timesteps = np.insert(timesteps, 0, 0)
        rewards = np.insert(rewards, 0, -100)

        # Gleitender Durchschnitt
                # Gleitender Durchschnitt mit Padding am Anfang
        if window > 1:
            # Padding mit dem Startwert
            rewards_padded = np.pad(rewards, (window-1, 0), mode='edge')
            rewards_smoothed = np.convolve(rewards_padded, np.ones(window)/window, mode='valid')
        else:
            rewards_smoothed = rewards


        plt.plot(timesteps, rewards_smoothed, label=labels[i])


    plt.xlabel("Timesteps")
    plt.ylabel("Belohnung")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot gespeichert unter: {save_path}")
    else:
        plt.show()

    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vergleich von Rewards aus mehreren Trainingsläufen")
    parser.add_argument("--csv", nargs='+', required=True, help="Liste von CSV-Dateien")
    parser.add_argument("--labels", nargs='+', help="Liste von Labels für die Legende")
    parser.add_argument("--title", type=str, default="Reward-Vergleich")
    parser.add_argument("--window", type=int, default=10, help="Fenstergröße für gleitenden Durchschnitt")
    parser.add_argument("--save", type=str, help="Pfad zum Speichern des Plots")
    
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.csv):
        raise ValueError("Anzahl der Labels muss mit Anzahl der CSV-Dateien übereinstimmen.")
    
    plot_multiple_reward_curves(
        csv_files=args.csv,
        labels=args.labels,
        title=args.title,
        window=args.window,
        save_path=args.save
    )
