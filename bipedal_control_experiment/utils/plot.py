import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

def load_training_csv_files(directory_pattern, base_dir="."):
    """
    Lädt alle CSV-Dateien, die einem Muster entsprechen.
    
    Args:
        directory_pattern: Muster für die Verzeichnisse (z.B. "ddpg_*")
        base_dir: Basis-Verzeichnis, in dem gesucht werden soll
    
    Returns:
        Liste von DataFrames mit den Daten aus den CSV-Dateien
    """
    csv_files = []
    
    # Vollständigen Pfad für das Muster erstellen
    full_pattern = os.path.join(base_dir, directory_pattern)
    
    # Alle Verzeichnisse finden, die dem Muster entsprechen
    directories = glob.glob(full_pattern)
    
    if not directories:
        print(f"Keine Verzeichnisse gefunden, die dem Muster '{directory_pattern}' entsprechen.")
        return []
    
    # Alle CSV-Dateien in den Verzeichnissen finden
    for directory in directories:
        csv_path = os.path.join(directory, "training_log.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'timestep' not in df.columns or 'reward' not in df.columns:
                    # Prüfen ob reward_std vorhanden ist, falls nicht, ignorieren wir die Datei
                    if 'timestep' not in df.columns or 'reward_std' not in df.columns:
                        print(f"Warnung: CSV-Datei {csv_path} hat nicht die erwarteten Spalten.")
                        continue
                csv_files.append(df)
            except Exception as e:
                print(f"Fehler beim Laden von {csv_path}: {e}")
    
    if not csv_files:
        print(f"Keine gültigen CSV-Dateien in den Verzeichnissen gefunden.")
    
    print(len(csv_files)) 
    return csv_files

def aggregate_data_for_algorithm(data_frames):
    """
    Aggregiert die Daten aus mehreren DataFrames.
    
    Args:
        data_frames: Liste von DataFrames
    
    Returns:
        DataFrame mit aggregierten Daten (Mittelwert und Standardabweichung pro Timestep)
    """
    if not data_frames:
        return None
    
    # Prüfen, ob alle DataFrames dieselben Timesteps haben
    # Falls nicht, müssen wir interpolieren oder auf gemeinsame Timesteps reduzieren
    timesteps = sorted(set(np.concatenate([df['timestep'].values for df in data_frames])))
    
    # Für jeden Timestep den Mittelwert und die Standardabweichung berechnen
    aggregated_data = {
        'timestep': [],
        'reward_mean': [],
        'reward_std_mean': [],  # Mittelwert der Standardabweichungen
        'reward_total_std': []  # Standardabweichung aller Werte
    }
    
    # Startwert bei 0 mit Reward -100 hinzufügen
    aggregated_data['timestep'].append(0)
    aggregated_data['reward_mean'].append(-100)
    aggregated_data['reward_std_mean'].append(0)
    aggregated_data['reward_total_std'].append(0)

    for timestep in timesteps:
        rewards = []
        std_values = []
        
        for df in data_frames:
            # Finde den nächsten passenden Timestep
            row = df[df['timestep'] == timestep]
            if not row.empty:
                rewards.append(row['reward'].values[0])
                if 'reward_std' in df.columns:
                    std_values.append(row['reward_std'].values[0])
        
        if rewards:
            aggregated_data['timestep'].append(timestep)
            aggregated_data['reward_mean'].append(np.mean(rewards))
            if std_values:
                aggregated_data['reward_std_mean'].append(np.mean(std_values))
            else:
                aggregated_data['reward_std_mean'].append(0)
            aggregated_data['reward_total_std'].append(np.std(rewards) if len(rewards) > 1 else 0)

    result_df = pd.DataFrame(aggregated_data)
    print(result_df)  # Zeigt die ersten 5 Zeilen

    return pd.DataFrame(aggregated_data)

def plot_algorithm_comparison(algorithm_patterns, base_dir=".", labels=None, title="Comparison of algorithms",
                             window=10, save_path=None, colors=None):
    """
    Plottet den Vergleich mehrerer Algorithmen mit Mittelwert und Standardabweichung.
    
    Args:
        algorithm_patterns: Liste von Mustern für die Verzeichnisse der Algorithmen
        base_dir: Basis-Verzeichnis, in dem nach den Algorithmus-Verzeichnissen gesucht wird
        labels: Optional Liste von Labels (muss gleiche Länge wie algorithm_patterns haben)
        title: Titel des Plots
        window: Fenstergröße für gleitenden Durchschnitt
        save_path: Wenn angegeben, wird der Plot gespeichert
        colors: Liste von Farben für die Algorithmen
    """
    plt.figure(figsize=(12, 8))
    
    if labels is None:
        labels = [pattern.split('_')[0] for pattern in algorithm_patterns]
    
    if colors is None:
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, pattern in enumerate(algorithm_patterns):
        algorithm_name = labels[i] if i < len(labels) else pattern.split('_')[0]
        color = colors[i % len(colors)]
        
        # Alle CSV-Dateien für diesen Algorithmus laden
        data_frames = load_training_csv_files(pattern, base_dir)
        
        if not data_frames:
            print(f"Keine Daten für Algorithmus {algorithm_name} gefunden.")
            continue
        
        # Daten aggregieren
        aggregated_data = aggregate_data_for_algorithm(data_frames)
        
        if aggregated_data is None or aggregated_data.empty:
            print(f"Keine gültigen Daten für Algorithmus {algorithm_name}.")
            continue
        
        timesteps = aggregated_data['timestep'].values
        rewards_mean = aggregated_data['reward_mean'].values
        rewards_std = aggregated_data['reward_total_std'].values
        
        # Gleitender Durchschnitt für Mittelwert und Standardabweichung
        if window > 1:
            rewards_mean_padded = np.pad(rewards_mean, (window-1, 0), mode='edge')
            rewards_mean_smoothed = np.convolve(rewards_mean_padded, np.ones(window)/window, mode='valid')
            
            rewards_std_padded = np.pad(rewards_std, (window-1, 0), mode='edge')
            rewards_std_smoothed = np.convolve(rewards_std_padded, np.ones(window)/window, mode='valid')
        else:
            rewards_mean_smoothed = rewards_mean
            rewards_std_smoothed = rewards_std
        
        # Mittelwert als Linie plotten
        plt.plot(timesteps, rewards_mean_smoothed, label=algorithm_name, color=color, linewidth=2)
        
        # Standardabweichung als Band plotten
        plt.fill_between(timesteps, 
                         rewards_mean_smoothed - rewards_std_smoothed, 
                         rewards_mean_smoothed + rewards_std_smoothed, 
                         color=color, alpha=0.2)
    
    plt.xlabel("Environment Interactions (Steps)", fontsize=12)
    plt.ylabel("Mean Reward", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot gespeichert unter: {save_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vergleich von Algorithmen mit mehreren Durchläufen")
    parser.add_argument("--dir", type=str, default=".", 
                        help="Basis-Verzeichnis, in dem nach den Algorithmus-Verzeichnissen gesucht wird")
    parser.add_argument("--patterns", nargs='+', required=True, 
                        help="Liste von Mustern für die Algorithmus-Verzeichnisse (z.B. 'ddpg_*' 'sac_*')")
    parser.add_argument("--labels", nargs='+', 
                        help="Liste von Labels für die Legende")
    parser.add_argument("--title", type=str, default="Comparison of different continuous controll algorithms")
    parser.add_argument("--window", type=int, default=10, 
                        help="Fenstergröße für gleitenden Durchschnitt")
    parser.add_argument("--save", type=str, 
                        help="Pfad zum Speichern des Plots")
    parser.add_argument("--colors", nargs='+',
                        help="Liste von Farben für die Algorithmen")
    
    args = parser.parse_args()
    
    if args.labels and len(args.labels) != len(args.patterns):
        raise ValueError("Anzahl der Labels muss mit Anzahl der Muster übereinstimmen.")
    
    if args.colors and len(args.colors) < len(args.patterns):
        print("Warnung: Zu wenige Farben angegeben. Es werden Standardfarben verwendet.")
        args.colors = None
    
    plot_algorithm_comparison(
        algorithm_patterns=args.patterns,
        base_dir=args.dir,
        labels=args.labels,
        title=args.title,
        window=args.window,
        save_path=args.save,
        colors=args.colors
    )