import gymnasium as gym
import numpy as np
import time

# Umgebung erstellen
env = gym.make('BipedalWalker-v3', render_mode='human')

# Startposition zurücksetzen und erste Beobachtung erhalten
observation, info = env.reset(seed=42)

# Einige Schritte mit zufälligen Aktionen durchführen
for _ in range(1000):
    # Zufällige Aktion generieren (4 kontinuierliche Werte zwischen -1 und 1)
    action = np.random.uniform(-1, 1, size=4)
    
    # Aktion ausführen
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Ausgabe von Beobachtung und Belohnung
    print(f"Reward: {reward:.2f}")
    
    # Kurze Pause für bessere Visualisierung
    time.sleep(0.01)
    
    # Wenn Episode beendet, zurücksetzen
    if terminated or truncated:
        observation, info = env.reset()

# Umgebung schließen
env.close()