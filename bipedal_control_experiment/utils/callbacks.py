import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

class TrainingLoggerCallback(BaseCallback):
    """
    Callback für das Speichern von Trainingsfortschritten in regelmäßigen Abständen.
    Speichert Belohnungen und andere relevante Metriken für späteres Plotten der Lernkurven.
    """
    def __init__(self, log_dir, eval_env=None, eval_freq=1000, n_eval_episodes=5, 
                 save_freq=10000, save_path=None, verbose=1):
        super(TrainingLoggerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.save_path = save_path
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations_rewards = []
        self.evaluations_timesteps = []
        
        # Stelle sicher, dass das Log-Verzeichnis existiert
        os.makedirs(log_dir, exist_ok=True)
        
        # Dateinamen für die Logs
        self.eval_results_file = os.path.join(log_dir, "eval_results.npz")
        self.training_log_file = os.path.join(log_dir, "training_log.csv")
        
        # Erstelle CSV-Datei mit Header
        with open(self.training_log_file, "w") as f:
            f.write("timestep,episode,reward,episode_length,success_rate\n")
        
        self.last_mean_reward = -np.inf
        
    def _init_callback(self):
        # Erstelle Ergebnisordner, falls nicht vorhanden
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
    
    def _on_step(self):
        # Speichere Modell periodisch
        if self.save_path is not None and self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}_{self.n_calls}_steps"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Modell gespeichert bei {model_path}")
        
        # Evaluiere das aktuelle Modell und speichere die Ergebnisse
        if self.eval_env is not None and self.n_calls % self.eval_freq == 0:
            episode_rewards = []
            episode_lengths = []
            success_count = 0
            
            # Führe mehrere Evaluierungsepisoden durch
            for i in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                truncated = False
                episode_reward = 0
                episode_length = 0
                
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    # Prüfe auf Erfolg, falls vom Environment unterstützt
                    if 'is_success' in info:
                        success_count += info['is_success']
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            success_rate = success_count / self.n_eval_episodes if self.n_eval_episodes > 0 else 0
            
            # Speichere die Ergebnisse
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_rewards.append(mean_reward)
            
            # Log in Datei speichern
            with open(self.training_log_file, "a") as f:
                f.write(f"{self.num_timesteps},{len(self.evaluations_timesteps)},{mean_reward},{mean_length},{success_rate}\n")
            
            # Speichere detaillierte Ergebnisse als NPZ-Datei
            np.savez(
                self.eval_results_file,
                timesteps=self.evaluations_timesteps,
                rewards=self.evaluations_rewards,
                ep_lengths=episode_lengths,
                success_rate=success_rate
            )
            
            if self.verbose > 0:
                print(f"Timestep: {self.num_timesteps}, mittlere Belohnung: {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Speichere das beste Modell
            if mean_reward > self.last_mean_reward and self.save_path is not None:
                self.last_mean_reward = mean_reward
                best_model_path = f"{self.save_path}_best"
                self.model.save(best_model_path)
                if self.verbose > 0:
                    print(f"Bestes Modell gespeichert bei {best_model_path}")
        
        return True