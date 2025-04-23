import os
import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, DDPG, TD3, DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from utils.logger import setup_logger
from utils.plotting import plot_learning_curve
from utils.evaluation import evaluate_and_save
from utils.callbacks import TrainingLoggerCallback

class Experiment:
    def __init__(self, algo_name, env_id, config, seed=42, pretrained_model_path=None):
        self.algo_name = algo_name
        self.env_id = env_id
        self.config = config
        self.seed = seed
        
        # Erstelle Verzeichnisse falls sie nicht existieren
        os.makedirs("./models", exist_ok=True)
        self.results_dir = f"./results/{algo_name}_{env_id}_{seed}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.logger = setup_logger(f"{algo_name}_{env_id}_{seed}", log_dir=f"{self.results_dir}/logs")
        
        self.env = self._create_env()
        # Erstelle eine separate Umgebung für Evaluierungen
        self.eval_env = self._create_env()
        
        # Lade vortrainiertes Modell oder erstelle ein neues
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self.model = self._load_model(pretrained_model_path)
            self.logger.info(f"Vortrainiertes Modell geladen von: {pretrained_model_path}")
        else:
            self.model = self._create_model()
            self.logger.info(f"Neues Modell erstellt: {algo_name}")
        
    def _create_env(self):
        """Erstellt und konfiguriert die Umgebung."""
        env = gym.make(self.env_id)
        env = Monitor(env, f"./results/{self.algo_name}")
        env.reset(seed=self.seed)
        return env
    
    def _load_model(self, model_path):
        """Lädt ein vortrainiertes Modell basierend auf dem Algorithmus."""
        if self.algo_name == "ppo":
            return PPO.load(model_path, env=self.env)
        elif self.algo_name == "sac":
            return SAC.load(model_path, env=self.env) 
        elif self.algo_name == "ddpg":
            return DDPG.load(model_path, env=self.env)
        elif self.algo_name == "td3":
            return TD3.load(model_path, env=self.env)
        else:
            raise ValueError(f"Unbekannter Algorithmus: {self.algo_name}")
        
    def _create_model(self):
        """Erstellt das Modell basierend auf dem gewählten Algorithmus."""
        if self.algo_name == "ppo":
            return PPO(self.config["policy"], self.env, **{k: v for k, v in self.config.items() if k != "policy"})
        elif self.algo_name == "sac":
            return SAC(self.config["policy"], self.env, **{k: v for k, v in self.config.items() if k != "policy"})
        elif self.algo_name == "ddpg":
            # Spezialbehandlung für Action Noise
            if self.config.get("action_noise") == "OrnsteinUhlenbeckActionNoise":
                action_dim = self.env.action_space.shape[0]
                action_noise = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(action_dim),
                    sigma=self.config.get("noise_sigma", 0.1) * np.ones(action_dim)
                )
                config = {k: v for k, v in self.config.items() 
                          if k not in ["action_noise", "noise_sigma"]}
                return DDPG(policy='MlpPolicy', env=self.env, action_noise=action_noise, **config)
            return DDPG(self.config["policy"], self.env, **{k: v for k, v in self.config.items() if k != "policy"})
        elif self.algo_name == "td3":
            # Spezialbehandlung für Action Noise
            if self.config.get("action_noise") == "NormalActionNoise":
                action_dim = self.env.action_space.shape[0]
                action_noise = NormalActionNoise(
                    mean=np.zeros(action_dim),
                    sigma=self.config.get("noise_sigma", 0.1) * np.ones(action_dim)
                )
                config = {k: v for k, v in self.config.items() 
                          if k not in ["action_noise", "noise_sigma"]}
                return TD3(self.config["policy"], self.env, action_noise=action_noise, **config)
            return TD3(self.config["policy"], self.env, **{k: v for k, v in self.config.items() if k != "policy"})
        else:
            raise ValueError(f"Unbekannter Algorithmus: {self.algo_name}")
    
    def train(self, total_timesteps, eval_freq=10000, n_eval_episodes=5, save_freq=50000):
        """Trainiert das Modell und evaluiert es regelmäßig."""
        self.logger.info(f"Start Training: {self.algo_name} on {self.env_id}")
        start_time = time.time()
        
        # Callback für die regelmäßige Evaluierung erstellen
        callbacks = [
            TrainingLoggerCallback(
                log_dir=self.results_dir,
                eval_env=self.eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                save_freq=save_freq,
                save_path=f"./models/{self.algo_name}_{self.env_id}_{self.seed}",
                verbose=1
            )
        ]
        
        # Training starten
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=100
        )
        
        # Modell speichern
        model_path = f"./models/{self.algo_name}_{self.env_id}_{self.seed}_final.zip"
        self.model.save(model_path)
        self.logger.info(f"Modell gespeichert unter: {model_path}")
        
        training_time = time.time() - start_time
        self.logger.info(f"Training abgeschlossen in {training_time:.2f} Sekunden")
        
        return model_path
    
    def evaluate(self, n_eval_episodes=10):
        """Evaluiert das trainierte Modell."""
        self.logger.info(f"Evaluiere {self.algo_name} für {n_eval_episodes} Episoden")
        
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )
        
        self.logger.info(f"Mittlere Belohnung: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Ergebnisse speichern
        results_path = f"./results/{self.algo_name}_{self.env_id}_{self.seed}_eval.npz"
        np.savez(
            results_path,
            mean_reward=mean_reward,
            std_reward=std_reward,
            n_episodes=n_eval_episodes
        )
        
        return mean_reward, std_reward
    
    def visualize(self, n_episodes=50):
        """Visualisiert das trainierte Modell."""
        self.logger.info(f"Visualisiere {self.algo_name} für {n_episodes} Episoden")
        
        env = gym.make(self.env_id, render_mode="human")
        obs, _ = env.reset()
        
        for episode in range(n_episodes):
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
            self.logger.info(f"Episode {episode+1}: Belohnung = {episode_reward:.2f}")
            obs, _ = env.reset()
            
        env.close()