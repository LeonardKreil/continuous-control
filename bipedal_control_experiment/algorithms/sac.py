import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

class SACTrainer:
    """
    Wrapper-Klasse für das SAC-Training.
    """
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Initialisiere das SAC-Modell
        self.model = SAC(
            policy=config.get("policy", "MlpPolicy"),
            env=env,
            learning_rate=config.get("learning_rate", 3e-4),
            buffer_size=config.get("buffer_size", 1000000),
            learning_starts=config.get("learning_starts", 100),
            batch_size=config.get("batch_size", 256),
            tau=config.get("tau", 0.005),
            gamma=config.get("gamma", 0.99),
            train_freq=config.get("train_freq", 1),
            gradient_steps=config.get("gradient_steps", 1),
            action_noise=config.get("action_noise", None),
            replay_buffer_class=config.get("replay_buffer_class", None),
            replay_buffer_kwargs=config.get("replay_buffer_kwargs", None),
            optimize_memory_usage=config.get("optimize_memory_usage", False),
            ent_coef=config.get("ent_coef", "auto"),
            target_update_interval=config.get("target_update_interval", 1),
            target_entropy=config.get("target_entropy", "auto"),
            use_sde=config.get("use_sde", False),
            sde_sample_freq=config.get("sde_sample_freq", -1),
            use_sde_at_warmup=config.get("use_sde_at_warmup", False),
            tensorboard_log=config.get("tensorboard_log", None),
            policy_kwargs=config.get("policy_kwargs", None),
            verbose=config.get("verbose", 0),
            seed=config.get("seed", None),
            device=config.get("device", "auto"),
        )
        
    def train(self, total_timesteps, callback=None):
        """
        Trainiert das SAC-Modell.
        """
        return self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def save(self, path):
        """
        Speichert das trainierte Modell.
        """
        self.model.save(path)
    
    def load(self, path):
        """
        Lädt ein vortrainiertes Modell.
        """
        self.model = SAC.load(path, env=self.env)
    
    def predict(self, observation, deterministic=True):
        """
        Führt eine Vorhersage basierend auf der Beobachtung durch.
        """
        return self.model.predict(observation, deterministic=deterministic)