import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback

class DDPGTrainer:
    """
    Wrapper-Klasse für das DDPG-Training.
    """
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Aktion-Noise erstellen, falls angegeben
        action_noise = None
        if config.get("action_noise") is not None:
            action_dim = env.action_space.shape[0]
            if config.get("action_noise") == "NormalActionNoise":
                action_noise = NormalActionNoise(
                    mean=np.zeros(action_dim),
                    sigma=config.get("noise_sigma", 0.1) * np.ones(action_dim)
                )
            elif config.get("action_noise") == "OrnsteinUhlenbeckActionNoise":
                action_noise = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(action_dim),
                    sigma=config.get("noise_sigma", 0.1) * np.ones(action_dim)
                )
        
        # Initialisiere das DDPG-Modell
        self.model = DDPG(
            policy=config.get("policy", "MlpPolicy"),
            env=env,
            learning_rate=config.get("learning_rate", 3e-4),
            buffer_size=config.get("buffer_size", 1000000),
            learning_starts=config.get("learning_starts", 100),
            batch_size=config.get("batch_size", 256),
            tau=config.get("tau", 0.005),
            gamma=config.get("gamma", 0.99),
            train_freq=config.get("train_freq", 1),
            gradient_steps=config.get("gradient_steps", -1),
            action_noise=action_noise,
            replay_buffer_class=config.get("replay_buffer_class", None),
            replay_buffer_kwargs=config.get("replay_buffer_kwargs", None),
            optimize_memory_usage=config.get("optimize_memory_usage", False),
            tensorboard_log=config.get("tensorboard_log", None),
            policy_kwargs=config.get("policy_kwargs", None),
            verbose=config.get("verbose", 0),
            seed=config.get("seed", None),
            device=config.get("device", "auto"),
        )
        
    def train(self, total_timesteps, callback=None):
        """
        Trainiert das DDPG-Modell.
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
        self.model = DDPG.load(path, env=self.env)
    
    def predict(self, observation, deterministic=True):
        """
        Führt eine Vorhersage basierend auf der Beobachtung durch.
        """
        return self.model.predict(observation, deterministic=deterministic)