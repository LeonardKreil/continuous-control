import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomPPOPolicy(ActorCriticPolicy):
    """
    Eine angepasste Policy für PPO, falls spezielle Anpassungen benötigt werden.
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomPPOPolicy, self).__init__(
            observation_space, 
            action_space, 
            lr_schedule, 
            *args, **kwargs
        )
        # Hier können angepasste Netzwerk-Architekturen oder andere Änderungen implementiert werden

class PPOTrainer:
    """
    Wrapper-Klasse für das PPO-Training.
    """
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Initialisiere das PPO-Modell
        self.model = PPO(
            policy=config.get("policy", "MlpPolicy"),
            env=env,
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 2048),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            clip_range_vf=config.get("clip_range_vf", None),
            normalize_advantage=config.get("normalize_advantage", True),
            ent_coef=config.get("ent_coef", 0.0),
            vf_coef=config.get("vf_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            use_sde=config.get("use_sde", False),
            sde_sample_freq=config.get("sde_sample_freq", -1),
            target_kl=config.get("target_kl", None),
            tensorboard_log=config.get("tensorboard_log", None),
            policy_kwargs=config.get("policy_kwargs", None),
            verbose=config.get("verbose", 0),
            seed=config.get("seed", None),
            device=config.get("device", "auto"),
        )
        
    def train(self, total_timesteps, callback=None):
        """
        Trainiert das PPO-Modell.
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
        self.model = PPO.load(path, env=self.env)
    
    def predict(self, observation, deterministic=True):
        """
        Führt eine Vorhersage basierend auf der Beobachtung durch.
        """
        return self.model.predict(observation, deterministic=deterministic)