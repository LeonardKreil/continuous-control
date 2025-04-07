def get_algorithm_config(algo_name):
    """Gibt die Hyperparameter-Konfiguration für den angegebenen Algorithmus zurück."""
    
    # Gemeinsame Parameter für alle Algorithmen
    common_params = {
        "learning_rate": 3e-4,
        "buffer_size": 1000000,
        "batch_size": 256,
        "gamma": 0.99,
        "verbose": 1,
        "tensorboard_log": "./results/tensorboard/",
    }
    
    # Spezifische Parameter für jeden Algorithmus
    algo_params = {
        "ppo": {
            **common_params,
            "n_steps": 2048,
            "ent_coef": 0.0,
            "learning_rate": 3e-4,
            # "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "normalize_advantage": True,
        },
        "sac": {
            **common_params,
            "policy": "MlpPolicy",
            "tau": 0.005,
            "ent_coef": "auto",
            "target_update_interval": 1,
            "train_freq": 1,
            "gradient_steps": 1,
        },
        "ddpg": {
            **common_params,
            "tau": 0.005,
            "action_noise": "OrnsteinUhlenbeckActionNoise",
            "noise_sigma": 0.1,
            "train_freq": (1, "episode"),
        },
        "td3": {
            **common_params,
            "tau": 0.005,
            "action_noise": "NormalActionNoise",
            "noise_sigma": 0.1,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "train_freq": (1, "episode"),
        },
    }
    
    return algo_params.get(algo_name, {})

def get_all_configs():
    """Gibt die Konfigurationen für alle Algorithmen zurück."""
    return {
        algo: get_algorithm_config(algo) 
        for algo in ["ppo", "sac", "ddpg", "td3", "dqn"]
    }