import argparse
import gymnasium as gym
from experiments.experiment import Experiment
from experiments.compare import compare_algorithms
from config.hyperparameters import get_algorithm_config

def parse_args():
    parser = argparse.ArgumentParser(description="Bipedal Walker Control Experiment")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "ddpg", "td3", "dqn"],
                      help="Reinforcement learning algorithm to use")
    parser.add_argument("--env", type=str, default="BipedalWalker-v3",
                      help="Gymnasium environment")
    parser.add_argument("--train_steps", type=int, default=1000000,
                      help="Number of training timesteps")
    parser.add_argument("--eval_freq", type=int, default=10000,
                      help="Evaluation frequency during training")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--compare", action="store_true",
                      help="Compare all algorithms")
    parser.add_argument("--visualize", action="store_true",
                      help="Visualize trained agent")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.compare:
        compare_algorithms(env_id=args.env, 
                         total_timesteps=args.train_steps,
                         eval_freq=args.eval_freq,
                         seed=args.seed)
    else:
        config = get_algorithm_config(args.algo)
        experiment = Experiment(
            algo_name=args.algo,
            env_id=args.env,
            config=config,
            seed=args.seed,
            pretrained_model_path="./models/ppo_BipedalWalker-v3_42_final.zip"
        )
        
        if args.visualize:
            experiment.visualize()
        else:
            experiment.train(total_timesteps=args.train_steps, eval_freq=args.eval_freq)
            experiment.evaluate(n_eval_episodes=10)
            experiment.visualize()

if __name__ == "__main__":
    main()