# train_single_catch.py
# Script to train a PPO agent on the SingleCatchEnv using Ray's old API stack.

import argparse
import os
import gymnasium as gym # Changed from import gym
import numpy as np # Added for a more precise shape check if needed
import ray
from ray import tune
from ray.rllib.algorithms.ppo.ppo import PPOConfig # Unified PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from ray.tune.registry import register_env
import torch
import torch.nn as nn

# Assuming SingleCatchEnv is defined in this file or correctly imported.
# For this example, let's use the provided envs.single_catch_env
# Make sure this path is correct relative to where you run the script,
# or install it as a package.
from envs.single_catch_env import SingleCatchEnv


# Example of a custom model if needed for the old API stack (ModelV2)
# If the default FCNet is sufficient, this is not strictly necessary
# but good to know for ModelV2.
class CustomModelV2(TorchModelV2, nn.Module):
    """Example of a custom model that_is_compatible with ModelV2."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFCNet(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, [] # Old API expects list of state_out

    def value_function(self):
        return self.torch_sub_model.value_function()


def main(args):
    """Main training script."""

    # --- Initial Environment Sanity Check ---
    # This check is added to help diagnose observation space mismatches.
    # The error "Observation (...) outside given space (...)" indicates that
    # the observation returned by your environment's reset() or step() method
    # does not match the 'self.observation_space' defined in your environment.
    print("--- Initial Environment Sanity Check ---")
    try:
        # env_check_config = {
        #     "time_limit": 100, # Or whatever is appropriate for your env
        #     "xml_path": args.xml_path,
        #     "show_gui": False, # Should be False for this check
        # }
        # print(f"Attempting to instantiate SingleCatchEnv with config: {env_check_config}")
        check_env = SingleCatchEnv()#**env_check_config)
        print(f"Successfully instantiated SingleCatchEnv.")
        print(f"Environment's declared observation space: {check_env.observation_space}")
        print(f"Shape of declared observation_space: {check_env.observation_space.shape}")
        
        obs, info = check_env.reset() # Get an initial observation
        print(f"Actual observation from env.reset(): {obs}")
        print(f"Shape of actual observation from env.reset(): {obs.shape}")

        # Check for the specific mismatch from the error log: obs_shape=(10,) vs space_shape=(1,15)
        # The error indicates the space is (1,15) and the observation is effectively (10,)
        expected_shape_from_error = (1, 15) # Based on the error: Box(-inf, inf, (1, 15), float32)
        actual_obs_elements = obs.size # Total number of elements in the observation

        if check_env.observation_space.shape != expected_shape_from_error:
            print(f"WARNING: The environment's declared observation_space.shape {check_env.observation_space.shape} "
                  f"does not match the expected shape (1,15) from the error log. "
                  f"Please ensure it's defined correctly in SingleCatchEnv.")

        if actual_obs_elements != expected_shape_from_error[1]: # Comparing num elements
             print(f"CRITICAL WARNING: Shape mismatch! "
                  f"Actual observation has {actual_obs_elements} elements (shape {obs.shape}), "
                  f"but the error log indicates RLlib expects {expected_shape_from_error[1]} elements "
                  f"(based on space shape {expected_shape_from_error}).")
             print("This is the most likely cause of the 'Observation outside given space' error.")
             print("Please check your SingleCatchEnv definition in 'envs/single_catch_env.py':")
             print(f"  1. If you intend {actual_obs_elements} features: Ensure self.observation_space in SingleCatchEnv is defined like: "
                   f"gym.spaces.Box(..., shape=({actual_obs_elements},), dtype=np.float32) or gym.spaces.Box(..., shape=(1, {actual_obs_elements}), ...).")
             print(f"  2. If you intend {expected_shape_from_error[1]} features: Ensure your environment's reset() and step() methods return "
                   f"observations with {expected_shape_from_error[1]} elements, matching the shape {expected_shape_from_error}.")
        
        check_env.close()
        print("Environment sanity check complete.")
    except Exception as e:
        print(f"Could not perform environment sanity check: {e}")
        print("This might be due to an issue in SingleCatchEnv's __init__ or reset method, or incorrect args.xml_path.")
    print("------------------------------------")


    ray.init(num_cpus=args.num_cpus or None, num_gpus=args.num_gpus or None, local_mode=args.local_mode)

    # Environment configuration
    env_config = {
        "time_limit": 100, # Example, adjust as needed
        "xml_path": args.xml_path, # Pass xml_path to env
        "show_gui": False, # No GUI during training
    }

    # Register the custom environment
    # The lambda function takes the merged env_config
    register_env("single_catch_env", lambda config_dict: SingleCatchEnv(**config_dict))

    # Model configuration for the old API stack (ModelV2)
    # This will be used if default FCNet is desired.
    # If using a custom ModelV2, you'd register it and specify its name here.
    model_config = {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
        # If using CustomModelV2:
        # "custom_model": "my_custom_model_v2",
        # "custom_model_config": {}, # Any extra config for your custom model
    }
    # If you have a custom ModelV2 like the example above:
    # ModelCatalog.register_custom_model("my_custom_model_v2", CustomModelV2)


    # PPO Configuration for the OLD API stack
    config = (
        PPOConfig()
        .environment("single_catch_env", env_config=env_config)
        .framework("torch")
        .training(
            model=model_config,
            gamma=0.99,
            lr=0.0001, 
            train_batch_size=args.train_batch_size, 
            sgd_minibatch_size=args.sgd_minibatch_size, 
            num_sgd_iter=args.num_sgd_iter, 
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .env_runners(
            num_env_runners=args.num_workers, 
            rollout_fragment_length=args.rollout_fragment_length
        )
        .resources(
            num_gpus=args.num_gpus,
            num_cpus_per_worker=1,
            num_gpus_per_worker=0 
        )
    )

    # Explicitly disable new API stack components
    config = config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )

    # Define stop criteria for training
    stop_criteria = {
        "training_iteration": args.stop_iters,
    }

    # Convert storage_path to an absolute path
    storage_path_abs = os.path.abspath(args.storage_path)

    if not os.path.exists(storage_path_abs):
        os.makedirs(storage_path_abs, exist_ok=True)
    
    print("Starting training with PPO (Old API Stack)...")
    print(f"Configuration: {config.to_dict()}")
    print(f"Results will be saved to: {storage_path_abs}")

    results = tune.run(
        "PPO", 
        config=config.to_dict(),
        stop=stop_criteria,
        storage_path=storage_path_abs, 
        name=args.exp_name,
        verbose=1, 
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
    )

    print("Training finished.")
    best_trial = results.get_best_trial(metric="episode_reward_mean", mode="max")
    if best_trial:
        print(f"Best trial results saved to: {best_trial.path}") 
    else:
        print("No best trial found based on 'episode_reward_mean'. Check experiment results directly.")
        if results.trials:
             print(f"Last trial results saved to: {results.trials[-1].path}")

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on SingleCatchEnv using old API stack.")
    parser.add_argument("--num-cpus", type=int, default=0, help="Number of CPUs to use (0 for all available).")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to use.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of rollout workers (used for num_env_runners).")
    parser.add_argument("--local-mode", action="store_true", help="Run Ray in local mode for debugging.")
    parser.add_argument("--exp-name", type=str, default="ppo_single_catch_old_api", help="Name of the experiment.")
    parser.add_argument("--storage-path", type=str, default="./ray_results_old_api", help="Directory to save training results (replaces local_dir).")
    parser.add_argument("--stop-iters", type=int, default=100, help="Number of training iterations to run.")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Frequency of saving checkpoints (in iterations).")
    parser.add_argument(
        "--xml_path", 
        type=str, 
        default="models/single_catch.xml", 
        help="Path to the MuJoCo XML file for the environment."
    )
    parser.add_argument("--train-batch-size", type=int, default=4000, help="PPO train batch size.")
    parser.add_argument("--sgd-minibatch-size", type=int, default=128, help="PPO SGD minibatch size.")
    parser.add_argument("--num-sgd-iter", type=int, default=30, help="PPO number of SGD iterations per training batch.")
    parser.add_argument("--rollout-fragment-length", type=str, default="auto", help="Rollout fragment length. Use 'auto' or an integer.")

    args = parser.parse_args()
    
    if args.rollout_fragment_length != "auto":
        try:
            args.rollout_fragment_length = int(args.rollout_fragment_length)
        except ValueError:
            print(f"Invalid value for --rollout-fragment-length: {args.rollout_fragment_length}. Must be 'auto' or an integer.")
            exit(1)

    main(args)