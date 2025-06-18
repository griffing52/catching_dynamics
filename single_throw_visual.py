import mujoco
import mujoco.viewer
import time
import os, ray
from ray.tune.registry import register_env
from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_ENV_TO_MODULE_CONNECTOR,
    COMPONENT_MODULE_TO_ENV_CONNECTOR,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)
from ray.rllib.core.rl_module.rl_module import RLModule
import numpy as np
import torch

from envs.single_throw_env import SingleThrowEnv          # your Gymnasium base
from gymnasium_robotics import mamujoco_v1
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

base_env = SingleThrowEnv()

def env_creator(config=None):
    return ParallelPettingZooEnv(
        mamujoco_v1.parallel_env(
            scenario="SingleThrow",
            agent_conf="1x5",
            agent_factorization=base_env.get_agent_factorization(),
            global_categories=base_env.get_global_categories(),
            gym_env=base_env,
            render_mode=None,
        )
    )

register_env("SingleThrowMultiEnv", env_creator)

ray.init(ignore_reinit_error=True)

ckpt_dir = "checkpoints_st/"
ckpt_root = os.path.abspath(ckpt_dir)
os.makedirs(ckpt_root, exist_ok=True)
# Create RLModule from a checkpoint.
rl_module = RLModule.from_checkpoint(
    os.path.join(
        ckpt_root, # (or directly a string path to the checkpoint dir)
        COMPONENT_LEARNER_GROUP,
        COMPONENT_LEARNER,
        COMPONENT_RL_MODULE,
        "shared",
    )
)

env = env_creator({})
obs, _ = env.reset(seed=0)
terminations = {"__all__": False}

# Initialize episode tracking
episode_reward = 0
step_count = 0
max_steps = 1000  # Maximum steps per episode

# Slow down the simulation
base_env .model.opt.timestep = 0.001  # Slower physics timestep
base_env.model.opt.iterations = 20  # More physics iterations per step

print("Starting environment test...")
print("Press Ctrl+C to exit")

# Use MuJoCo's native viewer
with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
    try:
        while True:
            while viewer.is_running() and not terminations["__all__"]:
                step_start = time.time()

                # Get actions for each agent (random for testing)
                # actions = {
                #     "agent_0": env.action_space("agent_0").sample(),  # Left arm actions
                #     "agent_1": env.action_space("agent_1").sample()   # Right arm actions
                # }

                action_dist_inputs = rl_module.forward_inference({'obs': torch.from_numpy(obs['agent_0']).float()})[
                    'action_dist_inputs']

                distr = torch.distributions.Normal(
                    loc=action_dist_inputs[:5],
                    scale=torch.exp(action_dist_inputs[5:])*0+0.0001
                )
                actions = distr.sample().numpy()

                obs, rewards, terminations, truncations, infos = env.step({'agent_0': actions})

                # Update episode tracking
                episode_reward += sum(rewards.values())
                step_count += 1

                # Print information every 100 steps
                if step_count % 100 == 0:
                    print(f"\nStep {step_count}")
                    print(f"Current catcher: {infos['agent_0']['catcher']}")
                    print(f"Ball position: {infos['agent_0']['ball_position']}")
                    print(f"Episode reward so far: {episode_reward:.2f}")

                # Check if episode is done
                if all(terminations.values()) or all(truncations.values()) or step_count >= max_steps:
                    print("\nEpisode finished!")
                    print(f"Total steps: {step_count}")
                    print(f"Total reward: {episode_reward:.2f}")

                    # Reset for next episode
                    obs, info = env.reset()
                    episode_reward = 0
                    step_count = 0
                    print("\nStarting new episode...")

                # Synchronize the viewer
                viewer.sync()

                # Rudimentary time keeping
                time_until_next_step = base_env.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            if not viewer.is_running():
                break

            # Reset for next episode
            obs, info = env.reset(seed=0)
            terminations = {"__all__": False}
            episode_reward = 0
            step_count = 0


    except KeyboardInterrupt:
        print("\nTest terminated by user")
    finally:
        env.close()
        print("Environment closed")