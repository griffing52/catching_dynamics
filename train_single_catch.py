#!/usr/bin/env python3
"""
RLlib training script for the MaMuJoCo **SingleCatch** task (1×5 agent).

This version adds an **API‑compatibility wrapper** so RLlib's strict env check
(no longer forgiving as of Ray 2.3+) passes without throwing the common

    ValueError: Your environment (<PettingZooEnv<…>>) does not abide to the new gymnasium‑style API!

error you just hit.  Everything else (PPO, shared policy, checkpoints) is the
same.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import (
    MultiAgentEnvCompatibility,
)
from ray.tune.registry import register_env

# Gymnasium (not gym) is mandatory ‑‑ this makes sure the right version is on the
# path before any sub‑package (PettingZoo, MaMuJoCo, etc.) is imported.
import gymnasium as gym  # noqa: F401 – imported for side effects

# ---- 1.  Build the MaMuJoCo parallel environment --------------------------------

from gymnasium_robotics import mamujoco_v1
from envs.single_catch_env import SingleCatchEnv  # your own env module


def _make_parallel_env(**overrides):
    """Mirror the construction in *single_catch_visual.py* but keep it headless."""

    base_env = SingleCatchEnv(**overrides)

    # Convert the single‑agent Gymnasium env into a MaMuJoCo ParallelEnv.
    env = mamujoco_v1.parallel_env(
        scenario="SingleCatch",
        agent_conf="1x5",  # one arm, five actuators
        agent_factorization=base_env.get_agent_factorization(),
        global_categories=base_env.get_global_categories(),
        gym_env=base_env,
        render_mode=None,  # headless ‑> much faster rollouts
    )

    # Add required PettingZoo attributes
    env.agents = ["agent_0"]  # Single agent in our case
    env.possible_agents = ["agent_0"]
    env.agent_selection = "agent_0"

    # Add required methods
    def observe(agent):
        if agent not in env.agents:
            return None
        # Get the observation from the base environment
        obs = base_env._get_obs()
        return obs
    env.observe = observe
    
    # Override reset and step methods to ensure proper dictionary returns
    original_reset = env.reset
    def reset(*args, **kwargs):
        obs, info = original_reset(*args, **kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]  # Take first element if it's a tuple
        return {"agent_0": obs}, info
    env.reset = reset

    original_step = env.step
    def step(action):
        obs, reward, terminated, truncated, info = original_step(action)
        if isinstance(obs, tuple):
            obs = obs[0]  # Take first element if it's a tuple
        return (
            {"agent_0": obs},
            {"agent_0": reward},
            {"agent_0": terminated, "__all__": terminated},
            {"agent_0": truncated, "__all__": truncated},
            {"agent_0": info}
        )
    env.step = step

    return env


# ---- 2.  RLlib‑compatible creator ------------------------------------------------


def env_creator(config: Dict | None = None):
    """Return a brand‑new RLlib‑ready MultiAgentEnv instance."""

    pz_parallel = _make_parallel_env(**(config or {}))
    pz_env = PettingZooEnv(pz_parallel)
    # <‑‑ This wrapper does the gymnasium‑API normalization RLlib now insists on.
    return MultiAgentEnvCompatibility(pz_env, reset_returns_info=True)


register_env("SingleCatchMultiEnv", env_creator)


# ---- 3.  Main training loop ------------------------------------------------------


def main(args: argparse.Namespace):
    ray.init()

    # Peek once for spaces so we can build the policy dict.
    tmp = env_creator({})
    obs_space = tmp.observation_space["agent_0"]
    act_space = tmp.action_space["agent_0"]
    tmp.close()

    policies = {"shared_policy": (None, obs_space, act_space, {})}

    def map_fn(agent_id, *_) -> str:  # noqa: D401 – RLlib interface
        return "shared_policy"

    cfg = (
        PPOConfig()
        .environment(env="SingleCatchMultiEnv")
        .framework("torch")
        .env_runners(num_env_runners=args.num_workers)
        .resources(num_gpus=args.num_gpus)
        .training(train_batch_size=args.train_batch_size)
        .multi_agent(policies=policies, policy_mapping_fn=map_fn)
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=10,
            evaluation_config={"explore": False},
        )
        # If you *really* want to ignore RLlib's API checks entirely, uncomment:
        # .disable_env_checking()
    )

    algo = cfg.build()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for itr in range(1, args.num_iters + 1):
        result = algo.train()
        print(
            f"[Iter {itr}] reward_mean={result['episode_reward_mean']:.3f} "
            f"len_mean={result['episode_len_mean']:.1f} "
            f"time={result['time_total_s']:.1f}s",
            flush=True,
        )
        if itr % args.checkpoint_freq == 0 or itr == args.num_iters:
            ckpt = algo.save(args.checkpoint_dir)
            print(f" Saved checkpoint to {ckpt}\n", flush=True)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-iters", type=int, default=300)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-gpus", type=int, default=0)
    p.add_argument("--train-batch-size", type=int, default=320)
    p.add_argument("--checkpoint-freq", type=int, default=25)
    p.add_argument("--checkpoint-dir", type=str, default="single_catch_checkpoints")
    main(p.parse_args())
