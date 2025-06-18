from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import ray
import gymnasium as gym  # noqa: F401 – ensure Gymnasium API is active
from gymnasium.spaces import Box
from gymnasium_robotics import mamujoco_v1
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EPISODE_LEN_MEAN,
)
from ray.tune.registry import register_env

# -----------------------------------------------------------------------------
# 1.  Build the MaMuJoCo parallel env
# -----------------------------------------------------------------------------

from envs.single_throw_env import SingleThrowEnv


def _make_parallel_env(**overrides):
    base_env = SingleThrowEnv(**overrides)
    return mamujoco_v1.parallel_env(
        scenario="SingleThrow",
        agent_conf="1x5",
        agent_factorization=base_env.get_agent_factorization(),
        global_categories=base_env.get_global_categories(),
        gym_env=base_env,
        render_mode=None,
    )


# -----------------------------------------------------------------------------
# 2.  Utility to force float32 dtypes
# -----------------------------------------------------------------------------


def _ensure_float32(env):
    """Align observation dtypes/spaces with RLlib expectations (float32)."""

    # Patch observation spaces
    for ag in env.possible_agents:
        space = env.observation_space(ag)
        if space.dtype != np.float32:
            env.observation_spaces[ag] = Box(
                low=-np.inf, high=np.inf, shape=space.shape, dtype=np.float32
            )

    # Wrap reset/step to cast observations
    orig_reset, orig_step = env.reset, env.step

    def reset(*args, **kwargs):
        res = orig_reset(*args, **kwargs)
        obs, info = res if isinstance(res, tuple) else (res, {})
        obs = {k: v.astype(np.float32) for k, v in obs.items()}
        return obs, info

    def step(actions):
        obs, rew, term, trunc, info = orig_step(actions)
        obs = {k: v.astype(np.float32) for k, v in obs.items()}
        return obs, rew, term, trunc, info

    env.reset = reset  # type: ignore
    env.step = step    # type: ignore
    return env


# -----------------------------------------------------------------------------
# 3.  RLlib env creator
# -----------------------------------------------------------------------------


def env_creator(config: Dict | None = None):
    parallel_env = _make_parallel_env(**(config or {}))
    parallel_env = _ensure_float32(parallel_env)
    return ParallelPettingZooEnv(parallel_env)


register_env("SingleThrowMultiEnv", env_creator)


# -----------------------------------------------------------------------------
# 4.  Main
# -----------------------------------------------------------------------------


def main(args: argparse.Namespace):
    ray.init()

    # Grab spaces for policy spec
    with env_creator({}) as tmp_env:
        obs_space = tmp_env.observation_space["agent_0"]
        act_space = tmp_env.action_space["agent_0"]

    policies = {"shared": (None, obs_space, act_space, {})}

    cfg = (
        PPOConfig()
        .environment(env="SingleThrowMultiEnv", disable_env_checking=True)
        .framework("torch")
        .env_runners(num_env_runners=args.num_workers)
        .resources(num_gpus=args.num_gpus)
        .training(train_batch_size=args.train_batch_size)
        .multi_agent(policies=policies, policy_mapping_fn=lambda aid, *_: "shared")
    )

    algo = cfg.build()

    # Resolve checkpoint path as absolute URI so pyarrow recognizes it
    checkpoint_root = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_root, exist_ok=True)

    # Restore from checkpoint
    if args.restore_checkpoint:
        try:
            algo.restore_from_path(checkpoint_root)
        except:
            print("Failed to restore checkpoint")

    for itr in range(1, args.num_iters + 1):
        print(f"Beginning iter {itr}")
        result = algo.train()

        # ---- Robust metric extraction -------------------------------------
        runner_metrics = result.get(ENV_RUNNER_RESULTS, {})
        rew_mean = runner_metrics.get(EPISODE_RETURN_MEAN, result.get("episode_reward_mean", float("nan")))
        len_mean = runner_metrics.get(EPISODE_LEN_MEAN, result.get("episode_len_mean", float("nan")))

        print(f"[Iter {itr}] return_mean={rew_mean:.3f} len_mean={len_mean:.1f}")

        if itr % args.checkpoint_freq == 0 or itr == args.num_iters:
            algo.save(checkpoint_root)
            print("checkpoint saved")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=32000)
    parser.add_argument("--checkpoint-freq", type=int, default=25)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_st")
    parser.add_argument("--restore-checkpoint", type=bool, default=True)
    main(parser.parse_args())
