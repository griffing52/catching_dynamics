#!/usr/bin/env python3
"""
RLlib training script for the MaMuJoCo **SingleCatch** task with a 1×5 agent configuration.

*   Converts your custom `SingleCatchEnv` into a PettingZoo parallel‑style multi‑agent
    environment exactly as shown in **single_catch_visual.py**.
*   Wraps the environment with RLlib via `PettingZooEnv` so you get seamless
    multi‑agent support (policies, policy mapping, rollouts, evaluation, checkpoints …).
*   Uses PPO by default, but you can swap in any RLlib algorithm that supports
    multi‑agent execution.

Run with e.g.:
```bash
python train_single_catch_rllib.py --num-iters 500 --num-workers 8 --checkpoint-freq 25
```
The script prints key metrics every iteration and writes checkpoints to the
`checkpoints/` directory (override with `--checkpoint-dir`).
"""

import argparse
import os
from typing import Dict

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

# ---- 1.  Build the MaMuJoCo parallel environment --------------------------------

from gymnasium_robotics import mamujoco_v1
from single_catch_env import SingleCatchEnv


def _make_parallel_env(**overrides):
    """Factory that reproduces the `parallel_env` call in single_catch_visual.py."""

    base_env = SingleCatchEnv(**overrides)

    # Convert the single‑agent Gymnasium env into a MaMuJoCo ParallelEnv.
    env = mamujoco_v1.parallel_env(
        scenario="SingleCatch",
        agent_conf="1x5",  # one arm with five actuated joints
        agent_factorization=base_env.get_agent_factorization(),
        global_categories=base_env.get_global_categories(),
        gym_env=base_env,
        render_mode=None,
    )
    return env


# ---- 2.  RLlib‑compatible creator ------------------------------------------------


def env_creator(config: Dict | None = None):
    """RLlib expects a callable that returns a *new* env instance."""

    parallel_env = _make_parallel_env()

    # PettingZooEnv turns any ParallelEnv/AECEnv into an RLlib MultiAgentEnv.
    return PettingZooEnv(parallel_env)


register_env("SingleCatchMultiEnv", env_creator)


# ---- 3.  Main training loop ------------------------------------------------------


def main(args):
    ray.init()

    # Peek at the env once so we can grab spaces for the policy spec.
    tmp_env = env_creator({})
    obs_space = tmp_env.observation_space("agent_0")
    act_space = tmp_env.action_space("agent_0")
    tmp_env.close()

    # Shared policy for all agents (there is only one agent in the 1×5 setup,
    # but this will seamlessly scale if you switch to, say, 2×5).
    policies = {
        "shared_policy": (None, obs_space, act_space, {}),
    }

    def policy_mapping_fn(agent_id, *_, **__):  # noqa: D401
        """Map every agent to the single shared policy."""
        return "shared_policy"

    config = (
        PPOConfig()
        .environment(env="SingleCatchMultiEnv")
        .framework("torch")  # switch to "tf2" if you prefer
        .rollouts(num_rollout_workers=args.num_workers)
        .resources(num_gpus=args.num_gpus)
        .training(train_batch_size=args.train_batch_size)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .evaluation(
            evaluation_interval=10,
            evaluation_num_episodes=10,
            evaluation_config={"explore": False},
        )
    )

    algo = config.build()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for itr in range(1, args.num_iters + 1):
        result = algo.train()
        print(
            f"[Iter {itr}] reward_mean={result['episode_reward_mean']:.3f} "
            f"len_mean={result['episode_len_mean']:.1f} time={result['time_total_s']:.1f}s"
        )

        if itr % args.checkpoint_freq == 0 or itr == args.num_iters:
            ckpt_path = algo.save(args.checkpoint_dir)
            print(f"Saved checkpoint to {ckpt_path}\n")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=200,
                        help="Training iterations (calls to algo.train)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel rollout workers")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="GPUs to allocate to the trainer")
    parser.add_argument("--train-batch-size", type=int, default=32000,
                        help="Timesteps per SGD round (increase for stable PPO)")
    parser.add_argument("--checkpoint-freq", type=int, default=50,
                        help="Save a checkpoint every N iterations")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to store RLlib checkpoints")
    main(parser.parse_args())
