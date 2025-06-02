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
import torch

from envs.single_catch_env import SingleCatchEnv          # your Gymnasium base
from gymnasium_robotics import mamujoco_v1
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

def _make_parallel_env(**cfg):
    base = SingleCatchEnv(**cfg)
    return mamujoco_v1.parallel_env(
        scenario="SingleCatch",
        agent_conf="1x5",
        agent_factorization=base.get_agent_factorization(),
        global_categories=base.get_global_categories(),
        gym_env=base,
        render_mode=None,
    )

def env_creator(config=None):
    return ParallelPettingZooEnv(_make_parallel_env())

register_env("SingleCatchMultiEnv", env_creator)

ckpt = "/Users/sohampatil/Desktop/catching_dynamics/checkpoints_sc/"

ray.init(ignore_reinit_error=True)

# Create RLModule from a checkpoint.
rl_module = RLModule.from_checkpoint(
    os.path.join(
        ckpt, # (or directly a string path to the checkpoint dir)
        COMPONENT_LEARNER_GROUP,
        COMPONENT_LEARNER,
        COMPONENT_RL_MODULE,
        "shared",
    )
)

env = env_creator({})
obs, _ = env.reset(seed=0)

action_dist_inputs = rl_module.forward_inference({'obs': torch.from_numpy(obs['agent_0']).float()})['action_dist_inputs']

distr = torch.distributions.Normal(
    loc=action_dist_inputs[:5],
    scale=torch.exp(action_dist_inputs[5:])
)
action = distr.sample().numpy()

next_obs, rewards, terms, truncs, infos = env.step(action)

print("actions dict:", action)
print("reward for this step:", sum(rewards.values()))
env.close()
ray.shutdown()