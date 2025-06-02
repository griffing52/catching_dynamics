import os
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from envs.single_catch_env import SingleCatchEnv
from gymnasium_robotics import mamujoco_v1
import mujoco
import numpy as np
import time

# ---------- 1. create & register the exact env ----------
def make_parallel_env(**cfg):
    base = SingleCatchEnv(**cfg)
    return mamujoco_v1.parallel_env(
        scenario="SingleCatch",
        agent_conf="1x5",
        agent_factorization=base.get_agent_factorization(),
        global_categories=base.get_global_categories(),
        gym_env=base,
        render_mode=None,          # will override below
    )

def env_creator(env_config=None):
    # render_mode set later by PettingZooEnv.render()
    return ParallelPettingZooEnv(make_parallel_env())

register_env("SingleCatchMultiEnv", env_creator)

# ---------- 2. init Ray BEFORE loading ----------
ray.init(ignore_reinit_error=True)    # no address needed for local

# ---------- 3. load whole algorithm ----------
ckpt_dir = "checkpoints_sc"           # <-- folder that holds algorithm_state.pkl
ckpt_root = os.path.abspath(ckpt_dir)
os.makedirs(ckpt_root, exist_ok=True)
algo = Algorithm.from_checkpoint(ckpt_root)

# ---------- 4. create a renderable env ----------
base = SingleCatchEnv(render_mode=None)
base_wrapper = mamujoco_v1.parallel_env(
        scenario="SingleCatch",
        agent_conf="1x5",
        agent_factorization=base.get_agent_factorization(),
        global_categories=base.get_global_categories(),
        gym_env=base,
        render_mode=None,          # will override below
    )
env = ParallelPettingZooEnv(base_wrapper)

# ---------- 5. play one episode ----------
# for i in range(5):
#     obs, _ = env.reset(seed=0)
#     done = {"__all__": False}

#     cam = env.renderer.cam          # gymnasium uses mujoco.MjvCamera
#     cam.type = mujoco.mjtCamera.mjCAMERA_FREE
#     cam.fovy = 70                   # widen lens to 70°
#     cam.distance *= 1.8             # or just zoom further out
#     while not done["__all__"]:
#         actions = {aid: algo.compute_single_action(o, explore=False)
#                     for aid, o in obs.items()}
#         obs, _, done, _, _ = env.step(actions)
#         env.render()
#         time.sleep(0.01)
# env.close()
# algo.stop()

POLICY_PATH = "checkpoints_sc/learner_group/learner/rl_module/shared"
policy_root = os.path.abspath(POLICY_PATH)
os.makedirs(policy_root, exist_ok=True)
policy = Policy.from_checkpoint(policy_root)

# Reset the environment
obs, info = env.reset(seed=0)
print(obs, "OBS")

# Initialize episode tracking
episode_reward = 0
step_count = 0
max_steps = 1000  # Maximum steps per episode

# Slow down the simulation
base.unwrapped.model.opt.timestep = 0.01  # Slower physics timestep
base.unwrapped.model.opt.iterations = 20  # More physics iterations per step

print("Starting environment test...")
print("Press Ctrl+C to exit")

# Use MuJoCo's native viewer
with mujoco.viewer.launch_passive(base.model, base.data) as viewer:
    try:
        while True:
            step_start = time.time()

            # actions = {aid: algo.compute_single_action(o, explore=False)
            #         for aid, o in obs.items()}
            
            # actions, _, _ = algo.compute_actions(
            #     obs,                 # the full dict {agent_id: obs}
            #     explore=False,
            # )

            # shared = algo.get_policy("shared")            # name from PPOConfig
            # actions = {aid: shared.compute_single_action(obs_i, explore=False)[0]
                    # for aid, obs_i in obs.items()}     # [0] extracts the action; fn returns (action, state, info)

            actions, _, _ = algo.compute_actions(obs, explore=False)

            # Step the environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Update episode tracking
            episode_reward += sum(rewards.values())
            step_count += 1
            
            # Print information every 100 steps
            if step_count % 100 == 0:
                print(f"\nStep {step_count}")
                print(f"Current thrower: {infos['agent_0']['thrower']}")
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
            time_until_next_step = base.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
    except KeyboardInterrupt:
        print("\nTest terminated by user")
    finally:
        env.close()
        print("Environment closed")