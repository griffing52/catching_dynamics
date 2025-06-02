import numpy as np
from gymnasium_robotics import mamujoco_v1    # <- NEW home of MaMuJoCo

# choose a scenario and a joint-factorisation
env = mamujoco_v1.parallel_env(
    scenario="HalfCheetah",   # any of: Ant, Hopper, Humanoid…
    agent_conf="2x3",         # 2 agents, 3 joints each
    render_mode="human"       # "rgb_array" if headless
)

obs, info = env.reset(seed=0)

terminated = False
while not terminated:
    # sample a random legal action for every agent
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    terminated = all(terminations.values()) or all(truncations.values())

env.close()
