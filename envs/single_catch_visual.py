from gymnasium_robotics import mamujoco_v1
from single_catch_env import SingleCatchEnv
import numpy as np
import time
import mujoco
import mujoco.viewer

def test_environment():
    # Create the base environment
    base_env = SingleCatchEnv()
    
    # Create the multi-agent environment
    env = mamujoco_v1.parallel_env(
        scenario="SingleCatch",
        agent_conf="1x5",  # 1 agent, 5 joints each (including grasp)
        agent_factorization=base_env.get_agent_factorization(),
        global_categories=base_env.get_global_categories(),
        gym_env=base_env,
        render_mode=None  # We'll use MuJoCo's native viewer
    )
    
    # Reset the environment
    obs, info = env.reset(seed=0)
    
    # Initialize episode tracking
    episode_reward = 0
    step_count = 0
    max_steps = 1000  # Maximum steps per episode
    
    # Slow down the simulation
    base_env.model.opt.iterations = 20  # More physics iterations per step
    
    print("Starting environment test...")
    print("Press Ctrl+C to exit")

    # Use MuJoCo's native viewer
    with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
        try:
            while True:
                step_start = time.time()
                
                # Get actions for each agent (random for testing)
                # actions = {
                #     "agent_0": env.action_space("agent_0").sample(),  # Left arm actions
                #     "agent_1": env.action_space("agent_1").sample()   # Right arm actions
                # }
                actions = {
                    "agent_0": np.array([0.0, -0.0, 0.0, -0.0, 0.0]),
                }
                
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
                time_until_next_step = base_env.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                
        except KeyboardInterrupt:
            print("\nTest terminated by user")
        finally:
            env.close()
            print("Environment closed")

if __name__ == "__main__":
    test_environment()