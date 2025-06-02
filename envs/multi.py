from gymnasium_robotics import mamujoco_v1
from dual_arm_env import DualArmEnv
import numpy as np
import time

def test_environment():
    # Create the base environment
    base_env = DualArmEnv()

    # Create the multi-agent environment
    env = mamujoco_v1.parallel_env(
        scenario="DualArm",
        agent_conf="2x5",  # 2 agents, 5 joints each (including grasp)
        agent_factorization=base_env.get_agent_factorization(),
        gym_env=base_env,
        render_mode="human"
    )

    # Reset the environment
    obs, info = env.reset(seed=0)
    
    # Initialize episode tracking
    episode_reward = 0
    step_count = 0
    max_steps = 1000  # Maximum steps per episode
    
    print("Starting environment test...")
    print("Press Ctrl+C to exit")

    try:
        while True:
            # Get actions for each agent (random for testing)
            actions = {
                "agent_0": env.action_space("agent_0").sample(),  # Left arm actions
                "agent_1": env.action_space("agent_1").sample()   # Right arm actions
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
                print(f"Ball position: {infos['agent_0']['ball_velocity']}")
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
            
            # Add a small delay to make the visualization more visible
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nTest terminated by user")
    finally:
        env.close()
        print("Environment closed")

if __name__ == "__main__":
    test_environment()