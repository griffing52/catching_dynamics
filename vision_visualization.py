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

from envs.single_catch_env import SingleCatchEnv          # your Gymnasium base
from gymnasium_robotics import mamujoco_v1
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from collections import deque
from vision.CNN_LSTM import GatedFusionModel, VisualCNN

base_env = SingleCatchEnv()

def env_creator(config=None):
    return ParallelPettingZooEnv(
        mamujoco_v1.parallel_env(
            scenario="SingleCatch",
            agent_conf="1x5",
            agent_factorization=base_env.get_agent_factorization(),
            global_categories=base_env.get_global_categories(),
            gym_env=base_env,
            render_mode=None,
        )
    )

register_env("SingleCatchMultiEnv", env_creator)

ray.init(ignore_reinit_error=True)

ckpt_dir = "checkpoints_sc/"
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
base_env.model.opt.timestep = 0.01  # Slower physics timestep
base_env.model.opt.iterations = 20  # More physics iterations per step

print("Loading vision model...")
T=5 # Number of time steps to consider for the LSTM
position_buffer = deque(maxlen=T)
raycast_buffer = deque(maxlen=T)

# Initialize buffers for position and raycast data
for _ in range(T):
    position_buffer.append(torch.tensor([np.random.normal(loc=-1.27113819, scale=0.7155363), np.random.normal(loc=0.35084985, scale=0.54217707)]))  # start at (0, 0)
    raycast_buffer.append(torch.ones(107) * -1)  # placeholder for no hit

# vision_model = GatedFusionModel(num_rays=107)
# vision_model.load_state_dict(torch.load("vision/ray_transformer.pth"))
gated_model = GatedFusionModel(num_rays=107)
gated_model.load_state_dict(torch.load("vision/ray_transformer.pth"))
vision_model = VisualCNN(num_rays=107)
vision_model.load_state_dict(torch.load("vision/cnn.pth"))
# vision_model.load_state_dict(torch.load("vision/gated_model.pth"))
print("Vision model loaded successfully")

print("Starting environment test...")
print("Press Ctrl+C to exit")

def add_lines_to_viewer(viewer, points, color, width=0.005):
    """
    Adds a set of lines to the passive viewer scene.

    Args:
        viewer: The viewer handle returned by mujoco.viewer.launch_passive.
        points: A list of (start_point, end_point) tuples.
                Each point should be a 3D numpy array.
        color: A list or numpy array of 4 floats (r, g, b, a).
        width: The width of the lines.
    """
    # Reset the number of geoms to 0 to clear previous lines
    viewer.user_scn.ngeom = 0
    
    for start, end in points:
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break  # Stop if we've run out of geoms

        # Get the next available geom from the scene
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]

        # === THE FIX IS HERE ===
        # 1. Set the color of the geom directly.
        #    Use slicing [:] to ensure the array is modified in place.
        geom.rgba[:] = color
        
        # 2. Call mjv_makeConnector WITHOUT the color arguments.
        mujoco.mjv_makeConnector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            start[0], start[1], start[2],
            end[0], end[1], end[2]
        )
        
        # Increment the geom counter
        viewer.user_scn.ngeom += 1

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

                starting_angle = -30
                fov = 60
                nray = 107
                max_range = 5.5 # Define a maximum range for visualization
                geomid, dist = base_env.raycast(starting_angle, fov, nray, max_range)
                
                # Visualize rays
                ray_points = []
                for i in range(nray):
                    # Get the starting point of the ray (e.g., the position of the sensor)
                    start_pos = base_env.data.cam_xpos[base_env.model.camera('eye0').id]

                    # Calculate the direction of the ray
                    angle = np.deg2rad(starting_angle - fov/2 + i * fov / (nray - 1))
                    direction = np.array([np.cos(angle), 0, np.sin(angle)]) # Assuming 2D rays on the XY plane

                    # Determine the end point of the 
                    if geomid[i] != -1: # -1 indicates no collision
                        end_pos = start_pos + dist[i] * direction
                    else:
                        # If no collision, draw the ray to its maximum length
                        end_pos = start_pos + max_range * direction
                    ray_points.append((start_pos, end_pos))

                add_lines_to_viewer(viewer, ray_points, color=[0, 1, 0, 1], width=0.02)
                
                ray_output = torch.tensor(dist, dtype=torch.float32)

                # ball_pos = [base_env.data.xpos[base_env._ball_id][0], base_env.data.xpos[base_env._ball_id][2]]
                # ball_vel = [base_env.data.cvel[base_env._ball_id][3], base_env.data.cvel[base_env._ball_id][5]]

                # Prepare input for the vision model
                raycast_buffer.append(ray_output)
                ray_input = torch.stack(list(raycast_buffer)).unsqueeze(0)  # (1, T, R)
                # pos_input = torch.stack(list(position_buffer)).unsqueeze(0)  # (1, T, 2)
                # pos_input = torch.stack([torch.as_tensor(p, dtype=torch.float32) for p in position_buffer]).unsqueeze(0)

                vision_model.eval()
                gated_model.eval()
                with torch.no_grad():
                    pos_seq = vision_model(ray_input.squeeze(0))  # (1,5,2)
                    
                    position_buffer.append(pos_seq.numpy()[-1, :])
                    prediction = gated_model(ray_input, pos_seq.unsqueeze(0)).numpy()[0]

                    pos_pred = prediction[:2]
                    vel_pred = prediction[2:]
                    

                # Update position buffer with the predicted position
                # position_buffer.append(pos_pred)
                # position_buffer.append(torch.tensor(pos_pred, dtype=torch.float32))

                # base_env.data.qpos[base_env._target_ball_id][0] = pos_pred[0]
                # base_env.data.qpos[base_env._target_ball_id][2] = pos_pred[1]
                # base_env.model.body_pos[base_env._target_ball_id, ['x', 'z']] = pos_pred[0], pos_pred[1]

                base_env.model.body_pos[base_env._target_ball_id, 0] = pos_pred[0]  # Update x
                base_env.model.body_pos[base_env._target_ball_id, 2] = pos_pred[1]
                
                # Print information every 100 steps
                if step_count % 50 == 0:
                    print(pos_pred, (base_env.data.xpos[base_env._ball_id][0], base_env.data.xpos[base_env._ball_id][2]))

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

                    for _ in range(T):
                        position_buffer.append(torch.tensor([np.random.normal(loc=-1.27113819, scale=0.7155363), np.random.normal(loc=0.35084985, scale=0.54217707)]))  # start at (0, 0)
                        raycast_buffer.append(torch.ones(107) * -1)

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