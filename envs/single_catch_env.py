from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces, utils
import numpy as np
from gymnasium_robotics.envs.multiagent_mujoco.obsk import Node, HyperEdge
import os

from mujoco import mj_multiRay

class SingleCatchEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        # Define action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),  # 5 actuators per arm
            dtype=np.float32
        )

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, 10 + 4 + 1),  # 1 agent, 5 joint positions, 5 joint velocities, ball position(2), ball velocity(2), thrower indicator
            dtype=np.float32
        )
        
        # Get the absolute path to the XML file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "models", "single_catch.xml")

        # Set render modes
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": 200,
        }

        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=5,
            observation_space=self.observation_space,
            **kwargs
        )
        
        # Get body IDs for faster access
        self._ball_id = self.model.body('ball').id
        self._left_hand_id = self.model.body('hand0').id
        self._pinch_site_id = self.model.body('pinch site0').id
        self._finger_id = self.model.body('finger0').id
        
        # Get ball joint indices
        self._ball_joint_ids = [
            self.model.joint('ball_x').id,
            self.model.joint('ball_z').id,
            self.model.joint('ball_y').id
        ]
        
        # Define the nodes for each arm
        # Left Arm
        self.left_arm_root = Node("arm_root0", 0, 0, 0)
        self.left_arm_shoulder = Node("arm_shoulder0", 1, 1, 1)
        self.left_arm_elbow = Node("arm_elbow0", 2, 2, 2)
        self.left_arm_wrist = Node("arm_wrist0", 3, 3, 3)
        self.left_grasp = Node("grasp0", 4, 4, 4)

        # Create global nodes for ball and thrower information
        self.ball_node = Node("ball", 5, 5, 5)  # Using index 5 for ball
        self.thrower_node = Node("thrower", 6, 6, 6)  # Using index 6 for thrower

        # Add extra_obs to global nodes
        def get_ball_obs(data):
            ball_pos = [data.xpos[self._ball_id][0], data.xpos[self._ball_id][2]]
            ball_vel = [data.cvel[self._ball_id][3], data.cvel[self._ball_id][5]]
            return np.concatenate([ball_pos, ball_vel])

        def get_thrower_obs(data):
            return np.array([1.0 if self._current_thrower == 'left' else 0.0])

        self.ball_node.extra_obs = {
            "ball": get_ball_obs
        }
        self.thrower_node.extra_obs = {
            "thrower": get_thrower_obs
        }

        # Define global categories
        self.global_categories = {
            "ball": {
                "pos": [0, 1],  # x and z coordinates
                "vel": [2, 3]   # x and z velocities
            },
            "thrower": {
                "indicator": [4]  # single value indicating current thrower
            }
        }

        # Create partitions for each agent (arm)
        self.parts = [
            (  # Left Arm Agent
                self.left_arm_root,
                self.left_arm_shoulder,
                self.left_arm_elbow,
                self.left_arm_wrist,
                self.left_grasp,
            ),
        ]

        # Define the edges (connections between joints)
        self.edges = [
            # Left arm connections
            HyperEdge(self.left_arm_root, self.left_arm_shoulder),
            HyperEdge(self.left_arm_shoulder, self.left_arm_elbow),
            HyperEdge(self.left_arm_elbow, self.left_arm_wrist),
            HyperEdge(self.left_arm_wrist, self.left_grasp),
        ]

        # Task-specific variables
        self._ball = 'ball'
        self._ball_joints = ['_'.join([self._ball, dim]) for dim in 'xzy']
        self._catch_threshold = 0.05  # 5cm threshold for successful catch
        self._throw_force = 5.0  # Force to apply when throwing
        self._current_thrower = 'right'  # Track which arm is throwing
        self.time_limit = 4.0  # Switch every 10 seconds
        self.time_elapsed = 0.0  # Timer for switching

    def _get_action_space(self):
        return self.action_space
        
    def _get_observation_space(self):
        return self.observation_space
    
    def get_global_categories(self):
        return self.global_categories
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        obs = self.reset_model()
        return obs, {}  # Return observation and empty info dict
        
    def step(self, action):
        # Apply the action
        self.do_simulation(action, self.frame_skip)
        
        # Update timer
        self.time_elapsed += self.dt
        
        # Get observations
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._get_reward()
        
        # Check if episode is done
        terminated = self._is_done()
        truncated = False  # We don't have any truncation conditions
        
        # Additional info
        info = {
            'left_grasp': self.data.actuator_force[4],
            'ball_position': self.data.xpos[self._ball_id],
            'ball_velocity': self.data.cvel[self._ball_id][3:],  # Only use linear velocity
            'thrower': self._current_thrower,
            'time_elapsed': self.time_elapsed
        }
        
        return obs, reward, terminated, truncated, info
        
    def _get_obs(self):
        # Left arm indices: 0-4, right arm indices: 5-9
        left_joint_pos = self.data.qpos[:5]
        left_joint_vel = self.data.qvel[:5]

        # For each agent, stack only their own joint pos/vel
        left_obs = np.concatenate([
            left_joint_pos,
            left_joint_vel,
        ], dtype=np.float32)

        # Return a single observation vector instead of stacking
        return np.stack([left_obs], axis=0)
        
    def _get_reward(self):
        ball_pos = self.data.xpos[self._ball_id]
        ball_vel = self.data.cvel[self._ball_id]
        left_hand_pos = self.data.xpos[self._left_hand_id]
        pinch_site_pos = self.data.xpos[self._pinch_site_id]
        ball_target_pos = (left_hand_pos + pinch_site_pos)/2

        arm_center_pos = np.array([-1.5, 0, 0.4])
        arm_full_length = 0.52
        arm_length = np.linalg.norm(ball_target_pos - arm_center_pos)
        wall_height = 0.565686

        dist_reward = 0.0
        if np.linalg.norm(ball_pos - arm_center_pos) < arm_full_length:
            # TODO now that we're within arm's distance most of this math is unnecessary
            # Distance from hand to ball reward
            center_to_ball = ball_pos-arm_center_pos
            ball_dist = np.linalg.norm(center_to_ball)
            target_length = np.clip(ball_dist, 0.0, arm_full_length)
            theta_arm = np.arccos(np.dot(center_to_ball, ball_target_pos) / (ball_dist * np.linalg.norm(ball_target_pos)))
            arm_length_reward = 1 - np.abs((target_length - arm_length) / target_length)
            dist_reward = arm_length_reward * ((np.pi/2) - np.abs((np.pi/2) - theta_arm)) / (np.pi/2)

            # Palm touch reward
            if np.linalg.norm(ball_pos - left_hand_pos) < 0.01:
                dist_reward += 100

        # ball_dist = np.linalg.norm(ball_pos-arm_center_pos)
        # target_length = np.clip(ball_dist, 0.0, arm_length)
        # (arm_center_pos + target_length * (ball_pos - arm_center_pos) / ball_dist) - left_hand_pos
        # normed_dist_to_target = -np.linalg.norm((arm_center_pos + target_length * (ball_pos-arm_center_pos)/ball_dist) - left_hand_pos)
        # dist_reward = -normed_dist_to_target

        # Wrist direction reward
        wrist_reward = 0.0
        # wrist_dir_vec = (pinch_site_pos - ball_target_pos) / np.linalg.norm(pinch_site_pos - ball_target_pos)
        # wrist_theta = np.arctan2(wrist_dir_vec[2], wrist_dir_vec[0])
        # ball_vel_theta = np.arctan2(ball_vel[5], ball_vel[3])
        # ball_to_ball_target = (ball_pos - ball_target_pos) / np.linalg.norm(ball_pos - ball_target_pos)
        # hand_ball_theta = np.arctan2(ball_to_ball_target[2], ball_to_ball_target[0])
        # if ball_pos[2] > wall_height:
        #     wrist_reward = (-np.abs(wrist_theta + ball_vel_theta) + np.pi) / np.pi
        # else:
        #     wrist_reward = (-np.abs(wrist_theta - hand_ball_theta) + np.pi) / np.pi

        return dist_reward + wrist_reward
        
    def _is_done(self):
        ball_pos = self.data.xpos[self._ball_id]
        
        return (ball_pos[2] < -0.1 or  # Ball too low
                abs(ball_pos[0]) > 2.687686 or # Ball too far left/right)
                self.time_elapsed > self.time_limit) # Time
        
    def reset_model(self):
        # Set the velocity
        theta = np.random.uniform(np.pi/2 + np.pi/6, np.pi-np.pi/6)
        far_speed = np.sqrt((2 * 4.905 * (-2.321686 - 1.5))/np.sin(2*theta))
        near_speed = np.sqrt((2 * 4.905 * (-0.578314 - 1.5))/np.sin(2*theta))
        speed = np.random.uniform(near_speed, np.clip(far_speed, near_speed, np.clip(7, near_speed, far_speed)))
        qvel = self.init_qvel.copy()
        qvel[self._ball_joint_ids[0]] = speed * np.cos(theta)  # x velocity
        qvel[self._ball_joint_ids[1]] = speed * np.sin(theta)  # z velocity
        qvel[self._ball_joint_ids[2]] = 0.0   # y angular velocity

        # Apply the new state
        self.set_state(self.init_qpos, qvel)
        
        # Reset thrower and timer
        self._current_thrower = 'right'
        self.time_elapsed = 0.0
        
        return self._get_obs()

    def get_agent_factorization(self):
        """Return the agent factorization for MaMuJoCo"""
        return {
            "partition": self.parts,
            "edges": self.edges,
            "globals": [self.ball_node, self.thrower_node],
            "global_categories": self.global_categories
        }
    
    def raycast(self, starting_angle=0, fov=45, nray=9):
        """
        Calculates raycast result (evenly distributed angularly)--only observes group 3 objects, which includes ball

        Args:
            starting_angle: angle of middle ray in degrees
            fov: field of view in degrees
            nray: number of rays

        Returns:
           Raycast result.
        """
        cam_id = self.model.camera('eye0').id

        origin = np.array(self.data.cam_xpos[cam_id], dtype=np.float64)

        angle_step = fov / (nray - 1)  # Angle step between rays

        angles = np.linspace(starting_angle - fov / 2, starting_angle + fov / 2, nray)  # Generate angles for rays\

        x = np.cos(np.radians(angles))  # Sine for x direction
        y = np.zeros(nray)
        z = np.sin(np.radians(angles))  # Sine for x direction
        directions = np.column_stack((x, y, z))  # Cosine for z direction
        directions = directions.flatten().astype(np.float64)  # Flatten to 1D array

        # Output arrays
        geomid = np.zeros(nray, dtype=np.int32)
        dist = np.zeros(nray, dtype=np.float64)

        # Optional: filter groups
        # geomgroup = None  # or np.array([1]*model.ngeom, dtype=np.uint8)
        # geomgroup = np.array([0]*model.ngeom, dtype=np.uint8)
        # for i in range(model.ngeom):
        #     if model.geom_group[i] == 3:
        #         geomgroup[i] = 1
        geomgroup = np.array([0,1,1,1,1,1], dtype=np.uint8)

        # Call raycast
        mj_multiRay(self.model, self.data, origin, directions, 
                        geomgroup, 
                        flg_static=1,            # Include static geoms
                        bodyexclude=-1,          # No exclusion
                        geomid=geomid,
                        dist=dist,
                        nray=nray,
                        cutoff=10.0)             # Max ray length
        
        # for i in range(nray):
        #     if dist[i] > 0:
        #         print(f"Ray {i}: Hit geom {geomid[i]} at distance {dist[i]}")

        return geomid, dist