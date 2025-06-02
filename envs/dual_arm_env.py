from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces, utils
import numpy as np
from gymnasium_robotics.envs.multiagent_mujoco.obsk import Node, HyperEdge
import os

class DualArmEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        # Define action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(10,),  # 5 actuators per arm
            dtype=np.float32
        )

        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2, 10 + 4 + 1),  # 2 agents, 5 joint positions, 5 joint velocities, ball position(2), ball velocity(2), thrower indicator
            dtype=np.float32
        )
        
        # Get the absolute path to the XML file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "models", "dual_arm.xml")

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
        self._right_hand_id = self.model.body('hand1').id
        
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

        # Right Arm
        self.right_arm_root = Node("arm_root1", 5, 5, 5)
        self.right_arm_shoulder = Node("arm_shoulder1", 6, 6, 6)
        self.right_arm_elbow = Node("arm_elbow1", 7, 7, 7)
        self.right_arm_wrist = Node("arm_wrist1", 8, 8, 8)
        self.right_grasp = Node("grasp1", 9, 9, 9)

        # Create global nodes for ball and thrower information
        self.ball_node = Node("ball", 10, 10, 10)  # Using index 10 for ball
        self.thrower_node = Node("thrower", 11, 11, 11)  # Using index 11 for thrower

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
            (  # Right Arm Agent
                self.right_arm_root,
                self.right_arm_shoulder,
                self.right_arm_elbow,
                self.right_arm_wrist,
                self.right_grasp,
            ),
        ]

        # Define the edges (connections between joints)
        self.edges = [
            # Left arm connections
            HyperEdge(self.left_arm_root, self.left_arm_shoulder),
            HyperEdge(self.left_arm_shoulder, self.left_arm_elbow),
            HyperEdge(self.left_arm_elbow, self.left_arm_wrist),
            HyperEdge(self.left_arm_wrist, self.left_grasp),
            
            # Right arm connections
            HyperEdge(self.right_arm_root, self.right_arm_shoulder),
            HyperEdge(self.right_arm_shoulder, self.right_arm_elbow),
            HyperEdge(self.right_arm_elbow, self.right_arm_wrist),
            HyperEdge(self.right_arm_wrist, self.right_grasp),
        ]

        # Task-specific variables
        self._ball = 'ball'
        self._ball_joints = ['_'.join([self._ball, dim]) for dim in 'xzy']
        self._catch_threshold = 0.05  # 5cm threshold for successful catch
        self._throw_force = 5.0  # Force to apply when throwing
        self._current_thrower = 'left'  # Track which arm is throwing
        self._switch_interval = 10.0  # Switch every 10 seconds
        self._time_since_last_switch = 0.0  # Timer for switching

    def _get_action_space(self):
        return self.action_space
        
    def _get_observation_space(self):
        return self.observation_space
    
    def get_global_categories(self):
        return self.global_categories
        
    def step(self, action):
        # Apply the action
        self.do_simulation(action, self.frame_skip)
        
        # Update timer
        self._time_since_last_switch += self.dt
        
        # Check if it's time to switch thrower
        if self._time_since_last_switch >= self._switch_interval:
            self._current_thrower = 'right' if self._current_thrower == 'left' else 'left'
            self._time_since_last_switch = 0.0
        
        # Get observations
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._get_reward()
        
        # Check if episode is done
        done = self._is_done()
        
        # Additional info
        info = {
            'left_grasp': self.data.actuator_force[4],
            'right_grasp': self.data.actuator_force[9],
            'ball_position': self.data.xpos[self._ball_id],
            'ball_velocity': self.data.cvel[self._ball_id][3:],  # Only use linear velocity
            'thrower': self._current_thrower,
            'time_since_switch': self._time_since_last_switch
        }
        
        return obs, reward, done, False, info
        
    def _get_obs(self):
        # Each agent gets only their own joint positions and velocities
        thrower_indicator = 1.0 if self._current_thrower == 'left' else 0.0

        # Left arm indices: 0-4, right arm indices: 5-9
        left_joint_pos = self.data.qpos[:5]
        right_joint_pos = self.data.qpos[5:10]
        left_joint_vel = self.data.qvel[:5]
        right_joint_vel = self.data.qvel[5:10]

        # For each agent, stack only their own joint pos/vel
        left_obs = np.concatenate([
            left_joint_pos,
            left_joint_vel,
        ])
        right_obs = np.concatenate([
            right_joint_pos,
            right_joint_vel,
        ])

        # Stack as (2, 10) for (n_agents, obs_dim)
        obs = np.stack([left_obs, right_obs], axis=0)

        print(obs[0], "single")

        return obs
        
    def _get_reward(self):
        ball_pos = self.data.xpos[self._ball_id]
        ball_vel = self.data.cvel[self._ball_id][3:]  # Only use linear velocity
        
        # Get hand positions and velocities
        left_hand_pos = self.data.xpos[self._left_hand_id]
        right_hand_pos = self.data.xpos[self._right_hand_id]
        left_hand_vel = self.data.cvel[self._left_hand_id][3:]  # Only use linear velocity
        right_hand_vel = self.data.cvel[self._right_hand_id][3:]  # Only use linear velocity
        
        # Calculate distances to ball
        left_dist = np.linalg.norm(ball_pos - left_hand_pos)
        right_dist = np.linalg.norm(ball_pos - right_hand_pos)
        
        # Calculate velocity differences
        left_vel_diff = np.linalg.norm(ball_vel - left_hand_vel)
        right_vel_diff = np.linalg.norm(ball_vel - right_hand_vel)
        
        # Reward structure:
        # 1. Reward for successful catch
        # 2. Penalty for dropping the ball
        # 3. Linear reward for hand proximity to ball
        # 4. Reward for controlling the ball's motion
        
        catch_reward = 0
        if self._current_thrower == 'left':
            if right_dist < self._catch_threshold and np.linalg.norm(ball_vel) < 0.1:
                catch_reward = 10.0
            # Add proximity reward for the catching hand (right)
            proximity_reward = -right_dist  # Negative distance = closer is better
            
            # Control reward: encourage the catching hand to influence the ball's motion
            # This is high when the ball's velocity is similar to the hand's velocity
            # AND the hand is close to the ball (indicating control)
            control_reward = 0.0
            if right_dist < 0.1:  # Only consider control when hand is close to ball
                # Higher reward when velocities match and hand is close
                control_reward = -right_vel_diff * (1.0 - right_dist/0.1)
        else:
            if left_dist < self._catch_threshold and np.linalg.norm(ball_vel) < 0.1:
                catch_reward = 10.0
            # Add proximity reward for the catching hand (left)
            proximity_reward = -left_dist  # Negative distance = closer is better
            
            # Control reward for left hand
            control_reward = 0.0
            if left_dist < 0.1:  # Only consider control when hand is close to ball
                control_reward = -left_vel_diff * (1.0 - left_dist/0.1)
        
        # Penalty for dropping the ball
        drop_penalty = -5.0 if ball_pos[2] < 0.1 else 0.0

        # Scale the rewards to be comparable
        proximity_reward *= 2.0  # Adjust this scaling factor as needed
        control_reward *= 5.0    # Give high weight to control
        
        return catch_reward + drop_penalty + proximity_reward + control_reward
        
    def _is_done(self):
        ball_pos = self.data.xpos[self._ball_id]
        
        # Episode ends if:
        # 1. Ball falls below a certain height
        # 2. Ball goes too far from the workspace
        return (ball_pos[2] < 0.01 or  # Ball too low
                abs(ball_pos[0]) > 3.0)  # Ball too far left/right)
        
    def reset_model(self):
        # Reset the model to initial state
        self.set_state(
            self.init_qpos,
            self.init_qvel
        )
        
        # Reset thrower and timer
        self._current_thrower = 'left'
        self._time_since_last_switch = 0.0
        
        return self._get_obs()

    def get_agent_factorization(self):
        """Return the agent factorization for MaMuJoCo"""
        return {
            "partition": self.parts,
            "edges": self.edges,
            "globals": [self.ball_node, self.thrower_node],
            "global_categories": self.global_categories
        }