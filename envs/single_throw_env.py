from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces, utils
import numpy as np
from gymnasium_robotics.envs.multiagent_mujoco.obsk import Node, HyperEdge
import os

class SingleThrowEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, 10 + 4 + 1), dtype=np.float32
        )

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "models", "single_throw.xml")

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": 200,
        }

        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, model_path, 5, self.observation_space, **kwargs)

        # Body IDs
        self._ball_id = self.model.body("ball").id
        self._hand_id = self.model.body("hand0").id
        self._pinch_site_id = self.model.body('pinch site0').id
        self._finger_id = self.model.body('finger0').id

        # Ball joints (x, z, y)
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
        self.catcher_node = Node("catcher", 6, 6, 6)  # Using index 6 for thrower

        # Add extra_obs to global nodes
        def get_ball_obs(data):
            ball_pos = [data.xpos[self._ball_id][0], data.xpos[self._ball_id][2]]
            ball_vel = [data.cvel[self._ball_id][3], data.cvel[self._ball_id][5]]
            return np.concatenate([ball_pos, ball_vel])

        def get_catcher_obs(data):
            return np.array([1.0 if self._current_catcher == 'left' else 0.0])

        self.ball_node.extra_obs = {
            "ball": get_ball_obs
        }
        self.catcher_node.extra_obs = {
            "catcher": get_catcher_obs
        }

        # Define global categories
        self.global_categories = {
            "ball": {
                "pos": [0, 1],  # x and z coordinates
                "vel": [2, 3]   # x and z velocities
            },
            "catcher": {
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

        # TASK SPECIFIC VARS
        self._target_x = 3.0  # target zone x position
        self._target_radius = 0.2
        self._min_throw_velocity = 0.25
        self._current_catcher = 'right'  # Track which arm is throwing
        self._max_time = 4.0
        self.time_elapsed = 0.0

    def _get_action_space(self):
        return self.action_space
        
    def _get_observation_space(self):
        return self.observation_space
    
    def get_global_categories(self):
        return self.global_categories

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.reset_model(), {}

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # # Place ball in hand
        # ball_offset = np.array([0.0, 0.0, 0.0])  # relative to hand
        # hand_pos = self.data.xpos[self._hand_id]
        # ball_pos = hand_pos + ball_offset

        # print(hand_pos)

        # init_ball_pos = np.array([-1.5, 0.0, 1.0])

        # # Manually override ball position
        # qpos[self._ball_joint_ids[0]] = init_ball_pos[0]
        # qpos[self._ball_joint_ids[1]] = init_ball_pos[2]
        # qpos[self._ball_joint_ids[2]] = init_ball_pos[1]  # y-angle

        self.set_state(qpos, qvel)
        self.time_elapsed = 0.0

        self._current_catcher = 'right'

        return self._get_obs()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.time_elapsed += self.dt

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_done()
        truncated = False
        info = {
            'left_grasp': self.data.actuator_force[4],
            'ball_position': self.data.xpos[self._ball_id],
            'ball_velocity': self.data.cvel[self._ball_id][3:],  # Only use linear velocity
            'catcher': self._current_catcher,
            'time_elapsed': self.time_elapsed
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        joint_pos = self.data.qpos[:5]
        joint_vel = self.data.qvel[:5]

        return np.stack([np.concatenate([joint_pos, joint_vel], dtype=np.float32)], axis=0)

    def _get_reward(self):
        ball_pos = self.data.xpos[self._ball_id]
        ball_vel = self.data.cvel[self._ball_id][3:]

        # Dense shaping reward
        reward = 0.0

        # 1. Strong reward for forward x-velocity and position
        reward += 5.0 * ball_vel[0]
        if(ball_pos[0] > 0.25): reward += ball_pos[0]

        # for contact in self.data.contact:
        #     if contact.geom1 == self.model.geom("ball") or contact.geom2 == self.model.geom("ball"):
        #         print("Ball is in contact with the hand!")


        # # 2. Weaker reward for upward y-velocity
        # reward += 1.0 * ball_vel[2]

        # # 3. Bonus for proximity to target (within 1.0m)
        # distance_to_target = abs(ball_pos[0] - self._target_x)
        # reward += 1.0 * max(0.0, 1.0 - distance_to_target)

        # # 4. Success bonus: ball is thrown strong and is near target
        # if np.linalg.norm(ball_vel[:2]) > self._min_throw_velocity:
        #     if distance_to_target < self._target_radius:
        #         reward += 10.0  # strong bonus for a good throw
        #     else:
        #         reward += 5.0 * (2.0 - np.clip(distance_to_target, 0.0, 2.0))

        return reward

    def _is_done(self):
        ball_pos = self.data.xpos[self._ball_id]
        return (
            ball_pos[2] < -0.1 or
            self.time_elapsed > self._max_time
        )
    
    def get_agent_factorization(self):
        """Return the agent factorization for MaMuJoCo"""
        return {
            "partition": self.parts,
            "edges": self.edges,
            "globals": [self.ball_node, self.catcher_node],
            "global_categories": self.global_categories
        }