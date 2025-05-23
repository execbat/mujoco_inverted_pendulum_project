import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from inverted_pendulum_env import InvertedPendulumEnv
from gymnasium.wrappers import TimeLimit

class InvertedPendulumGymEnv_0(gym.Env):
    """
    Custom Gym environment for controlling an inverted pendulum using MuJoCo.

    Goal:
    - Stabilize the pole tip vertically.

    Supports continuous action space.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render: bool = False, spawn_ball_every: int = 500):
        """
        Initialize the environment.

        Args:
            render (bool): Whether to render the simulation.
            spawn_ball_every (int): Steps between spawning new targets.
        """
        super().__init__()
        self.render = render

        # Pole parameters
        self.l = 0.6  # Pole length
        self.pole_mass = 10.0
        self.g = 9.81
        self.max_tip_speed = 1

        # Cart parameters
        self.cart_mass = 10.0

        # Simulation
        self.sim = InvertedPendulumEnv(render=self.render)

        # State
        self.step_num = 0
        self.ball_pose = None #self.draw_ball()
        self.spawn_ball_every = spawn_ball_every
        self.previous_observation = None
        self.previous_dist_to_vertical = None

        # Inactivity parameters
        self.small_change_threshold = 0.1
        self.small_change_penalty = 1

        # Derivatives
        self.dt = 0.02
        self.v_tip_previous = 0

        # Target parameters
        self.vertical_point = None
        self.target_zone_radius = 0.05
        self.v_tip_target_in_zone = 0.1
        self.hold_bonus = 5.0
        self.acceleration_threshold = 0.000023

        # Spaces
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.

        Returns:
            observation (np.ndarray), info (dict)
        """
        super().reset(seed=seed)
        self.step_num = 0

        # Randomize pole mass
        self.pole_mass = random.uniform(10.0, 30.0)
        self.sim.set_new_pole_mass(self.pole_mass)

        observation = self.sim.reset_model()
        self.draw_ball()
        self.calc_vertical_point(observation)
        
        gravity_moment = self.calc_gravity_moment(observation)
        self.pole_end_point = self.calc_pole_end_point(observation)
        dist_to_target_pt = self.calc_euclidean(self.ball_pose, self.pole_end_point)
        dist_to_vertical_pt = self.arc_distance_to_vertical()
        v_tip_magnitude, vx_tip, vz_tip = self.get_pole_tip_velocity(observation, include_cart_speed = False) # speed og the pole tip exclude cart speed
        current_a_tip = self.compute_current_tip_acceleration(v_tip_magnitude)

        observation = np.concatenate([
            observation,
            [dist_to_vertical_pt],
            [dist_to_target_pt], 
            [self.pole_end_point[0]],
            [self.pole_end_point[2]],
            [self.ball_pose[0]],
            [self.ball_pose[2]],
            [self.pole_mass / 10.0],
            [gravity_moment / 10.0],            
            [v_tip_magnitude],
            [current_a_tip]
        ]).ravel()
        
        self.previous_dist_to_vertical = dist_to_vertical_pt
        return observation.astype(np.float32), {}

    def step(self, action: np.ndarray):
        """
        Perform one step in the environment.

        Args:
            action (np.ndarray): Action to apply.

        Returns:
            observation (np.ndarray), reward (float), terminated (bool), truncated (bool), info (dict)
        """
        self.step_num += 1
        reward = 0

        if self.step_num % self.spawn_ball_every == 0:
            self.draw_ball()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        observation, _, terminated = self.sim.step(action)
        truncated = False
        self.calc_vertical_point(observation)

        if abs(observation[0]) >= 1.0:
            cart_hit_penalty = 1.0
        else:
            cart_hit_penalty = 0
        
        gravity_moment = self.calc_gravity_moment(observation)
        #print("gravity_moment", gravity_moment)
        self.pole_end_point = self.calc_pole_end_point(observation)
        dist_to_target_pt = self.calc_euclidean(self.ball_pose, self.pole_end_point)
        dist_to_vertical_pt = self.arc_distance_to_vertical()
        v_tip_magnitude, vx_tip, vz_tip = self.get_pole_tip_velocity(observation, include_cart_speed = False) # speed og the pole tip exclude cart speed
        v_tip_and_cart, _, _ = self.get_pole_tip_velocity(observation, include_cart_speed = True)
        
       
        current_a_tip = self.compute_current_tip_acceleration(v_tip_magnitude)
        moment_compensation_bonus = self.calc_gravity_moment_bonus(observation, dist_to_vertical_pt, dist_to_target_pt)
        tip_speed_bonus_to_vertical = self.compute_tip_speed_bonus_to_vertical(dist_to_vertical_pt, v_tip_magnitude)
        tip_accel_bonus_to_vertical = self.compute_tip_acceleration_bonus(dist_to_vertical_pt, current_a_tip, v_tip_magnitude, observation[3])
        dist_to_target_bonus = self.calc_dist_to_target_bonus(dist_to_target_pt, dist_to_vertical_pt, v_tip_and_cart)
        

        bonus =  (1 + np.cos(observation[1]))/2 *(tip_accel_bonus_to_vertical +  tip_speed_bonus_to_vertical) / 2 + (moment_compensation_bonus +  dist_to_target_bonus)/2
            
             
              
            #0.1 * np.exp(-np.sqrt(dist_to_target_pt))             
        

        if np.isnan(bonus) or np.isinf(bonus):
            bonus = 0.0

        total_bonus = bonus
        total_penalty = cart_hit_penalty + self.calc_idle_penalty(observation) + self.calc_high_speed_penalty(v_tip_magnitude)
        reward = total_bonus - total_penalty

        if np.isnan(reward) or np.isinf(reward):
            reward = -1.0

        observation = np.concatenate([
            observation,
            [dist_to_vertical_pt],
            [dist_to_target_pt], 
            [self.pole_end_point[0]],
            [self.pole_end_point[2]],
            [self.ball_pose[0]],
            [self.ball_pose[2]],
            [self.pole_mass / 10.0],
            [gravity_moment / 10.0],            
            [v_tip_magnitude],
            [current_a_tip]
        ]).ravel()
        
        # save info about previous state
        self.v_tip_previous = v_tip_magnitude
        self.previous_dist_to_vertical = dist_to_vertical_pt
        return observation.astype(np.float32), reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass

    def draw_ball(self):
        """Spawn a new ball (target)."""
        target_pos = [np.random.rand() - 0.5, 0, 0.6]
        self.ball_pose = target_pos

        if self.render:
            self.sim.draw_ball(target_pos, radius=0.05)
            

    def calc_pole_end_point(self, observation: np.ndarray) -> list:
        """Calculate the pole tip position in world coordinates."""
        cart_pose = observation[0]
        pole_angle = observation[1]
        return [cart_pose + np.sin(pole_angle) * self.l, 0, np.cos(pole_angle) * self.l]

    def calc_euclidean(self, pt1: list, pt2: list) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(np.array(pt1) - np.array(pt2))

    def calculate_angle(self, a: list, b: list, c: list) -> float:
        """Calculate the angle between three points."""
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine_angle = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0)
        return np.arccos(cosine_angle)

    def get_e_potential(self, observation: np.ndarray) -> float:
        """Calculate the potential energy of the pole."""
        theta = observation[1]
        return self.pole_mass * self.g * (self.l / 2) * (1 - np.cos(theta))

    def get_e_kinetic_cart(self, observation: np.ndarray) -> float:
        """Calculate the kinetic energy of the cart."""
        v = observation[2]
        return 0.5 * self.cart_mass * v**2

    def get_e_kinetic_pole(self, observation: np.ndarray) -> float:
        """Calculate the kinetic energy of the pole."""
        theta = observation[1]
        v = observation[2]
        omega = observation[3]

        v_x = v + (self.l / 2) * omega * np.cos(theta)
        v_z = (self.l / 2) * omega * np.sin(theta)
        v_cm_sq = v_x**2 + v_z**2

        ke_translational = 0.5 * self.pole_mass * v_cm_sq
        I = (1/12) * self.pole_mass * self.l**2
        ke_rotational = 0.5 * I * omega**2

        return ke_translational + ke_rotational

    def get_pole_tip_velocity(self, observation: np.ndarray, include_cart_speed = True) -> tuple:
        """Calculate the velocity of the pole tip."""
        pole_angle = observation[1]
        cart_vel = observation[2]
        pole_omega = observation[3]

        if include_cart_speed:
            vx_tip = cart_vel + self.l * pole_omega * np.cos(pole_angle)
        else:
            vx_tip = self.l * pole_omega * np.cos(pole_angle)    
        vz_tip = self.l * pole_omega * np.sin(pole_angle)
        v_tip_magnitude = np.sqrt(vx_tip**2 + vz_tip**2)

        return v_tip_magnitude, vx_tip, vz_tip

    def compute_tip_speed_bonus_to_vertical(self, dist: float, v_tip: float) -> float:
        """Reward for pole tip velocity approaching ideal value."""
        #ideal_v_tip = self.max_tip_speed * (1 - np.exp(-2 * dist))
        theta = dist / self.l
        ideal_v_tip = (1 - np.cos(theta)) * self.max_tip_speed
        #print("ideal_v_tip",ideal_v_tip)
        error = v_tip - ideal_v_tip
        bonus = np.exp(-np.sqrt(abs(error)))
        #print("speed",bonus)

        #if dist < self.target_zone_radius and v_tip < 0.1 * self.max_tip_speed:            
        #    low_speed_at_top_bonus = self.hold_bonus * (1 - dist / self.target_zone_radius) / (1 + (v_tip / self.v_tip_target_in_zone)**2)
        #    bonus += low_speed_at_top_bonus 
            
        return bonus

    def calc_idle_penalty(self, current_observation: np.ndarray) -> float:
        """Penalty for inactivity (little change in observation)."""
        if self.previous_observation is None:
            self.previous_observation = current_observation
            return 0

        delta = np.abs(current_observation - self.previous_observation).sum()
        penalty = self.small_change_penalty if delta < self.small_change_threshold else 0
        self.previous_observation = current_observation
        return penalty

    def calc_high_speed_penalty(self, speed: float) -> float:
        """Penalty for pole tip exceeding maximum allowed speed."""
        return 0 # 1.0 if speed > self.max_tip_speed * 1.2 else 0.0

    def compute_current_tip_acceleration(self, v_tip_now: float) -> float:
        """Calculate current tip acceleration."""
        return (v_tip_now - self.v_tip_previous) / self.dt

    #def compute_ideal_tip_acceleration(self, dist: float, v_tip: float, omega: float) -> float:
    #    """Calculate ideal tip acceleration."""
    #    #ideal_v_tip_derivative = 2 * self.max_tip_speed * np.exp(-2 * dist)
    #    ideal_v_tip_derivative = np.sqrt(3/4 * self.g * self.l)/2 * np.cos((dist/self.l)/2) * omega
    #    
    #    return ideal_v_tip_derivative 

    def compute_tip_acceleration_bonus(self, dist: float, a_tip: float, v_tip_magnitude: float, omega: float) -> float:
        """Reward for approaching ideal acceleration profile."""
        theta = dist / self.l 
        ideal_a_tip = np.sin(theta) * omega * self.max_tip_speed
        #print("ideal_a_tip", ideal_a_tip)
        bonus = np.exp(-np.sqrt(abs(a_tip - ideal_a_tip)))
        #print("accel", bonus)

        #if dist < self.target_zone_radius and v_tip_magnitude < 0.2 * self.max_tip_speed:
        #    low_accel_at_top = self.hold_bonus * (1 - dist / self.target_zone_radius) / (1 + (a_tip / 0.1)**2)
        #    bonus += low_accel_at_top

        return bonus

    def calc_vertical_point(self, observation: np.ndarray):
        """Update vertical point position."""
        self.vertical_point = [observation[0], 0, self.l]
        
    def arc_distance_to_vertical(self) -> float:
        """
        Calculates the signed arc distance between the pole tip and the vertical upright point.


        Returns:
        Arc distance (meters), positive if pole is to the right, negative if to the left.
        """
        x_tip = self.pole_end_point[0]
        z_tip = self.pole_end_point[2]

        theta = np.arccos(np.clip(z_tip / self.l, -1.0, 1.0))  # angle from vertical
        arc_distance = self.l * theta
        return arc_distance
        
        
    def calc_gravity_moment(self, observation):
        theta = observation[1]
        return - self.pole_mass * self.g * self.l/2 * np.sin(theta)
        
    def calc_gravity_moment_bonus(self, observation, dist_to_vertical_pt, dist_to_target_pt):
        g_moment = self.calc_gravity_moment(observation)
        if dist_to_vertical_pt < 0.10 * np.pi * self.l:
            #print("g_moment", g_moment)
            return (0.5 * self.hold_bonus) / (1 + abs(g_moment) * 100 * dist_to_target_pt)   
        return 0
        
    def calc_dist_to_target_bonus(self, dist_to_target, dist_to_vertical_pt, v_tip_and_cart):        
        if dist_to_vertical_pt < 0.10 * np.pi * self.l:
            #return self.hold_bonus * ((np.exp(-dist_to_target * 2) + (np.exp(-v_tip_and_cart * 2)))/2) 
            return self.hold_bonus / (1 + v_tip_and_cart * 100 * dist_to_target)
        return 0       
        

    @property
    def current_time(self) -> float:
        """Get current simulation time."""
        return self.sim.current_time

