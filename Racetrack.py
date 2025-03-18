# Matthew Mackey
# 2174428 
# Group 16
# CSC-580-810 

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO
import torch
import highway_env


TRAIN = True

# Adusted env config so the duration is 60 instead of 300
CONFIG = {
    "observation": {
        "type": "OccupancyGrid",
        "features": ['presence', 'on_road'],
        "grid_size": [[-18, 18], [-18, 18]],
        "grid_step": [3, 3],
        "as_image": False,
        "align_to_vehicle_axes": True
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": False,
        "lateral": True
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 60,
    "collision_reward": -1,
    "lane_centering_cost": 4,
    "action_reward": -0.3,
    "controlled_vehicles": 1,
    "other_vehicles": 1,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}

# Train and Test TRPO
def train_trpo():
    # Set device
    device = 'cpu'

    # Experiment Params
    params = {"net_archs":([256, 256, 256]), 
          "gammas":(.95),
          "learning_rates":(1e-4),
        }

    # Build and Train Models
    # Filename for saving
    run_name = f"trpo_{1}"
    
    # Create the environment
    env = gym.make("racetrack-v0", render_mode="rgb_array", config=CONFIG)
    obs, info = env.reset()
    
    # Create the model
    model = TRPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(net_arch=params["net_archs"]),
        learning_rate=params['learning_rates'],
        gamma=params['gammas'],
        verbose=1,
        tensorboard_log=f"racetrack_trpo/",
        device=device
    )
    
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(5e4))
        model.save(f"racetrack_trpo/models/{run_name}")
        del model
    
    # Run the trained model and record video
    model = TRPO.load(f"racetrack_trpo/models/{run_name}", env=env)
    env = RecordVideo(
        env, video_folder=f"racetrack_trpo/videos/{run_name}", episode_trigger=lambda e: True
    )

    env.unwrapped.config["simulation_frequency"] = 15  
    env.unwrapped.set_record_video_wrapper(env)
    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()

# Train and Test PPO
def train_ppo():
    # Device to CPU for PPO without CNN
    device = 'cpu'

    # Experiment Params
    params = {"net_archs":([256, 256, 256]), 
          "gammas":(.95),
          "learning_rates":(1e-4),
        }

    # Build and Train models
    # Filename for saving
    run_name = f"ppo_{1}"
    
    # Create the environment
    env = gym.make("racetrack-v0", render_mode="rgb_array", config=CONFIG)
    obs, info = env.reset()
    
    # Create the model
    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(net_arch=params["net_archs"]),
        learning_rate=params['learning_rates'],
        gamma=params['gammas'],
        batch_size=64,
        n_epochs=10,
        verbose=1,
        tensorboard_log=f"racetrack_ppo/",
        device=device
    )
    
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(5e4))
        model.save(f"racetrack_ppo/models/{run_name}")
        del model
    
    # Run the trained model and record video
    model = PPO.load(f"racetrack_ppo/models/{run_name}", env=env)
    env = RecordVideo(
        env, video_folder=f"racetrack_ppo/videos/{run_name}", episode_trigger=lambda e: True
    )

    env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering
    env.unwrapped.set_record_video_wrapper(env)
     
    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()

# Train and Test A2C
def train_a2c():
    # Warnings suggest using cpu for A2C without CNN
    device = 'cpu'

    # Experimented Params
    params = {"net_archs":([256, 256, 256]), 
              "gammas":(.95),
              "learning_rates":(1e-4),
              "n_steps":(5),
              "ent_coefs":(0.001)}    

    # Build and Train mdoels
    # Filename for saving
    run_name = f"a2c_{1}"
    
    # Create the environment
    env = gym.make("racetrack-v0", render_mode="rgb_array", config=CONFIG)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=params["gammas"])
    obs, info = env.reset()
    
    # Create the model
    model = A2C(
        'MlpPolicy',
        env,
        policy_kwargs=dict(net_arch=params['net_archs']),
        learning_rate=params['learning_rates'],
        gamma=params['gammas'],
        n_steps=params['n_steps'],
        ent_coef=params['ent_coefs'],
        verbose=1,
        tensorboard_log=f"racetrack_a2c/",
        device=device
    )
    
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(5e4))
        model.save(f"racetrack_a2c/models/{run_name}")
        del model
    
    # Run the trained model and record video
    model = A2C.load(f"racetrack_a2c/models/{run_name}", env=env)
    env = RecordVideo(
        env, video_folder=f"racetrack_a2c/videos/{run_name}", episode_trigger=lambda e: True
    )

    env.unwrapped.config["simulation_frequency"] = 15
    env.unwrapped.set_record_video_wrapper(env)
     
    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()


if __name__ == "__main__":

    # Train models for comparison
    train_trpo()

    train_ppo()

    train_a2c()
        