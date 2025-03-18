# Matthew Mackey
# 2174428 
# Group 16
# CSC-580-810 

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, PPO, A2C
import torch
import highway_env


TRAIN = True

# Train and Test DQN
def train_dqn():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Experiment Params
    params = {"net_archs":([256, 256], [256, 128, 128]), 
          "gammas":(.90, .99),
          "learning_rates":(1e-4, 5e-4),
          "exploration":([.8, 1.0, 0.05], [.95, 1.0, 0.05])}

    # Build and Train Models
    for i in range(2):
            
        # Filename for saving
        run_name = f"dqn_{i+1}"
        
        # Create the environment
        env = gym.make("highway-fast-v0", render_mode="rgb_array")
        obs, info = env.reset()
        
        # Create the model
        model = DQN(
            'MlpPolicy',
            env,
            policy_kwargs=dict(net_arch=params["net_archs"][i]),
            learning_rate=params['learning_rates'][i],
            learning_starts=200,
            batch_size=64,
            gamma=params['gammas'][i],
            exploration_fraction=params['exploration'][i][0],
            exploration_initial_eps= params['exploration'][i][1],
            exploration_final_eps= params['exploration'][i][2],
            train_freq=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log=f"highway_dqn/",
            device=device
        )
        # Train the model
        if TRAIN:
            model.learn(total_timesteps=int(2e4))
            model.save(f"highway_dqn/models/{run_name}")
            del model
        
        # Run the trained model and record video
        model = DQN.load(f"highway_dqn/models/{run_name}", env=env)
        env = RecordVideo(
            env, video_folder=f"highway_dqn/videos/{run_name}", episode_trigger=lambda e: True
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

    # Experimented Params
    params = {"net_archs":([256, 256], [256, 128, 128]), 
              "gammas":(.90, .99),
              "learning_rates":(1e-4, 5e-4)}

    # Build and Train models
    for i in range(2):  
        # Filename for saving
        run_name = f"ppo_{i+1}"
        
        # Create the environment
        env = gym.make("highway-fast-v0", render_mode="rgb_array")
        obs, info = env.reset()
        
        # Create the model
        model = PPO(
            'MlpPolicy',
            env,
            policy_kwargs=dict(net_arch=params["net_archs"][i]),
            learning_rate=params['learning_rates'][i],
            gamma=params['gammas'][i],
            batch_size=64,
            n_epochs=10,
            verbose=1,
            tensorboard_log=f"highway_ppo/",
            device=device
        )

        # Train the model
        if TRAIN:
            model.learn(total_timesteps=int(5e4))
            model.save(f"highway_ppo/models/{run_name}")
            del model
        
        # Run the trained model and record video
        model = PPO.load(f"highway_ppo/models/{run_name}", env=env)
        env = RecordVideo(
            env, video_folder=f"highway_ppo/videos/{run_name}", episode_trigger=lambda e: True
        )
        env.unwrapped.config["simulation_frequency"] = 15  # Higher FPS for rendering
        env.unwrapped.set_record_video_wrapper(env)
         
        for videos in range(10):
            done = truncated = False
            obs, info = env.reset()
            while not (done or truncated):
                # Predict
                action, _states = model.predict(obs, deterministic=True)
                # Get reward
                obs, reward, done, truncated, info = env.step(action)
                # Render
                env.render()
        env.close()

# Train and Test A2C
def train_a2c():
    # Warnings suggest using cpu for A2C without CNN
    device = 'cpu'

    # Experimented Params
    params = {"net_archs":([256, 256], [256, 128, 128]), 
              "gammas":(.90, .99),
              "learning_rates":(1e-4, 5e-4),
              "n_steps":(5, 7),
              "ent_coefs":(0.001, 0.01)}    

    # Build and Train mdoels
    for i in range(2):
        # Filename for saving
        run_name = f"a2c_{i+1}"
        
        # Create the environment
        env = gym.make("highway-fast-v0", render_mode="rgb_array")
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env, gamma=params["gammas"][i])
        obs, info = env.reset()
        
        # Create the model
        model = A2C(
            'MlpPolicy',
            env,
            policy_kwargs=dict(net_arch=params['net_archs'][i]),
            learning_rate=params['learning_rates'][i],
            gamma=params['gammas'][i],
            n_steps=params['n_steps'][i],
            ent_coef=params['ent_coefs'][i],
            verbose=1,
            tensorboard_log=f"highway_a2c/",
            device=device
        )
        
        # Train the model
        if TRAIN:
            model.learn(total_timesteps=int(1e5))
            model.save(f"highway_a2c/models/{run_name}")
            del model
        
        # Run the trained model and record video
        model = A2C.load(f"highway_a2c/models/{run_name}", env=env)
        env = RecordVideo(
            env, video_folder=f"highway_a2c/videos/{run_name}", episode_trigger=lambda e: True
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
    train_dqn()

    train_ppo()

    train_a2c()
        