# Import ViZDoom for game env
import vizdoom as vzd
# Import random for action sampling
import random
# Import time for sleeping
import time
# Import numpy for identity matrix
import numpy as np
# Import environment base class from OpenAI Gym/Gymnasium
from gymnasium import Env
# Import gym spaces
from gymnasium.spaces import Discrete, Box
# Import opencv and matplotlib
import cv2 
import matplotlib.pyplot as plt
# Import os for file nav
import os
# Import callback class from sb3
from stable_baselines3.common.callbacks import BaseCallback
# Import Environment checker
from stable_baselines3.common import env_checker
# Import PPO for training
from stable_baselines3 import PPO
# Import eval policy to test agent
from stable_baselines3.common.evaluation import evaluate_policy

CHECKPOINT_DIR = "./train/train_defend"
LOG_DIR = "./logs/log_defend"

# # Setup game
# game = vzd.DoomGame()
# game.load_config("github/ViZDoom/scenarios/defend_the_center.cfg")
# game.init()

# This is the set of actions we can take in the environment
actions = np.identity(3, dtype=np.uint8)
print(random.choice(actions))

# # Loop through episodes
# episodes = 10
# for episode in range(episodes):
#     # Create a new episode or game
#     game.new_episode()
#     # Check the game isn't done
#     while not game.is_episode_finished():
#         # Get the game state
#         state = game.get_state()
#         # Get the game image
#         img = state.screen_buffer
#         # Get the game variables - ammo
#         info = state.game_variables
#         # Take an action
#         reward = game.make_action(random.choice(actions), 4)  # 4 is the FRAME SKIP, helps the agent 
#         # Print reward
#         print("reward:", reward)
#         time.sleep(0.02)
#     print("Result:", game.get_total_reward())
#     time.sleep(2)  
# game.close()

# Create Vizdoom OpenAi Gym Environment -> NOT NEEDED FOR NEWER GYMNASIUM RELEASES?
class VizDoomGym(Env):
    # Function that is called when we start the env
    def __init__(self, render=False):
        # Inherit from Env
        super().__init__()
        # Setup the game
        self.game = vzd.DoomGame()
        self.game.load_config("github/ViZDoom/scenarios/defend_the_center.cfg")
        
        
        # Render frame logic
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
            
        # Start the game    
        self.game.init()
        
        # Create the action space and the observation space
        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(3)
        
                
    # This is how we take a step in the environment
    def step(self, action):
        # Specify action and take step
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4)
        # Get all the other stuff we need to return
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0
                         
        info = {"info":info}                
        done = self.game.is_episode_finished()
        terminated = done
        truncated = done
        
        return state, reward, terminated, truncated, info
    
    # Define how to render the game or environment
    def render():
        pass
    
    # What happens when we start a new game 
    def reset(self, seed=None):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state), seed
    
   
    # Grayscale the game frame and resize it 
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY) # Needs moveaxis to flip the frame shape to grayscale it
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state
    # Call to close down the game
    def close(self):
        self.game.close()
   
# # View state    
# env = VizDoomGym(render=True)
# state = env.reset()
# plt.imshow(cv2.cvtColor(state, cv2.COLOR_BGR2RGB))  # Check if the frame is grayscale
# env_checker.check_env(env) # Check the environment

# Setup callback
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, "best_model_{}".format(self.n_calls))
            self.model.save(model_path)
            
        return True
    
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# # Train model -> uncomment to train
# # Non rendered environment
# env = VizDoomGym()
# model = PPO("CnnPolicy", env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
# model.learn(total_timesteps=100000, callback=callback)

# Test the model
# Reload model from disk
model = PPO.load("./train/train_defend/best_model_100000")
# Create rendered environment
env = VizDoomGym(render=True)
# Evaluate mean reward for 50 games
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=50)
print(mean_reward)

# # Manual evaluation, doesn't work, CHECK obs too many dimensions error -> Needs to be 1D ndarray?
# for episode in range(5):
#     obs = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         action = model.predict(obs[0])
#         obs, reward, terminated, truncated, info = env.step(action)
#         time.sleep(0.20)
#         total_reward += reward
#     print("Total reward for episode {} is {}".format(episode, total_reward))
#     time.sleep(2)            