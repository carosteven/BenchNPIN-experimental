"""
An example script to show configurable parameters for Box-Delivery
"""
import benchnpin.environments
import gymnasium as gym


############### VVVVVVV Configurable Parameters for Box-Delivery VVVVVVV ####################
cfg = {
    "output_dir": "logs/",      # Specify directory for loggings
}
############### ^^^^^^^ Configurable Parameters for Box-Delivery ^^^^^^^ ####################


env = gym.make('box-delivery-v0', cfg=cfg)
env.reset()

terminated = truncated = False
while not (terminated or truncated):
    observation, reward, terminated, truncated, info = env.step(0)
    env.render()
