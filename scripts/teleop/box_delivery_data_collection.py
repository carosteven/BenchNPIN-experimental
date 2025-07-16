"""
A simple script to run a teleoperation pipeline for demonstration dataset collection on box delivery environments
'A': large left turn; 'D' large right turn
'Z': small left turn; 'C' small right turn
'W': start moving
'X': stop turning (note: this does not stop linear motion)
'esc': exit teleoperation
"""
# TODO: record high and low dimenstion states
import random

import benchnpin.environments
import gymnasium as gym
import numpy as np
import pickle
# import zarr
from pynput import keyboard
from os.path import dirname
from benchnpin.baselines.box_delivery.SAM.policy import BoxDeliverySAM
from diffusion_policy.common.replay_buffer import ReplayBuffer
import pygame


# observations_low = []
# observations_high = []
# actions = []                # this is actually the states (i.e. 3 dof pose)
# rewards = []
# terminals = []              # This is true when episodes end due to termination conditions such as falling over.
# timeouts = []               # This is true when episodes end due to reaching the maximum episode length

WAYPOINT_MOVING_THRESHOLD = 0.6

FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3
STOP = 4
STOP_LINEAR = 5
STOP_TURNING = 6
SMALL_LEFT = 7
SMALL_RIGHT = 8
DELETE_PREV_DEMO = 9
DONE_DEMO = 10
BREAK_NONMOVEMENT = 11

command = STOP
# manual_stop = False

def on_press(key):
    global command
    try:
        if key.char == 'w':  # Move up
            command = FORWARD
        elif key.char == 's':  # Move down
            command = BACKWARD
        elif key.char == 'a':  # Move left
            command = LEFT
        elif key.char == 'd':  # Move right
            command = RIGHT
        elif key.char == 'e':  # Stop moving
            command = STOP
        elif key.char == 't': # Stop linear
            command = STOP_LINEAR
        elif key.char == 'r':  # Stop turning
            command = STOP_TURNING
        elif key.char == 'z':  # Move left slowly
            command = SMALL_LEFT
        elif key.char == 'c':  # Move right slowly
            command = SMALL_RIGHT
        elif key.char == 'q': # End current demonstration
            command = DONE_DEMO

    except AttributeError:
        if key == keyboard.Key.backspace: # Delete previous demonstration
            command = DELETE_PREV_DEMO
        elif key == keyboard.Key.space:
            command = BREAK_NONMOVEMENT # Action behind robot to break stuck cycle


# def on_release(key):
#     global action, manual_stop
#     if key == keyboard.Key.esc:  # Stop teleoperation when ESC is pressed
#         manual_stop = True
#         return False


# def record_transition(observation_low, observation_high, state, reward, terminal, timeout):
#     observations_low.append(observation_low)
#     observations_high.append(observation_high)
#     actions.append(state)
#     rewards.append(reward)
#     terminals.append(terminal)
#     timeouts.append(timeout)


'''
Plan:
    - record high/low dim observations
        - high dim: channel 0 (224x224)
        - low dim: [agent_pos, [boxes_pos]]

    - record goal as xy
        - append goal to state

    * COORDINATES RELATIVE TO ROBOT FRAME *


    Initialize policy
    one demo = one step
    need to use 'demo_mode' but also use the threshold logic to terminate step when reach goal
    need to visualize goal --> destination from SAM
'''
# TODO: set seed to length of replay buffer
def collect_demos():
    path = 'demo_data/box_delivery_demo.zarr'
    replay_buffer = ReplayBuffer.create_from_path(path, mode='a')

    # ensure different environments
    seed = replay_buffer.n_episodes
    print(f'starting seed {seed}')

    cfg = {
        'teleop_mode': True,
        'misc': {
            'inactivity_cutoff_sam': 10000,  # set to a large number to avoid inactivity cutoff
            'inactivity_cutoff': 10000,  # set to a large number to avoid inactivity cutoff
            'random_seed': seed,
        }
    }
    env = gym.make('box-delivery-v0', cfg=cfg)
    env = env.unwrapped
    dummy_observation, _ = env.reset()

    model_name = 'base_se'
    model_path = 'models/box_delivery'
    policy = BoxDeliverySAM(cfg=env.cfg, model_name=model_name, model_path=model_path)
    # Initialize the policy
    policy.act(dummy_observation, env.action_space.high, env.num_channels)

    path_length = 0
    # step_size = 0.1
    step_size = WAYPOINT_MOVING_THRESHOLD

    observation, info = env.reset()
    # record_transition(observation, observation, [info['state'][0], info['state'][1]], 0, False, False)
    # prev_state = [info['state'][0], info['state'][1]]

    terminated = False
    truncated = False

    manual_stop = False
    break_nonmovement_action = False

    episodes = []
    clock = pygame.time.Clock()
    # with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    with keyboard.Listener(on_press=on_press) as listener:
        try:
            # episode-level while loop (an episode is travelling to the next point)
            while listener.running:  # While the listener is active
                episode = list()

                prev_state = [info['state'][0], info['state'][1]]
                
                # current robot pose
                robot_current_position, robot_current_heading = env.robot.body.position, env.restrict_heading_range(env.robot.body.angle)
                robot_current_position = list(robot_current_position)  

                goal_ravelled = policy.act(observation)
                if break_nonmovement_action:
                    # sometimes the goal can be really close to the robot such that it hits without moving
                    # if so then can break the cycle with an action far behind the robot
                    goal_ravelled = 95*94 + 45
                    break_nonmovement_action = False

                goal = env.position_controller.get_target_position(robot_current_position, robot_current_heading, goal_ravelled) 

                # display goal in environment
                env.renderer.goal_point = goal

                reached_goal = False
                ignore_curr_demo = False
                
                t = 0
                transition_count = 1        # start from 1 as we recorded the reset step   
                global command
                command = STOP

                # step-level while loop
                while not terminated or not truncated or not reached_goal:
                    # global command
                    if command == DELETE_PREV_DEMO:
                        print("\nCurrent demonstration ignored")
                        ignore_curr_demo = True
                        command = STOP
                    
                    elif command == DONE_DEMO:
                        # break
                        reached_goal = True

                    elif command == BREAK_NONMOVEMENT:
                        break_nonmovement_action = True
                        command = STOP

                    print("command: ", command, "; step: ", t, \
                        "; num completed: ", info['cumulative_boxes'],  end="\r")

                    if env.distance((info['state'][0], info['state'][1]), goal) < WAYPOINT_MOVING_THRESHOLD: # or env.robot_hit_obstacle:
                        reached_goal = True

                    # command = OTHER
                    if t % 5 == 0:
                        env.render()

                    # only record points based on distance interval
                    if (((info['state'][0] - prev_state[0])**2 + (info['state'][1] - prev_state[1])**2)**(0.5) >= step_size) or terminated or truncated or reached_goal:
                        # record_transition(observation, observation, [info['state'][0], info['state'][1]], reward, terminated, truncated)
                        goal = np.array(goal)
                        action = np.array(info['state'][:2])
                        data = {
                            'img': observation[0],
                            'state_vertices': np.float32(info['obs_vertices']),
                            'state_positions': np.float32(info['obs_positions']),
                            'goal': np.float32(goal),
                            'action': np.float32(action)
                        }
                        episode.append(data)

                        prev_state = [info['state'][0], info['state'][1]]
                        transition_count += 1
                    
                    observation, reward, terminated, truncated, info = env.step(command)

                    if terminated or truncated or reached_goal:
                        print("\nterminated: ", terminated, "; truncated: ", truncated, "; reached goal: ", reached_goal)
                        path_length = transition_count
                        print()
                        print(transition_count)
                        if terminated or truncated:
                            observation, info = env.reset()
                        break

                    clock.tick(20)  # Limit the frame rate

                t += 1

                if not ignore_curr_demo:
                    episodes.append(episode)

                if terminated:
                    # save episode buffer to replay buffer (on disk)
                    response = input("\nSave demonstrations? (y/n) ").strip().lower()[-1]
                    if response == 'y':
                        for episode in episodes:
                            if len(episode) > 0:
                                data_dict = dict()
                                for key in episode[0].keys():
                                    data_dict[key] = np.stack(
                                        [x[key] for x in episode])
                                replay_buffer.add_episode(data_dict, compressors='disk')
                        
                        print("Demonstrations saved. Resetting environment...")
                    else:
                        print("Demonstrations ignored. Resetting environment...")

                    episodes = []
             
                # don't save the demo if this trial is truncated
                # if manual_stop:
                #     print("\nDemo manually stopped. Ignored")
                #     return


        except KeyboardInterrupt:
            print("Exiting teleoperation.")

        finally:
            env.close()
    
    # don't save the demo if this trial is truncated
    if truncated:
        print("\n Demo truncated. Ignored")
        return


    # store = zarr.DirectoryStore("data.zarr")
    # root = zarr.group(store=store)

    ''' 
    global observations, actions, rewards, terminals, timeouts
    observations = np.array(observations).astype(np.float32)
    actions = np.array(actions).astype(np.float32)
    rewards = np.array(rewards).astype(np.float32)
    terminals = np.array(terminals)
    timeouts = np.array(timeouts)
    path_lengths = np.array([path_length])

    print("observation shape: ", observations.shape)
    print("actions shape: ", actions.shape)
    print("rewards shape: ", rewards.shape)
    print("terminals shape: ", terminals.shape)
    print("timeouts shape: ", timeouts.shape)
    print("current path length: ", path_length)


    try:
        # load previous demos
        with open('delivery_demo.pkl', 'rb') as file:
            pickle_dict = pickle.load(file)

        with open('delivery_demo_info.pkl', 'rb') as f:
            pickle_dict_info = pickle.load(f)
        
        # append current demonstration data
        pickle_dict['observations'] = np.concatenate((pickle_dict['observations'], observations))
        pickle_dict['actions'] = np.concatenate((pickle_dict['actions'], actions))
        pickle_dict['rewards'] = np.concatenate((pickle_dict['rewards'], rewards))
        pickle_dict['terminals'] = np.concatenate((pickle_dict['terminals'], terminals))
        pickle_dict['timeouts'] = np.concatenate((pickle_dict['timeouts'], timeouts))

        # append current meta-info data
        pickle_dict_info['path_lengths'] = np.concatenate((pickle_dict_info['path_lengths'], path_lengths))
        pickle_dict_info['demo_count'] = pickle_dict_info['demo_count'] + 1

    except:
        # if delivery_demo file not exist, create one with current demos
        pickle_dict = {
            'observations': observations, 
            'actions': actions, 
            'rewards': rewards, 
            'terminals': terminals,
            'timeouts': timeouts
        }

        pickle_dict_info = {
            'path_lengths': path_lengths,
            'demo_count': 1
        }

    print("Total Demonstration Data ======== \n")
    print("observation shape: ", pickle_dict['observations'].shape)
    print("actions shape: ", pickle_dict['actions'].shape)
    print("rewards shape: ", pickle_dict['rewards'].shape)
    print("terminals shape: ", pickle_dict['terminals'].shape)
    print("timeouts shape: ", pickle_dict['timeouts'].shape)

    print("max path lengths: ", np.max(pickle_dict_info['path_lengths']), "; min path length: ", np.min(pickle_dict_info['path_lengths']), "; average path length: ", np.mean(pickle_dict_info['path_lengths']))
    print("Total number of demos: ", pickle_dict_info['demo_count'])

    
    # save demo data
    with open('delivery_demo.pkl', 'wb') as f:
        pickle.dump(pickle_dict, f)

    # save demo info data
    with open('delivery_demo_info.pkl', 'wb') as f:
        pickle.dump(pickle_dict_info, f)
    '''


if __name__ == "__main__":
    collect_demos()
    # env.close()
