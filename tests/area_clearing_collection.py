"""
A simple script to run a teleoperation pipeline for demonstration dataset collection on box pushing environments
'A': large left turn; 'D' large right turn
'Z': small left turn; 'C' small right turn
'W': start moving
'X': stop turning (note: this does not stop linear motion)
'esc': exit teleoperation
"""

import benchnpin.environments
import gymnasium as gym
import numpy as np
import pickle
from pynput import keyboard

env = gym.make('area-clearing-v0')

observations = []
actions = []                # this is actually the states (i.e. 3 dof pose)
rewards = []
terminals = []              # This is true when episodes end due to termination conditions such as falling over.
timeouts = []               # This is true when episodes end due to reaching the maximum episode length

FORWARD = 0
STOP_TURNING = 1
LEFT = 2
RIGHT = 3
STOP = 4
BACKWARD = 5
SMALL_LEFT = 6
SMALL_RIGHT = 7

command = STOP
manual_stop = False

def on_press(key):
    global command
    try:
        if key.char == 'w':  # Move up
            command = FORWARD
        elif key.char == 'x':  # Move down
            command = BACKWARD
        elif key.char == 'a':  # Move left
            command = LEFT
        elif key.char == 'd':  # Move right
            command = RIGHT
        elif key.char == 't':  # Stop moving
            command = STOP
        elif key.char == 'r':  # Stop turning
            command = STOP_TURNING
        elif key.char == 'z':  # Move left slowly
            command = SMALL_LEFT
        elif key.char == 'c':  # Move right slowly
            command = SMALL_RIGHT
    except AttributeError:
        pass


def on_release(key):
    global action, manual_stop
    if key == keyboard.Key.esc:  # Stop teleoperation when ESC is pressed
        manual_stop = True
        return False


def record_transition(observation, state, reward, terminal, timeout):
    observations.append(observation)
    actions.append(state)
    rewards.append(reward)
    terminals.append(terminal)
    timeouts.append(timeout)


def collect_demos():

    path_length = 0
    step_size = 0.1

    observation, info = env.reset()
    record_transition(observation, [info['state'][0], info['state'][1]], 0, False, False)
    prev_state = [info['state'][0], info['state'][1]]

    terminated = False
    truncated = False
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            t = 0
            transition_count = 1        # start from 1 as we recorded the reset step
            while listener.running:  # While the listener is active
                global command
                print("command: ", command, "; step: ", t, \
                    "; num completed: ", info['box_count'],  end="\r")
                observation, reward, terminated, truncated, info = env.step(command)

                # command = OTHER
                if t % 5 == 0:
                    env.render()

                if (((info['state'][0] - prev_state[0])**2 + (info['state'][1] - prev_state[1])**2)**(0.5) >= step_size) or terminated or truncated:
                    record_transition(observation, [info['state'][0], info['state'][1]], reward, terminated, truncated)
                    prev_state = [info['state'][0], info['state'][1]]
                    transition_count += 1

                if terminated or truncated:
                    print("\nterminated: ", terminated, "; truncated: ", truncated)
                    path_length = transition_count
                    break

                t += 1


        except KeyboardInterrupt:
            print("Exiting teleoperation.")
        finally:
            env.close()
    
    # don't save the demo if this trial is truncated
    if truncated:
        print("\n Demo truncated. Ignored")
        return

    # don't save the demo if this trial is truncated
    global manual_stop
    if manual_stop:
        print("\nDemo manually stopped. Ignored")
        return

if __name__ == "__main__":
    collect_demos()
