import numpy as np
import matplotlib.pyplot as plt
import torch
from diffusion_policy.box_delivery_dataset import BoxDeliveryLowdimDataset
from PositionDiffusionPolicy.policy import PositionDiffusionPolicy
from benchnpin.common.utils.utils import DotDict

def visualize_diffusion_prediction():
    # Load dataset to get a sample observation
    dataset = BoxDeliveryLowdimDataset(
        zarr_path='demo_data/box_delivery_expert_demo_padded.zarr',
        horizon=8,
        pad_before=2-1,
        pad_after=4-1,
        obs_key='state_positions',
        state_key='goal',
        action_key='action'
    )
    
    # Access the replay buffer to get full episodes
    replay_buffer = dataset.replay_buffer
    episode_ends = replay_buffer.episode_ends
    
    # Get a sample observation
    sample = dataset[0]
    obs = sample['obs']
    actual_actions = sample['action'][:8]
    
    print(f"Observation shape: {obs.shape}")
    print(f"Actual actions shape: {actual_actions.shape}")
    
    # Load your diffusion policy
    cfg = {
        'diffusion': {
            'checkpoint_path': 'data/outputs/expert_new.ckpt',
            'obs_dim': 10,
            'action_dim': 2,
            'n_obs_steps': 2,
            'horizon': 8,
            'n_action_steps': 4
        }
    }
    cfg = DotDict.to_dot_dict(cfg)
    
    policy = PositionDiffusionPolicy(cfg)
    idx = 0
    
    while idx < len(dataset):
        sample = dataset[idx]
        obs = sample['obs']
        actual_actions = sample['action'][:8]

        # Get the episode index for this sample
        sample_idx = dataset.sampler.indices[idx][0]  # Get the actual buffer index
        
        # Find which episode this sample belongs to
        episode_idx = 0
        for i, end in enumerate(episode_ends):
            if sample_idx < end:
                episode_idx = i
                break
        
        # Get full episode data
        episode_start = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
        episode_end = episode_ends[episode_idx]
        
        # Extract full episode trajectory
        full_episode_obs = replay_buffer['state_positions'][episode_start:episode_end]
        full_episode_obs = np.concatenate((full_episode_obs, replay_buffer['goal'][episode_start:episode_end]), axis=1)
        full_episode_actions = replay_buffer['action'][episode_start:episode_end]
        
        print(f"\nEpisode {episode_idx}: steps {episode_start} to {episode_end-1} (length: {episode_end - episode_start})")
        print(f"Sample position in episode: {sample_idx - episode_start}")

        # Predict actions
        obs_dict = {'obs': obs[:2].unsqueeze(0)}
        with torch.no_grad():
            result = policy.policy.predict_action(obs_dict)
            predicted_actions = result['action'].cpu().squeeze(0).numpy()

        print(f"Predicted actions shape: {predicted_actions.shape}")

        # Parse current observation
        current_obs = obs[1].numpy()
        robot_pos = current_obs[:2]
        box1_pos = current_obs[2:4]
        box2_pos = current_obs[4:6]
        receptacle_pos = current_obs[-4:-2]
        goal_pos = current_obs[-2:]

        # Extract robot positions from full episode
        # full_episode_robot_positions = full_episode_obs[:, :2]
        full_episode_robot_positions = full_episode_actions
        # distance from last action to goal
        distance_to_goal = np.linalg.norm(goal_pos - full_episode_actions[-1])
        print(f"Distance to goal: {distance_to_goal:.2f}")

        X_MIN, X_MAX = -5, 5
        Y_MIN, Y_MAX = -2.5, 2.5

        # Visualize
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        
        # Plot 1: Current state
        ax1.set_title('Current Observation')
        ax1.scatter(*robot_pos, c='blue', s=100, label='Robot', marker='o')
        ax1.scatter(*box1_pos, c='red', s=80, label='Box 1', marker='s')
        ax1.scatter(*box2_pos, c='orange', s=80, label='Box 2', marker='s')
        ax1.scatter(*receptacle_pos, c='green', s=120, label='Receptacle', marker='D')
        ax1.scatter(*goal_pos, c='purple', s=100, label='Goal', marker='*')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_xlim(X_MIN, X_MAX)
        ax1.set_ylim(Y_MIN, Y_MAX)
        # ax1.axis('equal')
        
        # Plot 2: Predicted vs Actual trajectories (your existing code)
        ax2.set_title('Predicted vs Actual Action Sequences')
        ax2.scatter(*robot_pos, c='blue', s=100, label='Robot Start', marker='o')
        ax2.scatter(*goal_pos, c='purple', s=100, label='Goal', marker='*')
        ax2.scatter(*box1_pos, c='red', s=60, label='Box 1', marker='s', alpha=0.7)
        ax2.scatter(*box2_pos, c='orange', s=60, label='Box 2', marker='s', alpha=0.7)
        ax2.scatter(*receptacle_pos, c='green', s=80, label='Receptacle', marker='D', alpha=0.7)
        
        # Plot predicted trajectory
        predicted_trajectory = [robot_pos]
        for i, action in enumerate(predicted_actions):
            next_pos = action
            predicted_trajectory.append(next_pos)
            ax2.scatter(*next_pos, c='red', s=40, alpha=0.8, marker='o')
            ax2.annotate(f'P{i+1}', next_pos, xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, color='red')

        # Plot actual trajectory
        actual_trajectory = [robot_pos]
        actual_actions_np = actual_actions.numpy()
        for i, action in enumerate(actual_actions_np):
            next_pos = action 
            actual_trajectory.append(next_pos)
            ax2.scatter(*next_pos, c='blue', s=40, alpha=0.8, marker='s')
            ax2.annotate(f'A{i+1}', next_pos, xytext=(-5, -15), textcoords='offset points', 
                        fontsize=8, color='blue')
        
        # Draw trajectory lines
        predicted_trajectory = np.array(predicted_trajectory)
        actual_trajectory = np.array(actual_trajectory)
        
        ax2.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'r--', 
                alpha=0.6, linewidth=2, label='Predicted Path')
        ax2.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], 'b-', 
                alpha=0.6, linewidth=2, label='Actual Path')
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_xlim(X_MIN, X_MAX)
        ax2.set_ylim(Y_MIN, Y_MAX)
        # ax2.axis('equal')

        # Plot 3: Full Episode Trajectory
        ax3.set_title(f'Full Episode {episode_idx} Trajectory')
        
        # Plot environment elements
        episode_start_obs = full_episode_obs[0]
        episode_box1_pos = episode_start_obs[2:4]
        episode_box2_pos = episode_start_obs[4:6]
        episode_receptacle_pos = episode_start_obs[-4:-2]
        episode_goal_pos = episode_start_obs[-2:]
        
        ax3.scatter(*episode_box1_pos, c='red', s=60, label='Box 1', marker='s', alpha=0.7)
        ax3.scatter(*episode_box2_pos, c='orange', s=60, label='Box 2', marker='s', alpha=0.7)
        ax3.scatter(*episode_receptacle_pos, c='green', s=80, label='Receptacle', marker='D', alpha=0.7)
        ax3.scatter(*episode_goal_pos, c='purple', s=100, label='Goal', marker='*')
        
        # Plot full episode robot trajectory
        ax3.plot(full_episode_robot_positions[:, 0], full_episode_robot_positions[:, 1], 
                'g-', alpha=0.8, linewidth=2, label='Full Episode Path')
        
        # Mark episode start and end
        ax3.scatter(*full_episode_robot_positions[0], c='green', s=150, 
                   label='Episode Start', marker='o', edgecolor='black', linewidth=2)
        ax3.scatter(*full_episode_robot_positions[-1], c='red', s=150, 
                   label='Episode End', marker='X', edgecolor='black', linewidth=2)
        
        # Mark current position in episode
        current_step_in_episode = sample_idx - episode_start
        if current_step_in_episode < len(full_episode_robot_positions):
            current_pos_in_episode = full_episode_robot_positions[current_step_in_episode]
            ax3.scatter(*current_pos_in_episode, c='blue', s=120, 
                       label='Current Sample', marker='o', edgecolor='yellow', linewidth=3)
        
        # Add step numbers along the trajectory
        for i in range(0, len(full_episode_robot_positions), max(1, len(full_episode_robot_positions)//10)):
            pos = full_episode_robot_positions[i]
            ax3.annotate(f'{i}', pos, xytext=(0, 10), textcoords='offset points', 
                        fontsize=6, color='green', ha='center')
        
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_xlim(X_MIN, X_MAX)
        ax3.set_ylim(Y_MIN, Y_MAX)
        # ax3.axis('equal')
        
        plt.tight_layout()
        plt.show()

        # Print episode statistics
        print(f"\nEpisode Statistics:")
        print(f"Episode length: {len(full_episode_robot_positions)} steps")
        print(f"Episode start position: {full_episode_robot_positions[0]}")
        print(f"Episode end position: {full_episode_robot_positions[-1]}")
        print(f"Current sample position: {robot_pos}")
        
        # Calculate comparison metrics (your existing code)
        if len(predicted_actions) == len(actual_actions_np):
            mse = np.mean((predicted_actions - actual_actions_np) ** 2)
            mae = np.mean(np.abs(predicted_actions - actual_actions_np))
            print(f"\nComparison metrics:")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
        
        pred_endpoint = predicted_trajectory[-1]
        actual_endpoint = actual_trajectory[-1]
        endpoint_distance = np.linalg.norm(pred_endpoint - actual_endpoint)
        print(f"Endpoint distance: {endpoint_distance:.4f}")

        input("\nPress Enter to continue to the next sample...")
        idx += 1

if __name__ == "__main__":
    visualize_diffusion_prediction()