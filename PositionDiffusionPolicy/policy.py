from benchnpin.baselines.base_class import BasePolicy
from benchnpin.common.metrics.task_driven_metric import TaskDrivenMetric
from diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
import collections
import torch
import numpy as np
import dill

class PositionDiffusionPolicy(BasePolicy):
    def __init__(self, cfg, env=None):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else self.device)
        self.env = env
        self.policy = self.create_policy()
        self.obs_buffer = collections.deque(maxlen=self.cfg.diffusion.n_obs_steps)

        self.load_checkpoint(self.cfg.diffusion.checkpoint_path)
    
    def load_checkpoint(self, path):
        print(f"Loading diffusion checkpoint from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(checkpoint['state_dicts']['model'])
        self.policy.eval()  # Set to evaluation mode
    
    def create_policy(self):
        # Model configuration
        model_config = {
            'input_dim': self.cfg.diffusion.action_dim,  # Only actions in trajectory for global conditioning
            'local_cond_dim': None,
            'global_cond_dim': self.cfg.diffusion.n_obs_steps * self.cfg.diffusion.obs_dim,  # n_obs_steps * obs_dim for global conditioning
            'diffusion_step_embed_dim': 256,
            'down_dims': [256, 512, 1024],
            'kernel_size': 5,
            'n_groups': 8,
            'cond_predict_scale': True
        }
        
        # Noise scheduler configuration
        scheduler_config = {
            'num_train_timesteps': 100,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'beta_schedule': 'squaredcos_cap_v2',
            'prediction_type': 'epsilon',
            'clip_sample': True,
            'variance_type': 'fixed_small',
        }
        
        # Policy configuration
        policy = DiffusionUnetLowdimPolicy(
            model=model_config,
            noise_scheduler=scheduler_config,
            horizon=self.cfg.diffusion.horizon,
            obs_dim=self.cfg.diffusion.obs_dim,
            action_dim=self.cfg.diffusion.action_dim,
            n_action_steps=self.cfg.diffusion.n_action_steps,
            n_obs_steps=self.cfg.diffusion.n_obs_steps,
            num_inference_steps=100,
            obs_as_global_cond=True,  # Use global conditioning for box delivery
            pred_action_steps_only=True,  # Predict only action steps
            condition_trajectory=True, # TODO: make this configurable
            env=self.env,
        ).to(self.device)

        return policy
    
    def reset(self):
        """Reset the observation buffer"""
        self.obs_buffer.clear()
        
    def act(self, observation, **kwargs):
        # Ensure observation has correct shape [obs_dim]
        if observation.ndim > 1:
            observation = observation.flatten()
        
        # Add observation to buffer
        self.obs_buffer.append(observation)
        
        n_obs_steps = self.cfg.diffusion.n_obs_steps
        
        # Handle insufficient history by padding with first observation
        if len(self.obs_buffer) < n_obs_steps:
            obs_list = [list(self.obs_buffer)[0]] * (n_obs_steps - len(self.obs_buffer)) + list(self.obs_buffer)
        else:
            obs_list = list(self.obs_buffer)
        
        # Shape: [1, n_obs_steps, obs_dim]
        obs_tensor = torch.from_numpy(np.array(obs_list)[np.newaxis, ...]).float().to(self.device)
        obs_dict = {'obs': obs_tensor}
        
        # Get action from policy
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        
        action_sequence = action_dict['action'].cpu().numpy()
        
        # return action
        return action_sequence
    
    def evaluate(self):
        raise NotImplementedError("Evaluation not implemented for PositionDiffusionPolicy")
