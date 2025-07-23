from benchnpin.baselines.base_class import BasePolicy
from benchnpin.baselines.feature_extractors import DenseActionSpaceDQN
from benchnpin.common.metrics.task_driven_metric import TaskDrivenMetric
from benchnpin.common.utils.utils import DotDict
import gymnasium as gym
from collections import namedtuple
import random
import os
import sys
import time
from datetime import datetime
import yaml

import torch
import torch.optim as optim
from torch.nn.functional import smooth_l1_loss
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import logging
import re

logging.getLogger('pymunk').propagate = False


# enable cuDNN auto-tuner to find the best algorithm to use for your hardware
torch.backends.cudnn.benchmark = True

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'ministeps', 'next_state'))

def get_latest_model(model_dir, model_name):
    # List all files in the directory
    files = os.listdir(model_dir)
    
    # Regex to match the model files with step count
    pattern = re.compile(rf'model-{model_name}(\d+)\.pt')
    
    # Extract step counts and corresponding file names
    models = []
    for file in files:
        match = pattern.match(file)
        if match:
            step_count = int(match.group(1))
            models.append((step_count, file))
    
    # Sort by step count and get the latest model
    if models:
        latest_model = max(models, key=lambda x: x[0])[1]
        return os.path.join(model_dir, latest_model)
    else:
        return None

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Meters:
    def __init__(self):
        self.meters = {}

    def get_names(self):
        return self.meters.keys()

    def reset(self):
        for _, meter in self.meters.items():
            meter.reset()

    def update(self, name, val):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(val)

    def avg(self, name):
        return self.meters[name].avg

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
    
    def push(self, *args):
        max_priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)
        self.buffer[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = torch.tensor(self.priorities, dtype=torch.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = random.choices(range(len(self.buffer)), probs, k=batch_size)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        # weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights.clone().detach().float()

        return Transition(*zip(*samples)), indices, weights
    
    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = p.item()
    
    def __len__(self):
        return len(self.buffer)
    
class DenseActionSpacePolicy:
    def __init__(self, action_space, num_input_channels, final_exploration, train=False, checkpoint_path='',
                resume_training=False, evaluate=False, job_id_to_resume=None, random_seed=None,
                model_name='sam_model', model_dir=None, half_action_space=False):
        self.action_space = action_space
        self.num_input_channels = num_input_channels
        self.final_exploration = final_exploration
        self.train = train
        self.half_action_space = half_action_space

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else self.device)
        self.policy_net = self.build_network()
        self.transform = transforms.ToTensor()

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), 'models/')

        # Resume from checkpoint if applicable
        if os.path.exists(checkpoint_path) or resume_training or evaluate:
            if resume_training:
                # model_path = os.path.join(os.path.dirname(__file__), f'checkpoint/{job_id_to_resume}/model-{model_name}121000.pt')
                checkpoint_dir = os.path.join(os.path.dirname(__file__), f'checkpoint/{job_id_to_resume}/')
                model_path = get_latest_model(checkpoint_dir, model_name)
            elif evaluate:
                model_path = os.path.join(model_dir, f'{model_name}.pt')
            else:
                checkpoint_dir = os.path.dirname(checkpoint_path)
                # model_path = f'{checkpoint_dir}/model-{model_name}.pt'
                model_path = get_latest_model(checkpoint_dir, model_name)
            model_checkpoint = torch.load(model_path, map_location=self.device)
            self.policy_net.load_state_dict(model_checkpoint['state_dict'])
            if self.train:
                self.policy_net.train()
            else:
                self.policy_net.eval()
            print(f"=> loaded model '{model_path}'")
            logging.info(f"=> loaded model '{model_path}'")

        if random_seed is not None:
            random.seed(random_seed)

    def build_network(self):
        return torch.nn.DataParallel(
            DenseActionSpaceDQN(num_input_channels=self.num_input_channels, half_action_space=self.half_action_space)
        ).to(self.device)

    def apply_transform(self, s):
        return self.transform(s).unsqueeze(0)

    def predict(self, state, exploration_eps=None, debug=False):
        if exploration_eps is None:
            exploration_eps = self.final_exploration
        state = self.apply_transform(state).to(self.device)
        with torch.no_grad():
            output = self.policy_net(state).squeeze(0)
        if random.random() < exploration_eps:
            action = random.randrange(self.action_space)
        else:
            action = output.view(1, -1).max(1)[1].item()
        info = {}
        if debug:
            info['output'] = output.squeeze(0)
        return action, info

class BoxDeliverySAM(BasePolicy):

    def __init__(self, cfg, model_name='sam_model', model_path=None, job_id=None) -> None:
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else self.device)

        if model_path is None:
            self.model_path = os.path.join(os.path.dirname(__file__), 'models/')
        else:
            self.model_path = model_path

        self.model = None
        self.job_id = job_id

        # Check if preemption occurred and if so, use the config file from current run
        checkpoint_dir = os.path.join(os.path.dirname(__file__), f'checkpoint/{self.job_id}')
        # create checkpoint directory if it does not exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint')]
        if checkpoint_files: # if there exists a checkpoint file, this indicates a run has been interrupted
            config_path = f'{checkpoint_dir}/config.yaml'
            self.cfg = DotDict.load_from_file(config_path)
            self.model_name = f'{self.cfg.train.job_name}_{job_id}'
        else:
            self.cfg = cfg
            self.model_name = model_name




    def update_policy(self, policy_net, target_net, optimizer, batch, transform_func, indices=None, weights=None, replay_buffer=None):
        state_batch = torch.cat([transform_func(s) for s in batch.state]).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        ministeps_batch = torch.tensor(batch.ministeps, dtype=torch.float32).to(self.device)
        non_final_next_states = torch.cat([transform_func(s) for s in batch.next_state if s is not None]).to(self.device, non_blocking=True)

        output = policy_net(state_batch)
        state_action_values = output.view(self.batch_size, -1).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_state_values = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=self.device)

        # Double DQN
        with torch.no_grad():
            best_action = policy_net(non_final_next_states).view(non_final_next_states.size(0), -1).max(1)[1].view(non_final_next_states.size(0), 1)
            next_state_values[non_final_mask] = target_net(non_final_next_states).view(non_final_next_states.size(0), -1).gather(1, best_action).view(-1)

        expected_state_action_values = (reward_batch + torch.pow(self.gamma, ministeps_batch) * next_state_values)
        td_error = torch.abs(state_action_values - expected_state_action_values).detach()

        # if PER is being used we want the loss per sample, not averaged, so we can apply the importance weights manually
        loss_fn = torch.nn.SmoothL1Loss(reduction='none' if weights is not None else 'mean')
        losses = loss_fn(state_action_values, expected_state_action_values)
        if weights is not None:
            loss = (losses * weights.to(self.device)).mean()
        else:
            loss = losses.mean()

        optimizer.zero_grad()
        loss.backward()
        if self.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), self.grad_norm_clipping)
        optimizer.step()

        if indices is not None and replay_buffer is not None:
            new_priorities = (torch.abs(state_action_values - expected_state_action_values) + 1e-6).detach()
            replay_buffer.update_priorities(indices, new_priorities)

        train_info = {}
        train_info['q_value_min'] = output.min().item()
        train_info['q_value_max'] = output.max().item()
        train_info['td_error'] = td_error.mean()
        train_info['loss'] = loss

        return train_info


    def train(self) -> None:
        # create environment
        env = gym.make('box-delivery-v0', cfg=self.cfg)
        env = env.unwrapped
        self.cfg = env.cfg # update cfg with env-specific config
        
        for key in self.cfg:
            print(f"{key}: {self.cfg[key]}")

        params = self.cfg['train']
        self.batch_size = params['batch_size']
        self.checkpoint_freq = params['checkpoint_freq']
        self.final_exploration = params['final_exploration']
        self.gamma = params['gamma']
        self.grad_norm_clipping = params['grad_norm_clipping']
        self.learning_rate = params['learning_rate']
        self.replay_buffer_size = params['replay_buffer_size']
        self.weight_decay = params['weight_decay']

        checkpoint_freq = params['checkpoint_freq']
        exploration_timesteps = params['exploration_timesteps']
        job_id_to_resume = params['job_id_to_resume']
        learning_starts = params['learning_starts']
        resume_training = params['resume_training']
        target_update_freq = params['target_update_freq']
        total_timesteps = params['total_timesteps']

        checkpoint_path = os.path.join(os.path.dirname(__file__), f'checkpoint/{self.job_id}/checkpoint-{self.model_name}.pt')

        log_dir = os.path.join(os.path.dirname(__file__), params['log_dir'])
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logging.basicConfig(filename=os.path.join(log_dir, f'{self.model_name}.log'), level=logging.DEBUG)
        logging.info("starting training...")
        logging.info(f"Job ID: {self.job_id}")

        # policy
        policy = DenseActionSpacePolicy(env.action_space.high, env.num_channels, self.final_exploration,
                                         train=True, checkpoint_path=checkpoint_path, resume_training=resume_training, job_id_to_resume=job_id_to_resume, model_name=self.model_name, random_seed=self.cfg.misc.random_seed, half_action_space=self.cfg.ablation.half_action_space)

        # optimizer
        optimizer = optim.SGD(policy.policy_net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)

        # replay buffer
        if self.cfg.ablation.per:
            replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_size, alpha=self.cfg.ablation.per_alpha)
        else:
            replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # resume if possible
        start_timestep = 0
        episode = 0
        if os.path.exists(checkpoint_path) or resume_training:
            if resume_training:
                checkpoint_path_to_load = os.path.join(os.path.dirname(__file__), f'checkpoint/{job_id_to_resume}/checkpoint-{self.model_name}.pt')
            else:
                checkpoint_path_to_load = checkpoint_path
            checkpoint = torch.load(checkpoint_path_to_load)
            start_timestep = checkpoint['timestep']
            episode = checkpoint['episode']
            optimizer.load_state_dict(checkpoint['optimizer'])
            replay_buffer = checkpoint['replay_buffer']
            print(f"=> loaded checkpoint '{checkpoint_path}' (timestep: {start_timestep})")
            logging.info(f"=> loaded checkpoint '{checkpoint_path}' (timestep: {start_timestep})")
        else:
            print("=> no checkpoint detected, starting from initial state")
            logging.info("=> no checkpoint detected, starting from initial state")
        
        # target net
        target_net = policy.build_network()
        target_net.load_state_dict(policy.policy_net.state_dict())
        target_net.eval()

        # logging
        train_summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, f'_new_{self.model_name}'))
        meters = Meters()

        state, _ = env.reset()
        total_timesteps_with_warmup = total_timesteps + learning_starts
        for timestep in tqdm(range(start_timestep, total_timesteps_with_warmup),
                             initial=start_timestep, total=total_timesteps_with_warmup, file=sys.stdout):
            
            start_time = time.time()

            # select action
            if exploration_timesteps > 0:
                exploration_eps = 1 - min(max(timestep - learning_starts, 0) / exploration_timesteps, 1) * (1 - self.final_exploration)
            else:
                exploration_eps = self.final_exploration
            action, _ = policy.predict(state, exploration_eps=exploration_eps)

            # step the simulation
            if self.cfg.ablation.curriculum:
                if timestep == 50000:
                    replay_buffer.buffer = replay_buffer.buffer[7500:]
                    replay_buffer.priorities = replay_buffer.priorities[7500:]
                    replay_buffer.position = len(replay_buffer.buffer)
                    print("Purging replay buffer to remove samples from the first 50k timesteps")
            next_state, reward, done, truncated, info = env.step(action, curric_starts=(timestep > 50000))
            ministeps = info['ministeps']

            # store in buffer
            replay_buffer.push(state, action, reward, ministeps, next_state)
            state = next_state

            # reset if episode ended
            if done:
                obs_config = None
                if self.cfg.ablation.general:
                    obs_config = random.choice(['large_columns', 'large_divider'])
                state, _ = env.reset(obs_config = obs_config)
                episode += 1
                if truncated:
                    logging.info(f"Episode {episode} truncated. {info['cumulative_boxes']} in goal. Resetting environment...")
                else:
                    logging.info(f"Episode {episode} completed. Resetting environment...")
            
            # train network
            if timestep >= learning_starts:
                if self.cfg.ablation.per:
                    batch, indices, weights = replay_buffer.sample(self.batch_size, beta=self.cfg.ablation.per_beta)
                    train_info = self.update_policy(policy.policy_net, target_net, optimizer, batch, policy.apply_transform, indices=indices, weights=weights, replay_buffer=replay_buffer)
                else:
                    batch = replay_buffer.sample(self.batch_size)
                    train_info = self.update_policy(policy.policy_net, target_net, optimizer, batch, policy.apply_transform)
            
            # update target network
            if (timestep + 1) % target_update_freq == 0:
                target_net.load_state_dict(policy.policy_net.state_dict())
            
            step_time = time.time() - start_time

            ################################################################################
            # Logging
            # meters
            meters.update('step_time', step_time)
            if timestep >= learning_starts:
                for name, val in train_info.items():
                    meters.update(name, val)
            
            if done:
                for name in meters.get_names():
                    train_summary_writer.add_scalar(name, meters.avg(name), timestep + 1)
                eta_seconds = meters.avg('step_time') * (total_timesteps_with_warmup - timestep)
                meters.reset()

                train_summary_writer.add_scalar('episodes', episode, timestep + 1)
                train_summary_writer.add_scalar('eta_hours', eta_seconds / 3600, timestep + 1)

                for name in ['cumulative_boxes', 'cumulative_distance', 'cumulative_reward']:
                    train_summary_writer.add_scalar(name, info[name], timestep + 1)

            ################################################################################
            # Checkpoint
            if (timestep + 1) % checkpoint_freq == 0 or timestep + 1 == total_timesteps_with_warmup:
                checkpoint_dir = os.path.dirname(checkpoint_path)
                model_path = f'{checkpoint_dir}/model-{self.model_name+str(timestep+1)}.pt'
                if not os.path.exists(checkpoint_dir):
                    try:
                        os.makedirs(checkpoint_dir, exist_ok=True)
                    except FileExistsError:
                        print(f"Directory {checkpoint_dir} already exists")
                        logging.info(f"Directory {checkpoint_dir} already exists")
                
                # Save the configuration file
                config_path = f'{checkpoint_dir}/config.yaml'
                with open(config_path, 'w') as file:
                    yaml.dump(dict(self.cfg), file, default_flow_style=False)
                
                # temp_model_path = f'{checkpoint_dir}/model-temp.pt'
                model = {
                    'timestep': timestep + 1,
                    'state_dict': policy.policy_net.state_dict(),
                }

                temp_checkpoint_path = f'{checkpoint_dir}/checkpoint-temp.pt'
                checkpoint = {
                    'timestep': timestep + 1,
                    'episode': episode,
                    'optimizer': optimizer.state_dict(),
                    'replay_buffer': replay_buffer,
                }

                # save model and checkpoint
                torch.save(model, model_path)
                torch.save(checkpoint, temp_checkpoint_path)

                # according to the GNU spec of rename, the state of checkpoint_path
                # is atomic, i.e. it will either be modified or not modified, but not in
                # between, during a system crash (i.e. preemtion)
                # os.replace(temp_model_path, model_path)
                os.replace(temp_checkpoint_path, checkpoint_path)
                msg = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": Checkpoint saved at " + checkpoint_path + self.model_name
                logging.info(msg)
        env.close()



    def evaluate(self, num_eps: int, model_eps: str ='latest'):

        env = gym.make('box-delivery-v0', cfg=self.cfg)
        env = env.unwrapped
        self.cfg = env.cfg # update cfg with env-specific config

        if model_eps == 'latest':
            self.model = DenseActionSpacePolicy(env.action_space.high, env.num_channels, 0.0,
                                                train=False, evaluate=True, model_name=self.model_name, model_dir=self.model_path, half_action_space=self.cfg.ablation.half_action_space)
        else:
            model_checkpoint = self.model_name + '_' + model_eps + '_steps'
            self.model = DenseActionSpacePolicy(env.action_space.high, env.num_channels, 0.0,
                                                train=False, evaluate=True, model_name=model_checkpoint, model_dir=self.model_path, half_action_space=self.cfg.ablation.half_action_space)
        
        metric = TaskDrivenMetric(alg_name="SAM", robot_mass=env.cfg.agent.mass)

        eps_rewards = []
        eps_steps = []
        eps_distance = []
        eps_avg_box_distance = []
        for eps_idx in range(num_eps):
            print("Progress: ", eps_idx, " / ", num_eps, " episodes", end='\r')
            obs, info = env.reset()
            metric.reset(info)
            done = truncated = False
            ep_steps = 0
            ep_reward = 0.0
            while True:
                ep_steps += 1
                # if path completed (endpoint of path is the action), then use the model to predict the action
                # useful for diffusion, as it takes multiple steps to create full path
                if env.path_completed:
                    action, _ = self.model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                ep_reward += reward
                metric.update(info=info, reward=reward, eps_complete=(done or truncated))
                if done or truncated:
                    break
            eps_steps.append(ep_steps)
            eps_rewards.append(ep_reward)
            eps_distance.append(info['cumulative_distance'])
            eps_avg_box_distance.append(sum(env.box_distances.values()) / len(env.box_distances))
        
        env.close()
        metric.plot_scores(save_fig_dir=env.cfg.output_dir)
        avg_eps_steps = sum(eps_steps) / len(eps_steps)
        std_dev_eps_steps = (sum((x - avg_eps_steps) ** 2 for x in eps_steps) / len(eps_steps)) ** 0.5
        avg_eps_rewards = sum(eps_rewards) / len(eps_rewards)
        std_dev_eps_rewards = (sum((x - avg_eps_rewards) ** 2 for x in eps_rewards) / len(eps_rewards)) ** 0.5
        avg_eps_distance = sum(eps_distance) / len(eps_distance)
        std_dev_eps_distance = (sum((x - avg_eps_distance) ** 2 for x in eps_distance) / len(eps_distance)) ** 0.5
        avg_eps_avg_box_distance = sum(eps_avg_box_distance) / len(eps_avg_box_distance)
        std_dev_eps_avg_box_distance = (sum((x - avg_eps_avg_box_distance) ** 2 for x in eps_avg_box_distance) / len(eps_avg_box_distance)) ** 0.5
        avg_success_rate = sum(metric.success_rates) / len(metric.success_rates)
        print(f"Average eps_steps: {avg_eps_steps:.2f} \\pm {std_dev_eps_steps:.2f}")
        print(f"Average eps_rewards: {avg_eps_rewards:.2f} \\pm {std_dev_eps_rewards:.2f}")
        print(f"Average eps_distance: {avg_eps_distance:.2f} \\pm {std_dev_eps_distance:.2f}")
        print(f"Average eps_avg_box_distance: {avg_eps_avg_box_distance:.2f} \\pm {std_dev_eps_avg_box_distance:.2f}")
        print(f"Average success rate: {avg_success_rate}")
        return metric.success_rates, metric.efficiency_scores, metric.effort_scores, metric.rewards, f"SAM_{self.model_name}"


    
    def act(self, observation, action_space: int = None, num_channels: int = None, model_eps='latest'):
        # load trained model for the first time
        if self.model is None:
            if action_space is None or num_channels is None:
                raise ValueError("action_space and num_channels must be provided")
            
            if model_eps == 'latest':
                self.model = DenseActionSpacePolicy(action_space, num_channels, 0.0,
                                                    train=False, evaluate=True, model_name=self.model_name, model_dir=self.model_path, half_action_space=self.cfg.ablation.half_action_space)
            else:
                model_checkpoint = self.model_name + '_' + model_eps + '_steps'
                self.model = DenseActionSpacePolicy(action_space.high, num_channels, 0.0,
                                                    train=False, evaluate=True, model_name=model_checkpoint, model_dir=self.model_path, half_action_space=self.cfg.ablation.half_action_space)

        action, _ = self.model.predict(observation)
        return action
