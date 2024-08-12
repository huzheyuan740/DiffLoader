import os
import copy
import numpy as np
import torch
import einops
import pdb
import random
from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
import metaworld
import time
import gym
import d4rl
import statistics

DTYPE = torch.float
from collections import namedtuple
import diffuser.utils as utils

DTBatch = namedtuple('DTBatch', 'actions rtg observations timestep mask')
DEVICE = 'cuda'
from torch.utils.tensorboard import SummaryWriter
from offline_training_mec.mec_env.environment_manager import EnvironmentManager
from offline_training_mec.mec_env.agent.action import Action


def cycle(dl):
    while True:
        for data in dl:
            yield data


def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
        # import pdb; pdb.set_trace()
    return torch.tensor(x, dtype=dtype, device=device)


class EMA():
    '''
        empirical moving average
    '''

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class MetaworldTrainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset,
            renderer,
            ema_decay=0.995,
            train_batch_size=32,
            train_lr=2e-5,
            gradient_accumulate_every=2,
            step_start_ema=2000,
            update_ema_every=10,
            log_freq=100,
            sample_freq=1000,
            save_freq=1000,
            label_freq=100000,
            save_parallel=False,
            results_folder='./results',
            n_reference=8,
            bucket=None,
            envs=[],
            task_list=[],
            is_unet=False,
            trainer_device=None,
            horizon=32,
            global_config=None
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.envs = envs
        self.device = trainer_device
        self.horizon = horizon
        self.task_list = task_list
        self.is_unet = is_unet

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.reset_parameters()
        self.step = 0
        self.global_config = global_config

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        best_score, prev_label = 0, 0
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                if self.step != 0:
                    if self.is_unet:
                        score, success_rate = self.evaluate(self.device)
                    else:
                        score, success_rate = self.evaluate(self.device)
                    label = str(label) + '_' + str(score) + '_' + str(success_rate)
                    if score > best_score:
                        self.save(label)
                        best_score = score
                else:
                    self.save(label)
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
            self.step += 1

    def get_env_for_run_list(self, env_list, global_config):
        query_str = 'env_'
        env_for_run_list = []
        for env in env_list:
            index = env.find(query_str)
            env_par = env[index + len(query_str):]
            env_slice = env_par.split('_')
            if global_config.train_config.is_testbed:
                global_config.base_station_set_config.data_transmitting_rate = float(env_slice[0])  # [Mbps]
            else:
                global_config.base_station_set_config.bandwidth = float(env_slice[0])
            global_config.base_station_set_config.base_station_config.base_station_computing_ability = float(
                env_slice[1])  # [GHz]
            global_config.base_station_set_config.task_config.task_data_size_now = float(env_slice[2])
            env_for_run = EnvironmentManager(global_config)
            env_for_run.reset()
            env_for_run_list.append(env_for_run)
        return env_for_run_list

    def get_obs_list(self, env_for_run_list):
        obs_list = []
        state_class_list = []
        for env_for_run in env_for_run_list:
            obs_list_each_env = []
            state_class_list_each_env = []
            env_for_run.create_task_per_step()
            for mobile_device_id in range(len(env_for_run.base_station_set.all_mobile_device_list)):
                each_state = env_for_run.get_state_per_mobile_device(mobile_device_id)
                state_class_list_each_env.append(each_state)
                state_array = each_state.get_normalized_state_array()
                obs_list_each_env.extend(state_array)
            obs_list_each_env = np.array(obs_list_each_env)
            obs_list.append(obs_list_each_env)
            state_class_list.append(state_class_list_each_env)
        return obs_list, state_class_list

    def evaluate(self, device):
        print("++++++++++++++++++eval")
        env_list = self.global_config.train_config.test_env_list
        num_eval = len(env_list)
        env_for_run_list = self.get_env_for_run_list(env_list, self.global_config)
        seed = 0
        # for i in range(len(env_list)):
        #     env_list[i].set_task(task[i])
        #     env_list[i].seed(seed)
        score = 0
        total_success = 0
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 20
        obs_list, state_class_list = self.get_obs_list(env_for_run_list)
        obs = np.array(obs_list)

        cond_task = torch.tensor([i for i in range(num_eval)], device=device).reshape(-1, )
        cond_task += len(self.task_list)
        conditions = torch.zeros([obs.shape[0], 2, obs.shape[-1]], device=device)
        rtg = torch.ones((len(env_list),), device=device) * 6
        episode_reward = np.zeros((len(env_for_run_list)))
        episode_cost_list = np.zeros((max_episode_length, len(env_for_run_list)))
        episode_cost_max_list = np.zeros((max_episode_length, len(env_for_run_list)))
        eposide_reward_list = np.zeros((max_episode_length, len(env_for_run_list)))
        for j in range(max_episode_length):
            print("step:", j)
            obs_list, state_class_list = self.get_obs_list(env_for_run_list)
            obs = np.array(obs_list)
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            conditions = torch.cat([conditions[:, 1:, :], to_torch(obs, device=device).unsqueeze(1)], dim=1)

            time_begin = time.time()
            samples = self.ema_model.conditional_sample(conditions, task=cond_task,
                                                        value=rtg,
                                                        verbose=False, horizon=self.horizon, guidance=1.2)

            action = samples.trajectories[:, 0, :]
            action = action.reshape(-1, self.dataset.action_dim)
            action = to_np(action)
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            for i in range(len(env_for_run_list)):
                a_reshaped = np.clip(action[i], -0.99, 0.99)
                action_class_list = []
                a_reshaped = a_reshaped.reshape(-1, self.global_config.agent_config.action_config.dimension)
                for ue_id, each_action_reshaped in enumerate(a_reshaped):
                    ue_action = Action(each_action_reshaped, self.global_config)
                    action_class_list.append(ue_action)
                next_state_class_list, cost_array, reward_array, cost_array_max, done_list, _ \
                    = env_for_run_list[i].step(state_class_list[i], action_class_list, j)
                reward = reward_array[0]
                temp_reward = reward

                next_obs = []
                for mobile_device_id in range(len(env_for_run_list[i].base_station_set.all_mobile_device_list)):
                    each_state = env_for_run_list[i].get_state_per_mobile_device(
                        mobile_device_id)
                    state_array = each_state.get_normalized_state_array()
                    next_obs.append(state_array)
                next_state = np.concatenate(next_obs, -1)
                done = any(done_list)
                terminal = False
                terminal = (j + 1 >= max_episode_length)
                done = terminal or done

                episode_reward[i] += reward
                episode_cost_list[j][i] = cost_array[0]
                episode_cost_max_list[j][i] = cost_array_max[0]
                eposide_reward_list[j][i] = reward
                if j + 1 == max_episode_length:
                    done = True
                if done:
                    break

        for i in range(len(env_for_run_list)):
            print(f"task:{i},reward:{episode_reward[i]}, cost:{np.mean(episode_cost_list[:, i])},")
        return np.mean(episode_rewards), np.mean(episode_cost_list)

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    # -----------------------------------------------------------------------------#
    # --------------------------------- rendering ---------------------------------#
    # -----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:, None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):
            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:, None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)


class AugTrainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset,
            renderer,
            ema_decay=0.995,
            train_batch_size=32,
            train_lr=2e-5,
            gradient_accumulate_every=2,
            step_start_ema=2000,
            update_ema_every=10,
            log_freq=100,
            sample_freq=1000,
            save_freq=1000,
            label_freq=100000,
            save_parallel=False,
            results_folder='./results',
            n_reference=8,
            bucket=None,
            envs=[],
            task_list=[],
            is_unet=False,
            trainer_device=None,
            horizon=32,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.envs = envs
        self.device = trainer_device
        self.horizon = horizon
        self.task_list = task_list
        self.is_unet = is_unet

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                # print("BATCH LOAD!")
                batch = batch_to_device(batch, self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            self.writer.add_scalar('Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


class MazeTrainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset,
            renderer,
            ema_decay=0.995,
            train_batch_size=32,
            train_lr=2e-5,
            gradient_accumulate_every=2,
            step_start_ema=2000,
            update_ema_every=10,
            log_freq=100,
            sample_freq=1000,
            save_freq=1000,
            label_freq=100000,
            save_parallel=False,
            results_folder='./results',
            n_reference=8,
            bucket=None,
            envs=[],
            task_list=[],
            is_unet=False,
            trainer_device=None,
            horizon=32,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.envs = envs
        self.device = trainer_device
        self.horizon = horizon
        self.task_list = task_list
        self.is_unet = is_unet

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        best_score = 0
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            self.writer.add_scalar('Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                if self.step != 0:
                    score, success_rate = self.evaluate(self.device)
                    label = str(label) + '_' + str(score) + '_' + str(success_rate)
                    if score > best_score:
                        self.save(label)
                        best_score = score
                else:
                    score, success_rate = self.evaluate(self.device)
                    self.save(label)
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
            self.step += 1

    def evaluate(self, device):
        num_eval = 10
        env_list = [gym.make(self.envs[i]) for j in range(num_eval) for i in range(len(self.envs))]
        score = 0
        dones = [False for j in range(num_eval) for i in range(len(self.envs))]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 600
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(self.envs))],
                                 device=device).reshape(-1, )
        conditions = torch.zeros([obs.shape[0], 5, obs.shape[-1]], device=device)
        rtg = torch.ones((len(env_list),), device=device) * 0.95
        while False in dones:  # for _ in range(max_episode_length):#
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            conditions = torch.cat([conditions[:, 1:, :], to_torch(obs, device=device).unsqueeze(1)], dim=1)
            samples = self.ema_model.conditional_sample(conditions, task=cond_task,
                                                        value=rtg,
                                                        verbose=False, horizon=self.horizon, guidance=1.2)
            action = samples.trajectories[:, 0, :]
            action = action.reshape(-1, self.dataset.action_dim)
            action = to_np(action)
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            for i in range(len(env_list)):
                if not dones[i]:
                    next_observation, reward, dones[i], info = env_list[i].step(action[i])
                    obs_list.append(next_observation[None])
                    episode_rewards[i] += reward
                else:
                    obs_list.append(torch.zeros(1, self.dataset.observation_dim))
            obs = np.concatenate(obs_list, axis=0)
        for i in range(len(self.envs)):
            tmp = []
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * len(self.envs)])
            this_score = statistics.mean(tmp)
            score += this_score
            print(f"task:{self.envs[i]},mean episodic return:{this_score}, "
                  f"std:{statistics.stdev(tmp)}")
        return score, 1.

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
