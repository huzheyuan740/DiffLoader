import gym
import torch
import multiprocessing as mp

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes
from offline_training_mec.mec_env.environment_manager import EnvironmentManager
import numpy as np


def make_env(env_name):
    """
    return a function
    :param env_name:
    :return:
    """

    def _make_env():
        return gym.make(env_name)

    return _make_env


def get_env(global_config):
    """
    return a function
    :param env_name:
    :return:
    """

    return EnvironmentManager(global_config)


class BatchSampler:

    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count(), global_config=None):
        """

        :param env_name:
        :param batch_size: fast batch size
        :param num_workers:
        """
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.queue = mp.Queue()
        self.global_config = global_config
        # [lambda function]
        # env_factorys = [make_env(env_name) for _ in range(num_workers)]
        env_pram = self.global_config.train_config.env_table[0]
        if global_config.train_config.is_testbed:
            global_config.base_station_set_config.data_transmitting_rate = env_pram[0]  # [Mbps]
        else:
            global_config.base_station_set_config.bandwidth = env_pram[0]
        global_config.base_station_set_config.base_station_config.base_station_computing_ability = env_pram[1]  # [GHz]
        global_config.base_station_set_config.task_config.task_data_size_now = env_pram[2]  # [kb]
        env_factorys = []
        for _ in range(num_workers):
            env = EnvironmentManager(global_config)
            # create_env_instance = EnvironmentManager(global_config).get_instance_creator(global_config)
            # env = create_env_instance(global_config)
            env.reset()
            env_factorys.append(env)

        # this is the main process manager, and it will be in charge of num_workers sub-processes interacting with
        # environment.
        self.envs = SubprocVecEnv(env_factorys, queue_=self.queue)
        self._env = EnvironmentManager(global_config)

    def sample(self, policy, params=None, gamma=0.95, device='cpu', writer=None):
        """

        :param policy:
        :param params:
        :param gamma:
        :param device:
        :return:
        """
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        # print("self.queue.empty():", self.queue.qsize())
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)

        observations, state_class_list, batch_ids = self.envs.reset()
        dones = [False]
        step_count = np.zeros((observations.shape[0], ))
        while (not all(dones)):  # if all done and queue is empty  # or (not self.queue.empty())
            # for reinforcement learning, the forward process requires no-gradient
            with torch.no_grad():
                # convert observation to cuda
                # compute policy on cuda
                # convert action to cpu
                observations, state_class_list, batch_ids = self.envs.create_mectask_per_step()
                observations_tensor = torch.from_numpy(observations).to(device=device)
                observations_tensor = observations_tensor.float()
                # forward via policy network
                # policy network will return Categorical(logits=logits)
                # print("policy:", policy)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()

            # print("self.envs.step:", self.envs.step)
            new_observations, rewards, dones, new_batch_ids, cost_array, offload_count = self.envs.step(state_class_list, actions, step_count)
            # print("dones:", dones)
            # if dones:
            #     break
            step_count += 1
            # here is observations NOT new_observations, batch_ids NOT new_batch_ids
            # batch_ids = (0,)
            # print("batch_ids:", batch_ids)
            episodes.append(observations, actions, rewards, batch_ids, cost_array)
            observations, batch_ids = new_observations, new_batch_ids
        print("step_count:", step_count)
        while not self.queue.empty():
            item = self.queue.get(True)

        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks, global_config):
        # tasks = self._env.unwrapped.sample_tasks(num_tasks)
        tasks = global_config.train_config.env_table
        return tasks

    def sample_eval_tasks(self, num_tasks, global_config):
        # tasks = self._env.unwrapped.sample_tasks(num_tasks)
        tasks = global_config.train_config.test_env_parm_list
        return tasks
