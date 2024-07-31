import time
import datetime
import os

import torch

from config import GlobalConfig
from offline_training_mec.mec_env.environment_manager import EnvironmentManager
# from offline_training_mec.mec_env.agent.ddpg import Agent
from offline_training_mec.mec_env.agent.action import Action

# from torch.utils.tensorboard import SummaryWriter
#
# from agent.components.transforms import OneHot
# from agent.components.episode_buffer import ReplayBuffer

# from env.env_object.utils import *

import offline_training_mec.rlkit.torch.pytorch_util as ptu
from offline_training_mec.rlkit.data_management.env_replay_buffer import EnvReplayBuffer
# from rlkit.envs.wrappers import NormalizedBoxEnv
# from rlkit.launchers.launcher_util import setup_logger
from offline_training_mec.rlkit.samplers.data_collector import MdpPathCollector
from offline_training_mec.rlkit.torch.PMOEsac.PMOEsac import PMOESACTrainer
from offline_training_mec.rlkit.torch.PMOEsac.policies import MakeDeterministic
from offline_training_mec.rlkit.torch.PMOEsac.policies import TanhPMOEGaussianPolicy
from offline_training_mec.rlkit.torch.networks import FlattenPMOEMlp
from offline_training_mec.rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import numpy as np


def select_action_random(global_config: GlobalConfig):
    return np.random.uniform(0, 1, global_config.agent_config.action_config.dimension)


class AgentOperator:
    def __init__(self, global_config, global_config_eval: GlobalConfig, json_name):
        # init the global config
        self.global_config = global_config
        self.global_config_eval = global_config_eval
        self.json_name = json_name

        # self.output_network_message = self.global_config.control_config.output_network_config
        # self.output_action_message = self.global_config.control_config.output_action_config
        #
        # self.easy_output_mode = self.global_config.control_config.easy_output_mode
        # self.easy_output_cycle = self.global_config.control_config.easy_output_cycle
        self.print_control = True

        # init the train config
        self.train_config = self.global_config.train_config
        self.action_config = self.global_config.agent_config.action_config

        # set the random seed
        np.random.seed(self.train_config.seed)
        torch.manual_seed(self.train_config.seed)

        # init the env manager
        if self.train_config.is_eval_mode:
            self.env_manager = EnvironmentManager(self.global_config_eval)
        else:
            self.env_manager = EnvironmentManager(self.global_config)
        self.env_manager_eval = EnvironmentManager(self.global_config_eval)

        # init the DDPG agent
        if torch.cuda.is_available():
            self.device = torch.device('cuda', index=self.train_config.gpu_index)
        else:
            self.device = torch.device('cpu')
        # self.ddpg_agent = Agent(self.device, self.global_config)

        # init the tensorboard writer
        self.exp_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # if self.train_config.tensorboard:
        #     dir_name = 'runs_new/' + self.json_name + "/" + self.train_config.algorithm \
        #                + '_s_' + str(self.train_config.seed) \
        #                + '_t_' + self.exp_time
        #     self.writer = SummaryWriter(log_dir=dir_name)

        # save the init time
        self.init_time = time.time()

        # init the parameter
        self.ue_num = len(self.env_manager.base_station_set.all_mobile_device_list)
        # self.cur_epsilon = self.ddpg_agent.epsilon_max

        # save the real step count
        self.step_real_count = 0

        self.env_manager.step_real_count = self.step_real_count
        self.env_manager.writer = None #self.writer

        # new add attributes, some of them might be deleted later
        self.mac = None
        self.batch_size = None

        # new add for matrix

        self.t_step_count = 0

    def run_all_episode(self, variant):
        # get state base message
        ue_state = None
        obs_dim = 0
        self.env_manager.reset()
        state_class_list = []
        obs = []
        for mobile_device_id in range(self.ue_num):
            each_state = self.env_manager.get_state_per_mobile_device(mobile_device_id)  # 只有调用这个方法才能得到state对象
            state_class_list.append(each_state)
            state_list = each_state.get_state_list()
            state_array = each_state.get_normalized_state_array()
            obs.append(state_array)
        state = np.concatenate(obs, -1)
        obs_dim = state.shape[0]

        action_dim = len(self.env_manager.base_station_set.all_mobile_device_list) * self.global_config.agent_config.action_config.dimension
        goals_dim = self.global_config.train_config.goals_dim * self.ue_num
        env = self.env_manager
        env_eval = self.env_manager_eval
        env.observation_dim = obs_dim
        env.action_dim = action_dim
        env.goals_dim = goals_dim
        env.ue_num = self.ue_num
        env_eval.observation_dim = obs_dim
        env_eval.action_dim = action_dim
        env_eval.goals_dim = goals_dim
        env_eval.ue_num = self.ue_num

        self.global_config.train_config.algorithm = 'PMOE'
        self.global_config_eval.train_config.algorithm = 'PMOE'
        self.global_config.train_config.episode_num *= len(self.global_config.train_config.env_table)

        M = variant['layer_size']
        qf1 = FlattenPMOEMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        qf2 = FlattenPMOEMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        target_qf1 = FlattenPMOEMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        target_qf2 = FlattenPMOEMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        policy = TanhPMOEGaussianPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            goal_dim=goals_dim,
            hidden_sizes=[M, M],
            k=variant['trainer_kwargs']['k'],
            global_config=self.global_config,
        )
        eval_policy = MakeDeterministic(policy)
        # TODO
        eval_path_collector = MdpPathCollector(
            env_eval,
            eval_policy,
            self.global_config
        )
        expl_path_collector = MdpPathCollector(
            env,
            policy,
            self.global_config
        )
        replay_buffer = EnvReplayBuffer(
            variant['replay_buffer_size'],
            env,
        )
        replay_buffer_eval = EnvReplayBuffer(
            variant['replay_buffer_size'],
            env_eval,
        )
        trainer = PMOESACTrainer(
            env=env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            global_config=self.global_config,
            **variant['trainer_kwargs']
        )
        trainer_eval = PMOESACTrainer(
            env=env_eval,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            global_config=self.global_config,
            **variant['trainer_kwargs']
        )
        algorithm = TorchBatchRLAlgorithm(
            global_config = self.global_config,
            trainer=trainer,
            is_eval_model=self.train_config.is_eval_mode,
            exploration_env=env,
            evaluation_env=env_eval,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            replay_buffer_eval=replay_buffer_eval,
            **variant['algorithm_kwargs']
        )
        algorithm.to(ptu.device)
        algorithm.train()

        print("Finished Training")
