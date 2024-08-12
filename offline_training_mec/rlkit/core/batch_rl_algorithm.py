import abc

import gtimer as gt

from offline_training_mec.rlkit.core.rl_algorithm import BaseRLAlgorithm
from offline_training_mec.rlkit.data_management.replay_buffer import ReplayBuffer
from offline_training_mec.rlkit.samplers.data_collector import PathCollector
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
# from visdom import Visdom
import torch
import os
from config import GlobalConfig
import time


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            global_config,
            trainer,
            is_eval_model,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            replay_buffer_eval: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            global_config,
            trainer,
            is_eval_model,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            replay_buffer_eval,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.exp_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._global_config = global_config
        self._base_station_set_config = self._global_config.base_station_set_config
        self._task_config = self._base_station_set_config.task_config
        self._mobile_device_config = self._base_station_set_config.mobile_device_config
        self._base_station_config = self._base_station_set_config.base_station_config
        # self._channel_gain_config = self._base_station_set_config.channel_gain_config
        self._action_config = self._global_config.agent_config.action_config
        # self.viz = Visdom(env="test4_bs_ca_eval_8.6_bandwidth_eval_9.5")
        if self._global_config.train_config.algorithm != 'PMOE':
            dir_name = '/mnt/windisk/dataset_diffloader_mec/tensorboard/' + "seed_" + str(
                self._global_config.train_config.seed) + "/" + 'SAC' \
                       + '_t_' + self.exp_time + '_env_' + str(
                self._global_config.base_station_set_config.bandwidth) + "_" + str(
                self._global_config.base_station_set_config.base_station_config.base_station_computing_ability) + "_" + str(
                self._global_config.base_station_set_config.task_config.task_data_size_now)
            self.writer = SummaryWriter(log_dir=dir_name)
            self.save_model_path = self._global_config.train_config.save_model_path
            self.save_buffer_npy_path = self._global_config.train_config.save_buffer_npy_path
            self.buffer_npy_path = self.save_buffer_npy_path + "seed_" + str(
                self._global_config.train_config.seed) + '/SAC' \
                                   + '_t_' + self.exp_time + '_env_' + str(
                self._global_config.base_station_set_config.bandwidth) + "_" + str(
                self._global_config.base_station_set_config.base_station_config.base_station_computing_ability) + "_" + str(
                self._global_config.base_station_set_config.task_config.task_data_size_now)
            os.makedirs(self.buffer_npy_path, exist_ok=True)
        else:
            dir_name = '/mnt/windisk/dataset_diffloader_mec/PMOE-based/tensorboard/' + "seed_" + str(
                self._global_config.train_config.seed) + "/" + 'PMOE' \
                       + '_t_' + self.exp_time + '_env_' + str(
                self._global_config.base_station_set_config.bandwidth) + "_" + str(
                self._global_config.base_station_set_config.base_station_config.base_station_computing_ability) + "_" + str(
                self._global_config.base_station_set_config.task_config.task_data_size_now)
            self.writer = SummaryWriter(log_dir=dir_name)
            self.save_model_path = '/mnt/windisk/dataset_diffloader_mec/PMOE-based/model/'
            # self.save_buffer_npy_path = '/mnt/windisk/dataset_diffloader_mec/PMOE-based/buffer/'
            # self.buffer_npy_path = self.save_buffer_npy_path + "seed_" + str(
            #     self._global_config.train_config.seed) + '/PMOE' \
            #                        + '_t_' + self.exp_time + '_env_' + str(
            #     self._global_config.base_station_set_config.bandwidth) + "_" + str(
            #     self._global_config.base_station_set_config.base_station_config.base_station_computing_ability) + "_" + str(
            #     self._global_config.base_station_set_config.task_config.task_data_size_now)

    def visualization(self, epoch, new_paths, title):
        ue_num = self._base_station_set_config.user_equipment_num
        primative_num = self._global_config.train_config.primitive_num
        state_dim = self._global_config.agent_config.state_config.dimension
        goal_dim = self._global_config.train_config.goals_dim
        path_length = new_paths[0]['path_length']
        observations = (new_paths[0]['observations']).reshape(path_length, ue_num, state_dim)
        means = new_paths[0]['means']
        stds = new_paths[0]['stds']
        actions = new_paths[0]['actions']
        data_transmission_rate = new_paths[0]['data_transmission_rate_all'].reshape(path_length, ue_num, -1)

        for i in range(path_length):
            means[i][1] = (means[i][1] + 1) / 2
            means[i][3] = (means[i][3] + 1) / 2

    def _train(self):
        network_name = ['policy', 'qf1', 'qf2', 'target_qf1', 'target_qf2']
        if self.is_eval_model:
            checkpoint_path = self._global_config.train_config.cheackpoint_path
            for idx, net in enumerate(self.trainer.networks):
                net.load_state_dict(torch.load("{}/{}.pth".format(checkpoint_path, network_name[idx])))
            print("Load Model Success....")

        if self.min_num_steps_before_training > 0:
            print("self.expl_data_collector.collect_new_paths:", self.expl_data_collector.collect_new_paths)
            self.expl_data_collector.num_epoch = 0
            self.expl_data_collector.is_expl = True
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
        print("---------------------before training end")

        reward_max = -np.inf
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            print("==============================:", epoch)
            self.eval_data_collector.num_epoch = epoch
            self.eval_data_collector.is_expl = False
            new_eval_paths = self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            self.writer.add_scalar("episode_reward_message/path_length_eval", np.mean(new_eval_paths[0]['path_length']),
                                   epoch)
            self.writer.add_scalar("episode_reward_message/offload_num_eval",
                                   np.mean(new_eval_paths[0]['data_transmission_rate_all']), epoch)
            self.writer.add_scalar("episode_reward_message/reward_eval_eval", np.mean(new_eval_paths[0]['rewards']),
                                   epoch)
            self.writer.add_scalar("episode_reward_message/cost_eval", np.mean(new_eval_paths[0]['cost_avg_list']),
                                   epoch)
            # gt.stamp('evaluation sampling')
            print("---------------------evaluation end")
            for _ in range(self.num_train_loops_per_epoch):
                self.expl_data_collector.num_epoch = epoch
                self.expl_data_collector.is_expl = True
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                print("---------------------exploration end")
                reward_now = np.mean(new_expl_paths[0]['rewards'])
                self.writer.add_scalar("episode_reward_message/reward", reward_now, epoch)
                # print("new_expl_paths:", np.mean(new_expl_paths[0]['rewards']))
                self.writer.add_scalar("episode_reward_message/path_length", np.mean(new_expl_paths[0]['path_length']),
                                       epoch)
                self.writer.add_scalar("episode_reward_message/offload_num",
                                       np.mean(new_expl_paths[0]['data_transmission_rate_all']), epoch)
                # self.writer.add_scalar("episode_reward_message/reward", np.mean(new_expl_paths[0]['rewards']), epoch)
                self.writer.add_scalar("episode_reward_message/cost", np.mean(new_expl_paths[0]['cost_avg_list']),
                                       epoch)
                dict_one_episode_raplay_buffer = {}
                dict_one_episode_raplay_buffer["observations"] = new_expl_paths[0]["observations"]
                dict_one_episode_raplay_buffer["actions"] = new_expl_paths[0]["actions"]
                dict_one_episode_raplay_buffer["next_observations"] = new_expl_paths[0]["next_observations"]
                dict_one_episode_raplay_buffer["rewards"] = new_expl_paths[0]["rewards"]
                dict_one_episode_raplay_buffer["terminals"] = new_expl_paths[0]["terminals"]
                if self._global_config.train_config.algorithm != 'PMOE':
                    np.save(os.path.join(self.buffer_npy_path, str(epoch) + '.npy'), dict_one_episode_raplay_buffer)

                self.replay_buffer.add_paths(new_expl_paths)

                self.training_mode(True)
                if self.is_eval_model:
                    for _ in range(self.num_trains_per_train_loop):
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size)
                        self.trainer.is_eval_mode = True
                        self.trainer.train(train_data)
                else:
                    for _ in range(self.num_trains_per_train_loop):
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size)
                        self.trainer.train(train_data)

                    if epoch % 1000 == 0 or reward_now > reward_max:
                        reward_max = reward_now
                        save_path = self.save_model_path + '/' + "seed_" + str(
                            self._global_config.train_config.seed) + "/" + 'SAC' \
                                    + '_t_' + self.exp_time + '_env_' + str(
                            self._global_config.base_station_set_config.bandwidth) + "_" + str(
                            self._global_config.base_station_set_config.base_station_config.base_station_computing_ability) + "_" + str(
                            self._global_config.base_station_set_config.task_config.task_data_size_now) + "/reward_" + str(
                            np.mean(new_expl_paths[0]['rewards']))
                        # save_path = self.save_model_path + 'PMOE' + '_t_' + self.exp_time \
                        #             + "/reward_" + str(np.mean(new_expl_paths[0]['rewards']))
                        os.makedirs(save_path, exist_ok=True)
                        for idx, net in enumerate(self.trainer.networks):
                            torch.save(net.state_dict(), "{}/{}.pth".format(save_path, network_name[idx]))

                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)
