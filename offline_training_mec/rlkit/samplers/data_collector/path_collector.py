from cmath import e
from collections import deque, OrderedDict
from this import s
import numpy as np
from offline_training_mec.rlkit.core.eval_util import create_stats_ordered_dict
from offline_training_mec.rlkit.samplers.data_collector.base import PathCollector
from offline_training_mec.rlkit.samplers.rollout_functions import rollout, multitask_rollout


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            global_config,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._global_config = global_config
        self._base_station_set_config = self._global_config.base_station_set_config
        self._task_config = self._base_station_set_config.task_config
        self._mobile_device_config = self._base_station_set_config.mobile_device_config
        self._base_station_config = self._base_station_set_config.base_station_config
        # self._channel_gain_config = self._base_station_set_config.channel_gain_config
        self._action_config = self._global_config.agent_config.action_config
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._num_steps_total = 0
        self._num_paths_total = 0
        self.num_epoch = 0
        self.is_expl = True

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        count = 0
        count += 1
        max_path_length_this_loop = min(  # Do not go over num_steps
            max_path_length,
            num_steps - num_steps_collected,
        )
        assert max_path_length_this_loop == self._global_config.train_config.step_num
        path = rollout(
            self._env,
            self._policy,
            max_path_length=max_path_length_this_loop,
            num_epoch=self.num_epoch,
            is_expl=self.is_expl
        )
        path_len = len(path['actions'])
        num_steps_collected += path_len
        paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    # restore observations (hh)
    def restore_observations(self, obs_normalized):
        path_length = self._epoch_paths[0]['path_length']
        for i in range(path_length):
            for j in range(self._base_station_set_config.mobile_device_num):  # ue num
                for k in range(9):  # state dimensions
                    if k == 0:      # task_data_size
                        obs_normalized[i][j*9+k] = obs_normalized[i][j*9+k] * (self._task_config.task_data_size_max \
                            - self._task_config.task_data_size_min) + self._task_config.task_data_size_min
                    elif k == 1:    # task_computing_size
                        obs_normalized[i][j*9+k] = obs_normalized[i][j*9+k] * (self._task_config.task_computing_size_max \
                            - self._task_config.task_computing_size_min) + self._task_config.task_computing_size_min
                    elif k == 2:    # task_tolerance_delay
                        obs_normalized[i][j*9+k] = obs_normalized[i][j*9+k] * (self._task_config.task_tolerance_delay_max \
                            - self._task_config.task_tolerance_delay_min) + self._task_config.task_tolerance_delay_min
                    elif k == 3:    # energy_now
                        obs_normalized[i][j*9+k] = obs_normalized[i][j*9+k] * self._mobile_device_config.user_equipment_energy
                    elif k == 4:    # channel_gain_to_base_station
                        obs_normalized[i][j*9+k] = obs_normalized[i][j*9+k] * (self._channel_gain_config.channel_gain_max \
                            - self._channel_gain_config.channel_gain_min) + self._channel_gain_config.channel_gain_min
                    elif k == 5:    # user_equipment_computing_ability
                        obs_normalized[i][j*9+k] = obs_normalized[i][j*9+k] * self._mobile_device_config.user_equipment_computing_ability
                    elif k == 6:    # user_equipment_queue_time_now
                        obs_normalized[i][j*9+k] = obs_normalized[i][j*9+k]
                    elif k == 7:    # base_station_computing_ability
                        obs_normalized[i][j*9+k] = obs_normalized[i][j*9+k] * self._base_station_config.base_station_computing_ability
                    elif k == 8:    # base_station_queue_time_now
                        obs_normalized[i][j*9+k] = obs_normalized[i][j*9+k]
        return obs_normalized

    # restore observations(hh)
    def restore_next_observations(self, obs_next_normalized):
        return self.restore_observations(obs_next_normalized)

    def restore_actions(self, act_normalized):
        path_length = self._epoch_paths[0]['path_length']
        for i in range(path_length):
            for j in range(self._base_station_set_config.user_equipment_num):  # ue num
                for k in range(2):  # action dimensions
                    if k == 0:      # offload
                        if act_normalized[i][j*2+k] > self._action_config.threshold_to_offload:
                            act_normalized[i][j*2+k] = 1
                        else:
                            act_normalized[i][j*2+k] = 0
                    elif k == 1:    # 
                        act_normalized[i][j*2+k] = (act_normalized[i][j*2+k] + 1) / 2 * self._mobile_device_config.max_transmitting_power
        return act_normalized

    def restore_goals(self, goal_normalized):
        path_length = self._epoch_paths[0]['path_length']
        goals_dim = self._global_config.train_config.goals_dim
        for i in range(path_length):
            for j in range(self._base_station_set_config.user_equipment_num):  # ue num
                for k in range(goals_dim):  # goal dimensions
                    if k == 0:      # mean
                        goal_normalized[i][j*goals_dim+k] = goal_normalized[i][j*goals_dim+k] * (self._task_config.task_data_size_max \
                            - self._task_config.task_data_size_min) + self._task_config.task_data_size_min
                    elif k == 1:    # std
                        goal_normalized[i][j*goals_dim+k] = goal_normalized[i][j*goals_dim+k] * (self._task_config.task_date_size_std[-1] \
                            - self._task_config.task_date_size_std[0]) + self._task_config.task_date_size_std[0]
                    elif k == 2:    # base_station_computing_ability
                        goal_normalized[i][j*goals_dim+k] = goal_normalized[i][j*goals_dim+k] * \
                            self._base_station_set_config.base_station_config.base_station_computing_ability_max
                    elif k == 3:    # base_station_bandwidth
                        goal_normalized[i][j*goals_dim+k] = goal_normalized[i][j*goals_dim+k] * \
                            self._global_config.env_config.env_interface_config.channel_config.bandwidth_max
                    
        return goal_normalized

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        # stats = OrderedDict([
        #     ('num steps total', self._num_steps_total),
        #     ('num paths total', self._num_paths_total),
        # ])
        # stats.update(create_stats_ordered_dict(
        #     "path length",
        #     path_lens,
        #     always_show_all_stats=True,
        # ))
        # print("x:", self._epoch_paths)
        path_length = self._epoch_paths[0]['path_length']
        obs_normalized = self._epoch_paths[0]['observations']
        obs_restored = self.restore_observations(obs_normalized)

        goal_normalized = self._epoch_paths[0]['goals_all']
        goal_restored = self.restore_goals(goal_normalized)
        
        obs_next_normalized = self._epoch_paths[0]['next_observations']
        obs_next_restored  = self.restore_next_observations(obs_next_normalized)
        act_normalized = self._epoch_paths[0]['actions']
        act_restored = self.restore_actions(act_normalized)

        mixing_coefficients = self._epoch_paths[0]['mixing_coefficients']
        act_all_normalized = self._epoch_paths[0]['actions_all']
        indexs = self._epoch_paths[0]['indexs']
        act_all_restored = np.array([])
        for primitive_idx in range(act_all_normalized.shape[2]):
            each_act = act_all_normalized[:, 0, primitive_idx, :]
            each_act_restored = self.restore_actions(each_act)
            act_all_restored = np.append(act_all_restored, each_act_restored)
        act_all_restored = act_all_normalized.reshape(act_normalized.shape[0], act_all_normalized.shape[2], -1)

        ue_num = self._base_station_set_config.user_equipment_num
        stats = OrderedDict([
            ('observations', obs_restored.reshape(path_length, ue_num, -1)),
            ('goals_all', self._epoch_paths[0]['goals_all'].reshape(path_length, -1)),
            ('actions', act_restored.reshape(path_length, -1)),
            ('actions_all', act_all_restored.reshape(path_length, -1)),
            ('mixing_coefficients', mixing_coefficients.reshape(path_length, -1)),
            ('indexs', indexs.reshape(path_length, -1)),
            ('means', self._epoch_paths[0]['means'].reshape(path_length, -1)),
            ('stds', self._epoch_paths[0]['stds'].reshape(path_length, -1)),
            ('rewards', self._epoch_paths[0]['rewards'].reshape(path_length, -1)),
            ('next_observations', obs_next_restored.reshape(path_length, ue_num, -1)),
            ('terminals', self._epoch_paths[0]['terminals'].reshape(path_length, -1)),
            ('path_length', path_length),
            ('cost_avg_list', self._epoch_paths[0]['cost_avg_list']),
            ('data_transmission_rate_all', self._epoch_paths[0]['data_transmission_rate_all'])
        ])
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )


class GoalConditionedPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = multitask_rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                observation_key=self._observation_key,
                desired_goal_key=self._desired_goal_key,
                return_dict_obs=True,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
