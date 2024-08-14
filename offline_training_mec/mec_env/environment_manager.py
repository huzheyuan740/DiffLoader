import numpy as np
from config import GlobalConfig
from offline_training_mec.mec_env.agent.state import State
from offline_training_mec.mec_env.agent.action import Action
from offline_training_mec.mec_env.base_station_set import BaseStationSet
import math


# 负责与环境交互
class EnvironmentManager:
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self.base_station_set_config = global_config.base_station_set_config
        self.task_config = self.base_station_set_config.task_config
        self.interface_config = global_config.interface_config

        self.observation_dim = None
        self.action_dim = None
        self.goals_dim = None
        self.ue_num = None

        self.writer = None
        self.step_real_count = None
        self.episode_num_now = 0
        self.is_save_json = global_config.train_config.is_save_json
        # self.load_model_path = global_config.train_config.load_model_path
        # self.load_model_name = global_config.train_config.load_model_name

        self.base_station_set = BaseStationSet(self.global_config)
        self.cost_array_all_available_step_in_episode = np.zeros(
            (self.global_config.train_config.step_num, self.base_station_set.mobile_device_num))
        self.reward_array_all_available_step_in_episode = np.zeros(
            (self.global_config.train_config.step_num, self.base_station_set.mobile_device_num))

    def get_instance_creator(self, default_param):

        def create_instance(param=default_param):
            return EnvironmentManager(param)

        return create_instance

    def reset(self):
        self.base_station_set = BaseStationSet(self.global_config)

        # self.base_station_set.shuffle_task_size_list()
        self.base_station_set.update_all_mobile_device_message()
        self.cost_array_all_available_step_in_episode = np.zeros(
            (self.global_config.train_config.step_num, self.base_station_set.mobile_device_num))
        self.reward_array_all_available_step_in_episode = np.zeros(
            (self.global_config.train_config.step_num, self.base_station_set.mobile_device_num))

    def reset_for_pmoe(self, num_epoch=0, is_expl=True):
        env_table = self.global_config.train_config.env_table
        test_env_parm_list = self.global_config.train_config.test_env_parm_list
        if is_expl:
            # Set Environment
            env_parm = env_table[num_epoch % len(env_table)]
            if self.global_config.train_config.is_testbed:
                self.global_config.base_station_set_config.data_transmitting_rate = env_parm[0]  # [Mbps]
            else:
                self.global_config.base_station_set_config.bandwidth = env_parm[0]
            self.global_config.base_station_set_config.base_station_config.base_station_computing_ability = env_parm[
                1]  # [GHz]
            self.global_config.base_station_set_config.task_config.task_data_size_now = env_parm[2]  # [kb]
        else:
            env_parm = test_env_parm_list[num_epoch % len(test_env_parm_list)]
            if self.global_config.train_config.is_testbed:
                self.global_config.base_station_set_config.data_transmitting_rate = env_parm[0]  # [Mbps]
            else:
                self.global_config.base_station_set_config.bandwidth = env_parm[0]
            self.global_config.base_station_set_config.base_station_config.base_station_computing_ability = env_parm[
                1]  # [GHz]
            self.global_config.base_station_set_config.task_config.task_data_size_now = env_parm[2]  # [kb]

        self.base_station_set = BaseStationSet(self.global_config)

        self.base_station_set.update_all_mobile_device_message()
        self.cost_array_all_available_step_in_episode = np.zeros(
            (self.global_config.train_config.step_num, self.base_station_set.mobile_device_num))
        self.reward_array_all_available_step_in_episode = np.zeros(
            (self.global_config.train_config.step_num, self.base_station_set.mobile_device_num))

    def create_task_per_step(self):
        self.base_station_set.update_all_mobile_device_message()
        for base_station in self.base_station_set.base_station_list:
            base_station.priority_task_list.clear()

    def get_transmit_power(self, idx, power, distance_to_base_station):
        # distance = math.sqrt(pos[2 * uid] * pos[2 * uid] + pos[2 * uid + 1] * pos[2 * uid + 1])
        path_loss = self.global_config.base_station_set_config.path_loss
        result = power * math.pow(distance_to_base_station, -path_loss)
        return result

    def get_transmit_time(self, current_idx, state_class_list, action_class_list):
        assert action_class_list[current_idx].get_whether_offload()
        power_all_except_current = 0
        transmitting_power = self.global_config.base_station_set_config.mobile_device_config.transmitting_power
        current_power = self.get_transmit_power(current_idx, transmitting_power, state_class_list[current_idx].distance_to_base_station)
        task_size = state_class_list[current_idx].task_data_size
        for idx, state_class in enumerate(state_class_list):
            if action_class_list[idx].wireless_channel == action_class_list[current_idx].wireless_channel and idx != current_idx and action_class_list[idx].get_whether_offload():
                power_all_except_current += self.get_transmit_power(idx, transmitting_power, state_class_list[idx].distance_to_base_station)
        noise_power = self.global_config.base_station_set_config.noise_power
        snr = current_power / (power_all_except_current + noise_power)
        data_rate = state_class_list[current_idx].bandwidth * math.log2(1 + snr)
        transmitting_time = task_size / 1024 / data_rate
        return transmitting_time

    def step(self, state_class_list, action_class_list, step_count):
        assert len(state_class_list) == len(action_class_list)
        cost_array = np.zeros(len(state_class_list))
        reward_array = np.zeros(len(state_class_list))
        done_list = [False for _ in range(len(state_class_list))]
        info = {}
        done = False
        for idx, each_mobile_device_state in enumerate(state_class_list):
            if action_class_list[idx].get_whether_offload():
                if self.global_config.train_config.is_testbed:
                    transmit_time = each_mobile_device_state.transmitting_time_to_all_base_station_list[0]
                else:
                    transmit_time = self.get_transmit_time(idx, state_class_list, action_class_list)
            else:
                transmit_time = 99999
            task = self.base_station_set.all_mobile_device_list[idx].task
            assert task.task_from_mobile_device_id == idx
            self.base_station_set.base_station_list[0].priority_task_list.append(
                {'transmit_time': transmit_time, 'task': task})
            self.base_station_set.all_mobile_device_list[
                idx].last_base_station_offload_choice = 0

        for idx, base_station in enumerate(self.base_station_set.base_station_list):
            base_station.priority_task_list.sort(key=lambda x: x['transmit_time'])
            for task_info in base_station.priority_task_list:
                task = task_info['task']
                base_station.task_queue.shared_task_execute_queue.append(
                    task)

                assert task.task_local_finish_time == 0 and task.task_offload_finish_time == 0 and task.task_current_process_time_in_queue == 0
                if not action_class_list[task.task_from_mobile_device_id].get_whether_offload():
                    local_time = task.task_data_size * 900 / \
                                 (self.base_station_set.all_mobile_device_list[
                                      task.task_from_mobile_device_id].computing_ability_now * 1024 * 1024)
                    local_energy = task.task_data_size * 900 * 1024 * self.global_config.base_station_set_config.mobile_device_config.energy_coefficient * (
                                self.base_station_set.all_mobile_device_list[
                                    task.task_from_mobile_device_id].computing_ability_now * 1024 * 1024 * 1024) ** 2
                    task.task_local_finish_time = local_time
                    task.task_local_energy = local_energy
                else:
                    task_switch_time = 0

                    task.task_offload_finish_time += task_info['transmit_time']  # 加上数据传输时间
                    transmit_energy = self.global_config.base_station_set_config.mobile_device_config.transmitting_power * task_info['transmit_time']
                    task.task_offload_energy += transmit_energy

                    task_exe_time = task.task_data_size * 900 / (base_station.computing_ability_now * 1024 * 1024)
                    task.task_current_process_time_in_queue = task_switch_time + task_exe_time
                    task_current_sum_process_time = base_station.task_queue.get_task_current_sum_process_time()
                    task.task_offload_finish_time += task_current_sum_process_time

                task_total_time = max(task.task_local_finish_time, task.task_offload_finish_time)
                task_total_energy = max(task.task_local_energy, task.task_offload_energy)
                cost = self.interface_config.cost_config.time_cost_weight * task_total_time + self.interface_config.cost_config.energy_cost_weight * task_total_energy
                reward = self.interface_config.reward_config.init_reward
                done = False
                cost_array[task.task_from_mobile_device_id] = cost
                if task_total_time > task.task_tolerance_delay:
                    reward = 0
                    reward += self.cost_to_reward_bad(cost)
                    done = True
                    print("over time!!!!!!!!!!!!!!!", task_total_time)
                else:
                    reward = 0
                    reward += self.cost_to_reward_add_adjust_bias_normal(cost)
                    if task_total_energy > self.base_station_set.all_mobile_device_list[task.task_from_mobile_device_id].energy_now:
                        reward = 0
                        reward += self.cost_to_reward_bad(cost)
                        done = True
                        print("run out energy!!!!!!!!!!!!!!!:", self.base_station_set.all_mobile_device_list[task.task_from_mobile_device_id].energy_now)
                    else:
                        reward = 0
                        self.base_station_set.all_mobile_device_list[task.task_from_mobile_device_id].energy_now -= task_total_energy
                        reward += self.cost_to_reward_add_adjust_bias_normal(cost)
                if step_count == self.global_config.train_config.step_num - 1:
                    done = True
                reward_array[task.task_from_mobile_device_id] = reward
                done_list[task.task_from_mobile_device_id] = done

            base_station.task_queue.update_task_sum_process_time(self.base_station_set_config.time_step_max)

        self.cost_array_all_available_step_in_episode[step_count] = cost_array
        self.reward_array_all_available_step_in_episode[step_count] = reward_array
        next_state_class_list = []
        for mobile_device_id in range(len(self.base_station_set.all_mobile_device_list)):
            each_state = self.get_state_per_mobile_device(mobile_device_id)
            next_state_class_list.append(each_state)
            next_state_list = each_state.get_state_list()
        reward_array = np.zeros_like(reward_array)
        cost_array = np.ones_like(cost_array) * np.mean(cost_array)
        cost_array_max = cost_array
        if any(done_list) or step_count == self.global_config.train_config.step_num - 1:
            cost_array, reward_array, cost_array_max = self.cost_reward_function_v2(reward_array, step_count)
        return next_state_class_list, cost_array, reward_array, cost_array_max, done_list, info

    def cost_to_reward_add_adjust_bias_normal(self, cost):
        # reward = self.interface_config.reward_config.cost_to_reward * (cost**2) + self.interface_config.reward_config.adjust_bias_for_normal
        reward = self.interface_config.reward_config.adjust_bias_for_normal * (cost ** (-2))
        return reward

    def cost_to_reward_bad(self, cost):
        reward = self.interface_config.reward_config.cost_to_reward * cost
        return reward

    def cost_reward_function_v2(self, reward_array, step_count):
        cost_array_mean_in_step = np.mean(self.cost_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_mean_in_step = np.mean(self.reward_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_in_last_step = self.reward_array_all_available_step_in_episode[step_count]
        cost_array_in_last_step = self.cost_array_all_available_step_in_episode[step_count]

        cost_array_mean_in_step_in_MD = np.ones_like(cost_array_mean_in_step)
        cost_array_max_in_step_in_MD = np.ones_like(cost_array_mean_in_step)
        reward_array_mean_in_step_in_MD = np.ones_like(reward_array_mean_in_step)

        cost_mean = None
        cost_max = None
        if step_count == self.global_config.train_config.step_num - 1:
            cost_mean = np.mean(cost_array_mean_in_step)
            cost_max = np.max(cost_array_mean_in_step)
            # reward_mean = np.mean(reward_array_mean_in_step)  #
            reward_mean = self.cost_to_reward_add_adjust_bias_normal(cost_mean)
        else:
            reward_array_in_last_step = np.where(reward_array_in_last_step > 0, 0, reward_array_in_last_step)
            non_zero_elements = reward_array_in_last_step[reward_array_in_last_step != 0]
            reward_mean = np.mean(non_zero_elements)
            cost_mean = np.max(cost_array_in_last_step)
            cost_max = np.max(cost_array_in_last_step)
        cost_array_mean_in_step_in_MD = cost_mean * cost_array_mean_in_step_in_MD
        cost_array_max_in_step_in_MD = cost_max * cost_array_max_in_step_in_MD
        reward_array_mean_in_step_in_MD = reward_mean * reward_array_mean_in_step_in_MD
        # reward_array = reward_array + self.global_config.interface_config.reward_config.cost_to_reward * cost_array_mean_in_step_in_MD
        return cost_array_mean_in_step_in_MD, reward_array_mean_in_step_in_MD, cost_array_max_in_step_in_MD

    def cost_reward_function(self, reward_array, step_count):
        cost_array_mean_in_step = np.mean(self.cost_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_mean_in_step = np.mean(self.reward_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_in_last_step = self.reward_array_all_available_step_in_episode[step_count]
        cost_array_mean_in_step_in_MD = np.ones_like(cost_array_mean_in_step)
        reward_array_mean_in_step_in_MD = np.ones_like(reward_array_mean_in_step)
        cost_mean = np.mean(cost_array_mean_in_step)
        if step_count == self.global_config.train_config.step_num - 1:
            reward_mean = np.mean(reward_array_mean_in_step)
        else:
            reward_array_in_last_step = np.where(reward_array_in_last_step > 0, 0, reward_array_in_last_step)
            reward_mean = np.mean(reward_array_in_last_step)
        cost_array_mean_in_step_in_MD = cost_mean * cost_array_mean_in_step_in_MD
        reward_array_mean_in_step_in_MD = reward_mean * reward_array_mean_in_step_in_MD
        # reward_array = reward_array + self.global_config.interface_config.reward_config.cost_to_reward * cost_array_mean_in_step_in_MD
        return cost_array_mean_in_step_in_MD, reward_array_mean_in_step_in_MD

    def cost_reward_function_last_step(self, reward_array, step_count):
        cost_array_mean_in_step = np.mean(self.cost_array_all_available_step_in_episode[:(step_count + 1)], axis=0)
        reward_array_in_last_step = self.reward_array_all_available_step_in_episode[step_count]
        cost_array_mean_in_step_in_MD = np.ones_like(cost_array_mean_in_step)
        reward_array_in_last_step_in_MD = np.ones_like(reward_array_in_last_step)
        cost_mean = np.mean(cost_array_mean_in_step)
        if step_count == self.global_config.train_config.step_num - 1:
            reward_mean = np.mean(reward_array_in_last_step)
        else:
            reward_array_in_last_step = np.where(reward_array_in_last_step > 0, 0, reward_array_in_last_step)
            reward_mean = np.mean(reward_array_in_last_step)
        cost_array_mean_in_step_in_MD = cost_mean * cost_array_mean_in_step_in_MD
        reward_array_mean_in_step_in_MD = reward_mean * reward_array_in_last_step_in_MD
        # reward_array = reward_array + self.global_config.interface_config.reward_config.cost_to_reward * cost_array_mean_in_step_in_MD
        return cost_array_mean_in_step_in_MD, reward_array_mean_in_step_in_MD

    def get_state_per_mobile_device(self, mobile_device_id):
        mobile_device = self.base_station_set.all_mobile_device_list[mobile_device_id]
        state = State(mobile_device, self.base_station_set)
        return state

    def get_random_action(self, global_config):
        action = Action([1, 7 / 8], global_config)
        return action


class EnvironmentManager_MT(EnvironmentManager):
    def __init__(self, obs_type,
            task_type,
            random_init,
            global_config: GlobalConfig):
        super().__init__(global_config)

        self.obs_type = obs_type
        self.env_table = self.global_config.train_config.env_table
        self.env_num = len(self.env_table)
        # self.goal = np.array([0, 0, 0])
        self.goal = np.zeros((self.env_num))
        self.ue_num = self.global_config.base_station_set_config.mobile_device_num
        self.observation_config = self.global_config.agent_config.state_config
        self.action_config = self.global_config.agent_config.action_config
        self.observation_space = self.observation_config.dimension * self.ue_num
        self.action_space = self.action_config.dimension * self.ue_num
