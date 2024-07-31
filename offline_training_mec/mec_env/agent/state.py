import numpy as np

from offline_training_mec.mec_env.base_station_set import BaseStationSet
from offline_training_mec.mec_env.base_station import BaseStation
from offline_training_mec.mec_env.mobile_device import MobileDevice


class State:
    def __init__(self, mobile_device: MobileDevice, base_station_set: BaseStationSet):
        self.mobile_device = mobile_device
        self.mobile_device_id = mobile_device.mobile_device_id

        self.base_station_list = base_station_set.base_station_list
        self.mobile_device_computing_ability = mobile_device.computing_ability_now  # for BS_com_ability_list
        self.mobile_device_list = base_station_set.all_mobile_device_list  # for other task size and mask
        self.task_data_size = mobile_device.task.task_data_size  # 不知道这里之后有没有是否能读到当前任务量值的问题
        self.task_tolerance_delay = mobile_device.task.task_tolerance_delay
        self.task_info = mobile_device.task
        self.task_data_index_list = base_station_set.base_station_set_config.task_config.task_data_index_list
        self.global_config = base_station_set.global_config
        if self.global_config.train_config.is_testbed:
            self.data_transmitting_rate = base_station_set.data_transmitting_rate
        else:
            self.bandwidth = base_station_set.bandwidth
            self.bandwidth_max = base_station_set.bandwidth_max

        self.mobile_device_task_queue_current_data_size = mobile_device.task_queue_current_data_size
        self.mobile_device_task_queue_size_max = mobile_device.task_queue_size_max
        self.base_station_task_current_sum_process_time_list = []
        self.base_station_task_queue_size_max_list = [base_station.task_queue_size_max for base_station
                                                      in self.base_station_list]

        # 以下为实际state的各个属性
        self.base_station_set_computing_ability_list = self.get_base_station_set_computing_ability_list(
            self.base_station_list)
        # self.mobile_device_computing_ability self.task_data_size self.task_tolerance_delay上面已定义
        # self.all_task_size_list, self.task_size_mask_list = self.get_other_task_size(self.mobile_device_list,
        #                                                                              self.mobile_device_id)
        self.all_task_size_list = self.get_task_size(self.mobile_device_list,
                                                           self.mobile_device_id)
        # print("self.all_task_size_list:", self.all_task_size_list)
        if self.global_config.train_config.is_testbed:
            self.transmitting_time_to_all_base_station_list = [
                self.all_task_size_list[0] / 1024 / self.data_transmitting_rate]
        else:
            self.distance_to_base_station = self.mobile_device.distance_to_base_station

        self.energy_now = self.mobile_device.energy_now
        self.energy_max = self.mobile_device.energy_max

        self.base_station_task_current_sum_process_time_list = self.get_base_station_task_current_sum_process_time_list(
            self.base_station_list)  # 这里记录的是当前MD观察到的所有BS的队列中的任务量状态
        self.last_base_station_offload_choice = mobile_device.last_base_station_offload_choice

        # TODO state的标准化使用一个统一的函数处理

        self.goals = None

    def get_base_station_set_computing_ability_list(self, base_station_list):
        base_station_set_computing_ability_list = []
        for base_station in base_station_list:
            base_station_set_computing_ability_list.append(base_station.computing_ability_now)
        return base_station_set_computing_ability_list

    def get_other_task_size(self, mobile_device_list, mobile_device_id):
        all_task_size_list = []
        task_size_mask_list = []
        mask_item = 0
        # print("self.task_data_index_list", self.task_data_index_list)
        for idx, mobile_device in enumerate(mobile_device_list):
            if self.task_data_index_list[mobile_device_id] == idx:
                mask_item = 1
            else:
                mask_item = 0
            all_task_size_list.append(mobile_device_list[self.task_data_index_list.index(idx)].task.task_data_size)
            task_size_mask_list.append(mask_item)
        # print("all_task_size_list:", all_task_size_list)
        # print("task_size_mask_list:", task_size_mask_list)
        return all_task_size_list, task_size_mask_list

    def get_task_size(self, mobile_device_list, mobile_device_id):
        all_task_size_list = []
        task_size_mask_list = []
        mask_item = 0
        # print("self.task_data_index_list", self.task_data_index_list)
        for idx, mobile_device in enumerate(mobile_device_list):
            if idx == mobile_device_id:
                all_task_size_list.append(mobile_device_list[idx].task.task_data_size)
        # print("all_task_size_list:", all_task_size_list)
        # print("task_size_mask_list:", task_size_mask_list)
        return all_task_size_list

    def get_base_station_task_current_sum_process_time_list(self, base_station_list):
        base_station_task_current_sum_process_time_list = []
        for base_station in base_station_list:
            task_current_sum_process_time = base_station.task_queue.get_task_current_sum_process_time()
            base_station_task_current_sum_process_time_list.append(task_current_sum_process_time)
        return base_station_task_current_sum_process_time_list

    def get_state_list(self):  # TODO 未进行state的标准化
        state_list = []
        base_station_set_computing_ability_list = self.get_base_station_set_computing_ability_list(
            self.base_station_list)
        state_list.extend(base_station_set_computing_ability_list)
        state_list.append(self.mobile_device_computing_ability)
        state_list.append(self.task_data_size)
        state_list.append(self.task_tolerance_delay)
        self.all_task_size_list = self.get_task_size(self.mobile_device_list, self.mobile_device_id)

        self.base_station_task_current_sum_process_time_list = self.get_base_station_task_current_sum_process_time_list(
            self.base_station_list)  # 这里记录的是当前MD观察到的所有BS的队列中的任务量状态

        if self.global_config.train_config.is_testbed:
            self.transmitting_time_to_all_base_station_list = [
                self.all_task_size_list[0] / 1024 / self.data_transmitting_rate]
            state_list += self.transmitting_time_to_all_base_station_list
        else:
            state_list += [self.bandwidth] + [self.distance_to_base_station]
        # self.last_base_station_offload_choice = self.mobile_device.last_base_station_offload_choice
        # state_list.append(self.last_base_station_offload_choice)
        state_list += [self.energy_now]
        state_list += self.base_station_task_current_sum_process_time_list

        return state_list

    def get_normalized_state_array(self):
        state_array = self.get_state_array()
        # print("state_array:\n", state_array)
        normalized_state_array = np.zeros_like(state_array)

        base_station_computing_ability_max = self.base_station_list[0].computing_ability_max
        normalized_state_array[0] = state_array[0] / base_station_computing_ability_max

        mobile_device_computing_ability_max = self.mobile_device_list[0].computing_ability_max
        normalized_state_array[1] = state_array[1] / mobile_device_computing_ability_max

        task_data_size_max = self.mobile_device.task.task_data_size_max
        normalized_state_array[2] = state_array[2] / task_data_size_max
        normalized_state_array[3] = state_array[3] / self.mobile_device.task.task_tolerance_delay_max
        # normalized_state_array[5:8] = state_array[5:8] / task_data_size_max
        # normalized_state_array[8:11] = state_array[8:11]

        if self.global_config.train_config.is_testbed:
            normalized_state_array[4] = state_array[4] / self.mobile_device_list[
                0].transmitting_time_to_base_station_max
        else:
            normalized_state_array[4] = state_array[4] / self.bandwidth_max
            normalized_state_array[5] = state_array[5] / self.global_config.base_station_set_config.mobile_device_config.distance_to_base_station_max

        normalized_state_array[6] = state_array[6] / self.energy_max
        # TODO 暂时没有归一化各个BS的任务处理时间
        normalized_state_array[-1] = state_array[-1]

        if self.global_config.train_config.is_testbed:
            self.goals = [self.task_info.task_config.task_data_size_now / task_data_size_max,
                          self.task_info.task_config.task_date_size_std / 100,
                          normalized_state_array[0], self.data_transmitting_rate / 5]
        else:
            self.goals = [self.task_info.task_config.task_data_size_now / task_data_size_max,
                          self.task_info.task_config.task_date_size_std / 100,
                          normalized_state_array[0], normalized_state_array[4]]
        # print("normalized_state_array:\n", normalized_state_array)
        # exit()
        return normalized_state_array

    def get_state_array(self):
        import numpy as np
        return np.array(self.get_state_list())
