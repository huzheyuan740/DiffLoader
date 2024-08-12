import random
from config import GlobalConfig
from offline_training_mec.mec_env.base_station import BaseStation
from offline_training_mec.mec_env.mobile_device import MobileDevice


class BaseStationSet:
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self.base_station_set_config = global_config.base_station_set_config

        self.task_data_size_list = self.base_station_set_config.task_config.task_data_size_now
        # self.origin_task_data_size_list = self.base_station_set_config.task_config.origin_task_data_size_now
        self.task_tolerance_delay_list = self.base_station_set_config.task_config.task_tolerance_delay_list
        self.task_data_index_list = self.base_station_set_config.task_config.task_data_index_list
        if self.global_config.train_config.is_testbed:
            self.data_transmitting_rate = self.base_station_set_config.data_transmitting_rate
        else:
            self.bandwidth = self.base_station_set_config.bandwidth
            self.bandwidth_max = self.base_station_set_config.bandwidth_max

        self.base_station_list = []
        self.all_mobile_device_list = []

        self.base_station_num = self.base_station_set_config.base_station_num
        self.mobile_device_num = self.base_station_set_config.mobile_device_num
        self.base_station0 = BaseStation(0, global_config)
        self.base_station0.computing_ability_now = \
            self.base_station0.computing_ability_now
        for mobile_device_idx in range(self.mobile_device_num):
            mobile_device = MobileDevice(mobile_device_idx, global_config)
            self.all_mobile_device_list.append(mobile_device)

        self.base_station_list = [self.base_station0]

        assert len(self.base_station_list) == self.base_station_num
        assert len(self.all_mobile_device_list) == self.mobile_device_num

    def update_state(self):
        pass

    def shuffle_task_size_list(self):
        assert len(self.task_data_size_list) == len(self.task_tolerance_delay_list)
        shuffled_list = random.sample(self.task_data_index_list, len(self.task_data_index_list))
        self.task_data_size_list = self.base_station_set_config.task_config.task_data_size_now = [
            self.task_data_size_list[self.task_data_index_list.index(i)] for i in shuffled_list]
        self.task_tolerance_delay_list = self.base_station_set_config.task_config.task_tolerance_delay_list = [
            self.task_tolerance_delay_list[self.task_data_index_list.index(i)] for i in shuffled_list]
        self.task_data_index_list = self.base_station_set_config.task_config.task_data_index_list = shuffled_list

    def update_all_mobile_device_message(self):
        for mobile_device_id, mobile_device in enumerate(self.all_mobile_device_list):
            mobile_device.create_task(mobile_device_id)

    def get_state_per_mobile_device(self):
        pass

    def draw_image(self):
        pass
