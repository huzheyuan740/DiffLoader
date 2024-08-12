from config import GlobalConfig
import numpy as np


class Action:
    def __init__(self, action_list, global_config: GlobalConfig):
        self.offload_choice_idx = action_list[0]
        if not global_config.train_config.is_testbed:
            origin_wireless_channel = action_list[1]
            self.wireless_channel = int(np.floor((origin_wireless_channel + 1) * global_config.base_station_set_config.channel_number / 2))

        self.global_config = global_config
        self.action_config = global_config.agent_config.action_config

    def get_action_list(self):
        return [self.offload_choice_idx,]

    def get_action_array(self):
        import numpy as np
        return np.array([self.offload_choice_idx,])

    def get_whether_offload(self):
        if self.offload_choice_idx > self.action_config.threshold_to_offload:
            return True
        else:
            return False

    def get_random_action(self):
        pass

    def get_determined_action(self, offload_choice_idx):  # , offload_task_percentage
        pass
