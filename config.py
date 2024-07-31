import numpy as np
import copy


class TrainConfig:
    def __init__(self):
        self.algorithm = "DiffLoader"
        self.total_timesteps = 2e5
        self.step_num = 20
        self.episode_num = int(self.total_timesteps / self.step_num)
        self.part = 1
        self.seed = 0
        self.gpu_index = 0
        self.tensorboard = True
        self.is_eval_mode = False
        self.dynamic_update_a_b_episode_begin = 300e3
        self.dynamic_update_a_b_episode_range = 1000
        self.dynamic_update_a_b_fluctuate = 27
        self.is_save_json = False

        self.if_save_buffer = False  # 是否存buffer用于画heatmap
        # self.if_load_buffer = False
        self.if_load_buffer = not self.if_save_buffer
        self.load_buffer_name = 'bug'  # D+载入good  D-载入bad
        self.save_model_path = '/mnt/windisk/dataset_diffloader_mec/model/'
        self.save_buffer_npy_path = '/mnt/windisk/dataset_diffloader_mec/buffer/'
        self.if_generate_prompt = False
        self.save_prompt_npy_path = '/mnt/windisk/dataset_diffloader_mec/prompt/'
        self.prompt_step = 5
        self.prompt_type = 'normal'  # 'normal', 'language', 'one-hot'
        self.env_table = [
            # MHz, GHz, kb
            [6, 5, 500],
            [3, 2, 100],
            [3, 2, 1000],
            [3, 10, 100],
            [3, 10, 1000],
            [10, 2, 100],
            [10, 2, 1000],
            [10, 10, 100],
            [10, 10, 1000],
            [8, 4, 800],
            [9, 5, 600],
            [4, 6, 300],
            [5, 7, 500],
            [2, 8, 700],
            [9, 4, 400],
        ]
        self.test_env_list = ['SAC_t_2024-07-04-07-48-17_env_8_4_800', 'SAC_t_2024-07-04-07-48-17_env_8.1_4.1_800.1']  #
        self.test_env_parm_list = [
            [4, 6, 300],
            [5, 3, 700],
            [5, 6, 600],
            [6, 8, 400],
            [8, 10, 700],
            [11, 12, 1100],
            [2, 11, 90],
            [12, 15, 1200],
        ]

        # baseline config
        # maml
        self.maml_policy_model_path = '/mnt/windisk/dataset_diffloader_mec/MAML-based/first-ten/policy-9000.pt'#'/mnt/windisk/dataset_diffloader_mec/MAML-based/mec/policy-8754.pt'
        # pmoe
        self.primitive_num = 3
        self.goals_dim = 4
        self.is_eval_mode = False
        self.policy_back_propagation_approach = 'max'  # 'max' 'individual' 'all'
        self.coefficient_back_propagation_approach = 'max'  # 'max' 'individual'
        self.category_sample_method = 'no_gumbel'  # 'gumbel' ,'no_gumbel'
        self.critic_index_calculation = 'max_q'  # 'max_q', 'cate'
        self.gumbel_tau = 0.2
        self.sample_action_num_in_critic_training = 10
        # SAC and pmoe
        self.cheackpoint_path = '/mnt/windisk/dataset_diffloader_mec/model/seed_0/SAC_t_2024-07-09-23-54-46_env_5_6_600/reward_2.106878131565037/'

        # testbed config
        self.is_testbed = False


class BaseStationSetConfig:
    def __init__(self):
        self.base_station_num = 1
        self.mobile_device_num = 20
        self.data_transmitting_rate = None
        self.bandwidth = None
        self.bandwidth_max = 10
        self.channel_number = 5
        self.path_loss = 4
        self.noise_power = 10 ** ((-100-30)/10)
        self.time_step_max = 0.5

        self.base_station_config = BaseStationConfig()
        self.mobile_device_config = MobileDeviceConfig()
        self.task_config = TaskConfig()


class BaseStationConfig:
    def __init__(self):
        self.base_station_computing_ability = None
        self.base_station_computing_ability_list = [6.0, 12.0]
        self.base_station_computing_ability_eval = 8.8
        self.base_station_computing_ability_eval_list = [8.6]  # [8.8 * 10 ** 9, 9.3 * 10 ** 9]
        self.base_station_computing_ability_max = 25.0
        self.base_station_computing_ability_eval_max = 18
        self.base_station_energy = 200.0
        self.base_station_height = 20.0
        self.task_queue_size_max = 100


class MobileDeviceConfig:
    def __init__(self):
        self.mobile_device_ability = 1.0
        self.mobile_device_ability_max = 3.0
        self.queue_time = 0.0
        self.transmitting_time_to_base_station_max = 1.5
        self.distance_to_base_station_max = 80
        self.transmitting_power = 0.1  # 单位:W
        # 能量目前暂时用不到d
        self.energy_coefficient = 10 ** (-27)
        self.user_equipment_energy = (10 ** -27) * ((1 * 1024*1024*1024) ** 2) * 1000 * 1024 * 1000 * 21
        # print("energy:", self.user_equipment_energy)
        # exit()
        self.task_queue_size_max = 100


class TaskConfig:
    def __init__(self):
        self.task_data_size_min = 100
        self.task_data_size_max = 1000
        self.task_data_size_now = None
        self.origin_task_data_size_now = copy.deepcopy(self.task_data_size_now)
        self.task_data_index_list = [0]
        self.task_data_size_now_eval = [7]
        self.task_date_size_std = 100 / 3  # [200, 220]
        self.task_date_size_std_eval = [0.1, 0.2]
        self.task_date_size_std_max = 0.5
        self.task_date_size_std_min = 0
        self.task_tolerance_delay_list = 1
        self.task_tolerance_delay_max = 5


class CostConfig:
    def __init__(self):
        self.time_cost_weight = 0.5
        self.energy_cost_weight = 0.5


class RewardConfig:
    def __init__(self):
        # self.penalty_over_time = -1000
        self.cost_to_reward = -10
        self.init_reward = 0  #
        self.mean_reward = self.init_reward
        self.lowest_reward = -10000
        self.adjust_bias_for_normal = 10


class EnvInterfaceConfig:
    def __init__(self):
        self.cost_config = CostConfig()
        self.reward_config = RewardConfig()


class StateConfig:
    def __init__(self):
        self.control_config = ControlConfig()
        self.train_config = TrainConfig()
        self.dimension = 9


class ActionConfig:
    def __init__(self):
        self.control_config = ControlConfig()
        self.train_config = TrainConfig()
        if self.train_config.is_testbed:
            self.dimension = 1
        else:
            self.dimension = 2
        self.action_noise = np.random.uniform(0, 1, self.dimension)
        self.action_noise_decay = 0.995
        self.threshold_to_offload = 0.0  # 0.5


class TorchConfig:
    def __init__(self):
        self.gamma = 0.98
        self.hidden_sizes = (128, 128)
        self.buffer_size = int(4e3)  # int(4e3)
        self.max_seq_length = 50
        # self.buffer_size = int(64)
        self.batch_size = 4
        self.policy_learning_rate = 1e-4
        self.critic_learning_rate = 1e-3
        self.policy_gradient_clip = 0.5
        self.critic_gradient_clip = 1.0
        self.epsilon_max = 1.0
        self.epsilon_decay = 0.998
        self.action_limit = 1.0


class AgentConfig:
    def __init__(self):
        self.state_config = StateConfig()
        self.action_config = ActionConfig()
        self.torch_config = TorchConfig()


class DebugConfig:
    def __init__(self):
        self.whether_output_finish_episode_reason = 1
        self.whether_output_replay_buffer_message = 0


class ControlConfig:
    def __init__(self):
        self.save_runs = True
        self.save_save_model = True

        self.output_network_config = True
        self.output_action_config = True
        self.output_other_config = False

        self.easy_output_mode = True
        self.easy_output_cycle = 100
        self.env_without_D2D = True
        self.bandwidth_disturb = True


class GlobalConfig:
    def __init__(self):
        self.train_config = TrainConfig()
        self.agent_config = AgentConfig()
        self.base_station_set_config = BaseStationSetConfig()
        self.interface_config = EnvInterfaceConfig()
