# from gym.envs.classic_control import PendulumEnv as env

import offline_training_mec.rlkit.torch.pytorch_util as ptu
from offline_training_mec.rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from offline_training_mec.rlkit.launchers.launcher_util import setup_logger
from offline_training_mec.rlkit.samplers.data_collector import MdpPathCollector
from offline_training_mec.rlkit.torch.networks import FlattenMlp
from offline_training_mec.rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from offline_training_mec.rlkit.torch.sac.sac import SACTrainer
from offline_training_mec.rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from config import GlobalConfig
from offline_training_mec.mec_env.environment_manager import EnvironmentManager
import numpy as np
from diffuser.utils import set_seed


def experiment(variant, global_config):
    # expl_env = NormalizedBoxEnv(env())
    # eval_env = NormalizedBoxEnv(env())
    env = EnvironmentManager(global_config)
    env.reset()
    state_class_list = []
    obs = []
    for mobile_device_id in range(len(env.base_station_set.all_mobile_device_list)):
        each_state = env.get_state_per_mobile_device(mobile_device_id)
        state_class_list.append(each_state)
        state_list = each_state.get_state_list()
        state_array = each_state.get_normalized_state_array()
        obs.append(state_array)
    state = np.concatenate(obs, -1)
    obs_dim = state.shape[0]
    action_dim = len(env.base_station_set.all_mobile_device_list) * global_config.agent_config.action_config.dimension
    env.observation_dim = obs_dim
    env.action_dim = action_dim
    # env.goals_dim = goals_dim
    env.ue_num = len(env.base_station_set.all_mobile_device_list)

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(  # for now, env_eval == env
        env,
        eval_policy,
        global_config
    )
    expl_path_collector = MdpPathCollector(
        env,
        policy,
        global_config
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        env,
    )
    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        global_config=global_config,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        global_config=global_config,
        trainer=trainer,
        is_eval_model=False,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        replay_buffer_eval=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    global_config = GlobalConfig()
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=global_config.train_config.episode_num,
            num_eval_steps_per_epoch=global_config.train_config.step_num,
            num_trains_per_train_loop=10,
            num_expl_steps_per_train_loop=global_config.train_config.step_num,
            min_num_steps_before_training=global_config.train_config.step_num,
            max_path_length=global_config.train_config.step_num,
            batch_size=global_config.agent_config.torch_config.batch_size,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    # setup_logger(env.__name__, variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    for env_idx, env_pram in enumerate(global_config.train_config.env_table):
        # Set seed
        set_seed(global_config.train_config.seed)
        # Set Environment
        if global_config.train_config.is_testbed:
            global_config.base_station_set_config.data_transmitting_rate = env_pram[0]  # [Mbps]
        else:
            global_config.base_station_set_config.bandwidth = env_pram[0]  # [MHz]
        global_config.base_station_set_config.base_station_config.base_station_computing_ability = env_pram[1]  # [GHz]
        global_config.base_station_set_config.task_config.task_data_size_now = env_pram[2]  # [kb]
        # Run Experiment
        experiment(variant, global_config)
