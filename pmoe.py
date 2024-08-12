import argparse
import json
import shutil
import os
from config import GlobalConfig
from run_in_no_D2D_env import AgentOperator
from offline_training_mec.rlkit.launchers.launcher_util import setup_logger
import offline_training_mec.rlkit.torch.pytorch_util as ptu
from diffuser.utils import set_seed


def load_global_config(global_config: GlobalConfig):
    env_table = global_config.train_config.env_table
    for env_id, env_parm in enumerate(env_table):
        # Set Environment
        if global_config.train_config.is_testbed:
            global_config.base_station_set_config.data_transmitting_rate = env_parm[0]  # [Mbps]
        else:
            global_config.base_station_set_config.bandwidth = env_parm[0]  # [MHz]
        global_config.base_station_set_config.base_station_config.base_station_computing_ability = env_parm[1]  # [GHz]
        global_config.base_station_set_config.task_config.task_data_size_now = env_parm[2]  # [kb]
        break

def modify_global_config(global_config: GlobalConfig, options):
    global_config.train_config.part = options["part"]
    global_config.train_config.seed = options["seed"]
    global_config.env_config.hexagon_network_config.base_station_config.base_station_computing_ability_eval_list = \
        options["base_station_computing_ability_eval_list"]
    global_config.env_config.env_interface_config.channel_config.bandwidth_eval_list = options["bandwidth_eval_list"]
    global_config.train_config.policy_back_propagation_approach = options["policy_back_propagation_approach"]
    global_config.train_config.coefficient_back_propagation_approach = options["coefficient_back_propagation_approach"]
    global_config.train_config.category_sample_method = options["category_sample_method"]
    global_config.train_config.critic_index_calculation = options["critic_index_calculation"]


def get_global_config_eval(global_config_eval: GlobalConfig):
    test_env_parm_list = global_config_eval.train_config.test_env_parm_list
    for test_env_id, test_env_parm in enumerate(test_env_parm_list):
        # Set Environment
        if global_config_eval.train_config.is_testbed:
            global_config_eval.base_station_set_config.data_transmitting_rate = test_env_parm[0]  # [Mbps]
        else:
            global_config_eval.base_station_set_config.bandwidth = test_env_parm[0]  # [MHz]
        global_config_eval.base_station_set_config.base_station_config.base_station_computing_ability = test_env_parm[1]  # [GHz]
        global_config_eval.base_station_set_config.task_config.task_data_size_now = test_env_parm[2]  # [kb]


def main():
    global_config = GlobalConfig()
    global_config_eval = GlobalConfig()
    set_seed(global_config.train_config.seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_name", type=str, default="default")
    run_args = parser.parse_args()

    json_name = run_args.json_name
    load_global_config(global_config)

    get_global_config_eval(global_config_eval)
    variant = dict(
        algorithm="PMOEsac",
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
            k=global_config.train_config.primitive_num
        ),
    )
    setup_logger('MEC', variant=variant)
    ptu.set_gpu_mode(True)
    agent_operator = AgentOperator(global_config, global_config_eval, json_name)
    agent_operator.run_all_episode(variant)


if __name__ == "__main__":
    main()
