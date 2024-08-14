import sys
# import sys
sys.path.append(".") 

import torch

import os
import time
import os.path as osp

import numpy as np

from mt_mec.torchrl.utils import get_args
from mt_mec.torchrl.utils import get_params

from mt_mec.torchrl.utils import Logger

args = get_args()
params = get_params(args.config)



import mt_mec.torchrl.policies.continuous_policy as policies_continuous_policy

import mt_mec.torchrl.networks.nets as networks_net
import mt_mec.torchrl.networks.base as networks_base

from mt_mec.torchrl.algo.off_policy.mt_sac import MTSAC
from mt_mec.torchrl.collector.para.async_mt import AsyncMultiTaskParallelCollectorUniform

from mt_mec.torchrl.replay_buffers.shared import AsyncSharedReplayBuffer
import gym

from mt_mec.metaworld_utils.meta_env import get_meta_env

import random


def experiment(args):

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")

    env, cls_dicts, cls_args = get_meta_env( params['env_name'], params['env'], params['meta_env'])

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.eval_worker_nums = args.worker_nums = len(env._task_envs)
    for job_idx, job in enumerate(env._task_envs):
        print(f"job_idx:{job_idx}, job:{job}")
        env_table = job.global_config.train_config.env_table[job_idx]
        if job.global_config.train_config.is_testbed:
            job.global_config.base_station_set_config.data_transmitting_rate = env_table[0]  # [Mbps]
        else:
            job.global_config.base_station_set_config.bandwidth = env_table[0]  # [MHz]
        job.global_config.base_station_set_config.base_station_config.base_station_computing_ability = env_table[1]  # [GHz]
        job.global_config.base_station_set_config.task_config.task_data_size_now = env_table[2]  # [kb]

    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']

    experiment_name = os.path.split( os.path.splitext( args.config )[0] )[-1] if args.id is None \
        else args.id
    logger = Logger( experiment_name , params['env_name'], args.seed, params, args.log_dir )

    params['general_setting']['env'] = env
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device

    params['net']['base_type']=networks_base.MLPBase

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # from torchrl.networks.init import normal_init
    example_ob = env.reset_for_init()

    pf = policies_continuous_policy.GuassianContPolicy(
        input_shape = env.mec_observation_space,
        output_shape = 2 * env.action_space,
        **params['net'] )
    qf1 = networks_net.FlattenNet(
        input_shape = env.mec_observation_space + env.action_space,
        output_shape = 1,
        **params['net'] )
    qf2 = networks_net.FlattenNet(
        input_shape=env.mec_observation_space + env.action_space,
        output_shape = 1,
        **params['net'] )

    # example_ob = env.reset()
    example_dict = { 
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_sample(),
        "rewards": [0],
        "terminals": [False],
        "task_idxs": [0]
    }
    replay_buffer = AsyncSharedReplayBuffer( int(buffer_param['size']),
            args.worker_nums
    )
    replay_buffer.build_by_example(example_dict)

    params['general_setting']['replay_buffer'] = replay_buffer

    epochs = params['general_setting']['pretrain_epochs'] + \
        params['general_setting']['num_epochs']

    params['general_setting']['collector'] = AsyncMultiTaskParallelCollectorUniform(
        env=env, pf=pf, replay_buffer=replay_buffer,
        env_cls = cls_dicts, env_args = [params["env"], cls_args, params["meta_env"]],
        device=device,
        reset_idx=True,
        epoch_frames=params['general_setting']['epoch_frames'],
        max_episode_frames=params['general_setting']['max_episode_frames'],
        eval_episodes = params['general_setting']['eval_episodes'],
        worker_nums=args.worker_nums, eval_worker_nums=args.eval_worker_nums,
        train_epochs = epochs, eval_epochs= params['general_setting']['num_epochs'],
        logger = logger
    )
    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    params['general_setting']['save_dir'] = osp.join(logger.work_dir,"model")
    agent = MTSAC(
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        task_nums=env.num_tasks,
        **params['sac'],
        **params['general_setting']
    )
    agent.train()

if __name__ == "__main__":
    experiment(args)
