import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.Tasksmeta',
        'diffusion': 'models.GaussianActDiffusion',
        'env_id': 'dial-turn-v2',
        'horizon': 8,
        'n_diffusion_steps': 40,
        'action_weight': 10,
        'num_tasks':4,
        'loss_weights': None,
        'loss_discount': 1.0,
        'predict_epsilon': True,
        'dim_mults': (1,2),
        'is_unet': False,
        'attention': True,
        'renderer': 'utils.MuJoCoRenderer',
        'replay_dir_metaworld': './collect/metaworld',
        'loader': 'datasets.MTValueDataset',
        #'normalizer': 'GaussianNormalizer',#SafeLimitsNormalizer
        'normalizer': 'SafeLimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        "is_walker": False,
        'optimal': True,
        'max_path_length': 1000,
        'discount': 0.99,
        'termination_penalty': -0,
        'normed': True,
        #multi-task dataset
        'dataset_dir':'./collect/walker/dataset',
        'task_list':['walker_run','walker_walk','walker_flip','walker_stand'],
            #['quadruped_walk', 'quadruped_jump','quadruped_run','quadruped_roll_fast'],
        'data_type_list':['replay','replay','replay','replay'],
        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 200000,
        'loss_type': 'huber',
        'n_train_steps': 2e6,
        'batch_size': 16,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 10000,  #
        'sample_freq': 20000,
        'n_saves': 20,
        'save_parallel': False,
        'n_reference': 8,
        'is_mt45': False,
        'bucket': None,
        'device': 'cuda:0',
        'seed': None,
        'inv_task': 'pick-place-v2'
    },


}


#------------------------ overrides ------------------------#

