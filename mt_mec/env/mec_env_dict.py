import numpy as np

from offline_training_mec.mec_env.environment_manager import EnvironmentManager_MT as ENV_COST
from config import GlobalConfig

global_config = GlobalConfig()

EASY_MODE_CLS_DICT = {}
for idx in range(len(global_config.train_config.env_table)):
    EASY_MODE_CLS_DICT[f'mec-job{idx+1}'] = ENV_COST

EASY_MODE_ARGS_KWARGS = {
    key: dict(args=[], kwargs={'obs_type': 'plain'})
    for key, _ in EASY_MODE_CLS_DICT.items()
}
for idx in range(len(global_config.train_config.env_table)):
    EASY_MODE_ARGS_KWARGS[f'mec-job{idx+1}']['kwargs']['task_type'] = f'mec_type{idx+1}'

