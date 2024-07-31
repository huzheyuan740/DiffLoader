import abc
from collections import OrderedDict
from time import sleep

import gtimer as gt
import os
import json

from offline_training_mec.rlkit.core import logger, eval_util
from offline_training_mec.rlkit.data_management.replay_buffer import ReplayBuffer
from offline_training_mec.rlkit.samplers.data_collector import DataCollector


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            global_config,
            trainer,
            is_eval_model,
            exploration_env,
            evaluation_env,
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
            replay_buffer_eval: ReplayBuffer,
    ):
        self.trainer = trainer
        self.is_eval_model = is_eval_model
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self.replay_buffer_eval = replay_buffer_eval
        self._start_epoch = 0

        self.post_epoch_funcs = []
        self.cluster_dict = {}

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        # gt.stamp('saving')
        self._log_stats(epoch)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.replay_buffer_eval.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        for k, v in self.replay_buffer_eval.get_snapshot().items():
            snapshot['replay_buffer_eval/' + k] = v
        return snapshot

    def _log_stats(self, epoch):
        # return 0
        if False:
            logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

            """
            Replay Buffer
            """
            logger.record_dict(
                self.replay_buffer.get_diagnostics(),
                prefix='replay_buffer/'
            )

            logger.record_dict(
                self.replay_buffer_eval.get_diagnostics(),
                prefix='replay_buffer_eval/'
            )

            """
            Trainer
            """
            logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

            """
            Exploration
            """
            expl_diagnostics = self.expl_data_collector.get_diagnostics()
            logger.record_dict(
                expl_diagnostics,
                prefix='exploration/'
            )
            expl_paths = self.expl_data_collector.get_epoch_paths()
            if hasattr(self.expl_env, 'get_diagnostics'):
                logger.record_dict(
                    self.expl_env.get_diagnostics(expl_paths),
                    prefix='exploration/',
                )
            logger.record_dict(
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/",
            )
            """
            Evaluation
            """
            eval_diagnostics = self.eval_data_collector.get_diagnostics()
            logger.record_dict(
                eval_diagnostics,
                prefix='evaluation/',
            )
            eval_paths = self.eval_data_collector.get_epoch_paths()
            if hasattr(self.eval_env, 'get_diagnostics'):
                logger.record_dict(
                    self.eval_env.get_diagnostics(eval_paths),
                    prefix='evaluation/',
                )
            # logger.record_dict(
            #     eval_util.get_generic_path_information(eval_paths),
            #     prefix="evaluation/",
            # )

            """
            Misc
            """
            gt.stamp('logging')
            logger.record_dict(_get_epoch_timings())
            logger.record_tabular('Epoch', epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

        # expl_diagnostics = self.expl_data_collector.get_diagnostics()
        # eval_diagnostics = self.eval_data_collector.get_diagnostics()
        # json_path = os.path.join(logger.get_snapshot_dir(), "data_for_cluster.json")
        # self.cluster_dict[str(epoch)] = {}
        # cluster_dict_expl = self.cluster_dict[str(epoch)]['exploration'] = {}
        # cluster_dict_expl['observations'] = expl_diagnostics['observations'].tolist()
        # cluster_dict_expl['goals_all'] = expl_diagnostics['goals_all'].tolist()
        # cluster_dict_expl['actions'] = expl_diagnostics['actions'].tolist()
        # cluster_dict_expl['actions_all'] = expl_diagnostics['actions_all'].tolist()
        # cluster_dict_expl['mixing_coefficients'] = expl_diagnostics['mixing_coefficients'].tolist()
        # cluster_dict_expl['indexs'] = expl_diagnostics['indexs'].tolist()
        # cluster_dict_expl['cost_avg_list'] = expl_diagnostics['cost_avg_list'].tolist()
        # cluster_dict_expl['rewards'] = expl_diagnostics['rewards'].tolist()
        #
        # cluster_dict_eval = self.cluster_dict[str(epoch)]['evaluation'] = {}
        # cluster_dict_eval['observations'] = eval_diagnostics['observations'].tolist()
        # cluster_dict_eval['goals_all'] = eval_diagnostics['goals_all'].tolist()
        # cluster_dict_eval['actions'] = eval_diagnostics['actions'].tolist()
        # cluster_dict_eval['actions_all'] = eval_diagnostics['actions_all'].tolist()
        # cluster_dict_eval['mixing_coefficients'] = eval_diagnostics['mixing_coefficients'].tolist()
        # cluster_dict_eval['indexs'] = eval_diagnostics['indexs'].tolist()
        # cluster_dict_eval['cost_avg_list'] = eval_diagnostics['cost_avg_list'].tolist()
        # cluster_dict_eval['rewards'] = eval_diagnostics['rewards'].tolist()
        # with open(json_path, "w") as f:
        #     json.dump(self.cluster_dict, f, indent=2)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
