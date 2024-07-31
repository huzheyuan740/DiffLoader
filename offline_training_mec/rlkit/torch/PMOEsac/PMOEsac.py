from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from torch.distributions import Categorical
from torch.nn import functional as f

import offline_training_mec.rlkit.torch.pytorch_util as ptu
from offline_training_mec.rlkit.core.eval_util import create_stats_ordered_dict
from offline_training_mec.rlkit.torch.torch_rl_algorithm import TorchTrainer


class PMOESACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            global_config,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            k=4
    ):
        super().__init__()
        self.k = k
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.is_eval_mode = False

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_dim).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.train_config = global_config.train_config
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        # print("buffer_reward:", rewards)
        terminals = batch['terminals']
        obs = batch['observations']
        goals = batch['goals']
        actions = batch['actions']
        next_obs = batch['next_observations']
        pri_indexs = batch['pri_indexs']
        # print("pri_indexs:\n", pri_indexs)

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, mixing_coefficient, *_ = self.policy(
            obs, goals, reparameterize=True, return_log_prob=True,
        )

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions, Critic_Repeat=True),
            self.qf2(obs, new_obs_actions, Critic_Repeat=True),
        )
        # print("q_new_actions:\n", q_new_actions)

        _, best_index = torch.max(q_new_actions, 1)

        if self.train_config.coefficient_back_propagation_approach == 'individual':
            for idx, index in enumerate(pri_indexs.cpu().numpy().reshape(1, -1)[0]):
                index = int(index)
                # q_new_actions[idx, 0] = q_new_actions[idx, index]
                best_index[idx] = index
            # q_new_actions = q_new_actions[:, 0]
        elif self.train_config.coefficient_back_propagation_approach == 'max':
            _, best_index = torch.max(q_new_actions, 1)

        if self.train_config.policy_back_propagation_approach == 'individual':
            for idx, index in enumerate(pri_indexs.cpu().numpy().reshape(1, -1)[0]):
                index = int(index)
                q_new_actions[idx, 0] = q_new_actions[idx, index]
                # best_index[idx] = index
            q_new_actions = q_new_actions[:, 0]
        elif self.train_config.policy_back_propagation_approach == 'max':
            q_new_actions, _ = torch.max(q_new_actions, 1)
        else:  # all
            q_new_actions = torch.mean(q_new_actions, 1)
            log_pi = torch.mean(log_pi, 1)
            best_index[:] = torch.randint(0, new_obs_actions.shape[1], (best_index.shape))

        if self.train_config.policy_back_propagation_approach == 'individual' or self.train_config.policy_back_propagation_approach == 'max':
            log_pi = torch.gather(log_pi, 1, best_index.unsqueeze(-1))
            log_pi = log_pi.squeeze(1)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        if self.train_config.category_sample_method == 'gumbel':
            mixing_coefficient = f.gumbel_softmax(mixing_coefficient, tau=self.train_config.gumbel_tau)
        mixing_coefficient_v = f.one_hot(best_index, self.k).float()
        # print("mixing_coefficient_v:", mixing_coefficient_v)
        mixing_coefficient_loss = f.mse_loss(mixing_coefficient_v, mixing_coefficient)

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        # print("best_index:{}\nq_new_actions:{}".format(best_index, q_new_actions))

        """
        QF Loss
        """

        with torch.no_grad():
            # Make sure policy accounts for squashing functions like tanh correctly!
            new_next_actions, new_next_means, _, new_log_pi, new_next_mixing_coefficient, _, new_next_stds, *_ = self.policy(
                next_obs, goals, reparameterize=True, return_log_prob=True,
            )

            if self.train_config.critic_index_calculation == 'cate':
                choose_index = Categorical(new_next_mixing_coefficient).sample().unsqueeze(-1)
            elif self.train_config.critic_index_calculation == 'max_q':
                target_q_values = torch.min(
                    self.target_qf1(next_obs, new_next_actions, Critic_Repeat=True),
                    self.target_qf2(next_obs, new_next_actions, Critic_Repeat=True),
                )
                _, choose_index = torch.max(target_q_values, 1)
                choose_index = choose_index.view(-1, 1)

            if len(new_next_actions.shape) == 2:
                new_next_actions = new_next_actions.unsqueeze(1)
            new_next_actions = torch.gather(new_next_actions,
                                            1,
                                            choose_index.unsqueeze(-1).repeat(1, 1, new_next_actions.shape[-1]))
            new_next_actions = new_next_actions.squeeze(1)

            new_next_means = torch.gather(new_next_means,
                                            1,
                                            choose_index.unsqueeze(-1).repeat(1, 1, new_next_means.shape[-1]))
            new_next_stds = torch.gather(new_next_stds,
                                            1,
                                            choose_index.unsqueeze(-1).repeat(1, 1, new_next_stds.shape[-1]))
            # 根据某一选定源语再采样多次action
            all_sample_actions = None
            all_sample_log_probs = None
            for i in range(self.train_config.sample_action_num_in_critic_training):
                sample_actions, sample_log_probs = self.policy.sample_action(new_next_means, new_next_stds)
                if all_sample_actions is None and all_sample_log_probs is None:
                    all_sample_actions = sample_actions
                    all_sample_log_probs = sample_log_probs
                else:
                    all_sample_actions = torch.cat((all_sample_actions, sample_actions), 1)
                    all_sample_log_probs = torch.cat((all_sample_log_probs, sample_log_probs), 1)

            all_target_q_values_from_sample = torch.min(
                self.target_qf1(next_obs, all_sample_actions, Critic_Repeat=True),
                self.target_qf2(next_obs, all_sample_actions, Critic_Repeat=True),
            )
            _, choose_index = torch.max(all_target_q_values_from_sample, 1)
            choose_index = choose_index.view(-1, 1)
            new_next_actions = torch.gather(all_sample_actions,
                                            1,
                                            choose_index.unsqueeze(-1).repeat(1, 1, all_sample_actions.shape[-1]))
            new_next_actions = new_next_actions.squeeze(1)
            new_log_pi = all_sample_log_probs

            new_log_pi = torch.gather(new_log_pi, 1, choose_index)

            target_q_values = torch.min(
                self.target_qf1(next_obs, new_next_actions),
                self.target_qf2(next_obs, new_next_actions),
            ) - alpha * new_log_pi

            q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # print("q1_pred:",torch.mean(q1_pred, 0))
        # print("q_target:", torch.mean(q_target, 0))
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        # print("qf1_loss:{},qf2_loss:{},policy_loss{},mixing_coefficient_loss{}".format(qf1_loss,qf2_loss,policy_loss,mixing_coefficient_loss))
        if self.is_eval_mode:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        else:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics['Mixing Coefficient Loss'] = np.mean(ptu.get_numpy(mixing_coefficient_loss))

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            # self.eval_statistics['Q1 Predictions value'] = ptu.get_numpy(q1_pred).shape
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Mixing Coefficient',
                ptu.get_numpy(mixing_coefficient),
            ))
            # TODO
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )
