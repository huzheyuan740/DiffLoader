from tkinter import OUTSIDE
import numpy as np
import torch
from torch import nn as nn

from offline_training_mec.rlkit.policies.base import ExplorationPolicy, Policy
from offline_training_mec.rlkit.torch.core import eval_np
from offline_training_mec.rlkit.torch.distributions import TanhNormal
from offline_training_mec.rlkit.torch.networks import PMOEMlp

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhPMOEGaussianPolicy(PMOEMlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            goal_dim,
            std=None,
            init_w=1e-3,
            global_config=None,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            goal_size=goal_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        self.train_config = global_config.train_config
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim * self.k)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, goals_np, deterministic=False):
        out = self.get_actions(obs_np[None], goals_np[None], deterministic=deterministic)
        mixing_coefficient = out[4][0]
        # print("origin:", mixing_coefficient)
        if self.train_config.category_sample_method == 'gumbel':
            mixing_coefficient_tmp = torch.Tensor(mixing_coefficient)
            mixing_coefficient = nn.functional.gumbel_softmax(mixing_coefficient_tmp, tau=self.train_config.gumbel_tau,
                                                              hard=True)
            _, index = torch.max(mixing_coefficient, 0)
            index = index.numpy()
            # mixing_coefficient = nn.functional.gumbel_softmax(mixing_coefficient_tmp, tau=self.train_config.gumbel_tau,
            #                                                   hard=False)
            mixing_coefficient = mixing_coefficient.numpy()
        else:
            index = np.random.choice(self.k, p=mixing_coefficient)
        action = (out[0][:, index])[0, :]
        action_all = out[0]
        mean = (out[1][:, index])[0, :]
        mean_all = out[1]
        log_prob = (out[2][:, index])
        log_std = {}  # (out[3][:, index])
        # if self.train_config.category_sample_method != 'gumbel':
        #     mixing_coefficient = (out[4][:, index])
        # print("final:", mixing_coefficient)
        entropy = {}  # (out[5][:, index])[0, :]
        std = (out[6][:, index])[0, :]
        std_all = out[6]
        mean_action_log_prob = {}  # (out[7][:, index])
        pre_tanh_value = {}  # (out[8][:, index])[0, :]
        return action, action_all, mean, mean_all, log_prob, log_std, mixing_coefficient, entropy, std, std_all, mean_action_log_prob, pre_tanh_value, index, {}

    def get_actions(self, obs_np, goals_np, deterministic=False):
        out = eval_np(self, obs_np, goals_np, deterministic=deterministic)
        # return out[0][:, index]
        return out

    def sample_action(self, mean, std, reparameterize=True, return_log_prob=True):
        tanh_normal = TanhNormal(mean, std)
        if return_log_prob:
            if reparameterize is True:
                action, pre_tanh_value = tanh_normal.rsample(
                    return_pretanh_value=True
                )
            else:
                action, pre_tanh_value = tanh_normal.sample(
                    return_pretanh_value=True
                )
            log_prob = tanh_normal.log_prob(
                action,
                pre_tanh_value=pre_tanh_value
            )
            log_prob = log_prob.sum(dim=2)
        else:
            if reparameterize is True:
                action = tanh_normal.rsample()
            else:
                action = tanh_normal.sample()
        return action, log_prob

    def forward(
            self,
            obs,
            goals,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        mean = mean.reshape(mean.shape[0], self.k, -1)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
            std = std.reshape(std.shape[0], self.k, -1)  # TODO
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None

        h_goals = torch.cat((h.detach(), goals.detach()), 1)
        mixing_coefficient = torch.softmax(self.mixing_coefficient_fc(h_goals), 1)
        # print("mixing_coefficient:", mixing_coefficient)

        if deterministic:
            action = torch.tanh(mean)
            # print("mean:{}".format(mean))
        else:
            tanh_normal = TanhNormal(mean, std)
            # print("tanh_normal", tanh_normal)
            # print("mean:{}, std:{}".format(mean, std))
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=2)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, mixing_coefficient, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )


class MakeDeterministic(nn.Module, Policy):
    def __init__(self, stochastic_policy):
        super().__init__()
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation, goals):
        return self.stochastic_policy.get_action(observation, goals,
                                                 deterministic=True)
