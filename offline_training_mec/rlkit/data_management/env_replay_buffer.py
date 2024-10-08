import numpy as np
from gym.spaces import Discrete

from offline_training_mec.rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_dim
        self._action_space = env.action_dim
        self._goals_space = env.goals_dim

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=self._ob_space,
            action_dim=self._action_space,
            goals_dim=self._goals_space,
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, goal, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            goal=goal,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )
