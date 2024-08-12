import numpy as np
import multiprocessing as mp
import gym
import sys

import queue
from offline_training_mec.mec_env.agent.action import Action


class EnvWorker(mp.Process):

	def __init__(self, remote, env_fn, queue_, lock):
		"""

		:param remote: send/recv connection, type of Pipe
		:param env_fn: construct environment function
		:param queue_: global queue instance
		:param lock: Every worker has a lock
		"""
		super(EnvWorker, self).__init__()

		self.remote = remote # Pipe()
		self.env = env_fn # return a function
		self.queue = queue_
		self.lock = lock
		self.task_id = None
		self.done = False

	def empty_step(self):
		"""
		conduct a dummy step
		:return:
		"""
		observation = np.zeros(self.env.observation_space.shape, dtype=np.float32)
		# reward, done = 0.0, True
		reward, done = -np.inf, True
		cost, cost_max = 8.0, 8.0

		return observation, cost, reward, cost_max, done, {}

	def try_reset(self):
		"""

		:return:
		"""
		with self.lock:
			try:
				self.task_id = self.queue.get(True) # block = True
				self.done = (self.task_id is None)
			except queue.Empty:
				self.done = True

		# construct empty state or get state from env.reset()
		# observation = np.zeros(self.env.observation_space, dtype=np.float32) if self.done else self.env.reset()
		observation = None

		return observation

	def run(self):
		"""

		:return:
		"""
		while True:
			command, data = self.remote.recv()

			if command == 'step':
				# print("data:", data)
				state_class_list = data['state_class_list']
				action = data['action']
				step_count = int(data['step_count'])
				ue_num = len(self.env.base_station_set.all_mobile_device_list)
				a_reshaped = action.reshape(ue_num, -1)
				a_reshaped = np.clip(a_reshaped, -0.99, 0.99)
				action_offload_mask = np.zeros((ue_num, int(self.env.action_dim)))
				action_offload_mask = action_offload_mask[:, 0]
				action_class_list = []
				for ue_id, action in enumerate(a_reshaped):
					# partial_offloading = action[0]
					# ue_action = Action([partial_offloading], self.env.global_config)
					ue_action = Action(action, self.env.global_config)
					action_class_list.append(ue_action)

				data_transmission_rate_list = np.ones(ue_num)
				offload_count = 0
				for ue_id in range(ue_num):
					if action_class_list[ue_id].get_whether_offload():
						offload_count += 1

				data_transmission_rate_list = data_transmission_rate_list * offload_count

				# observation, reward, done, info = (self.empty_step() if self.done else self.env.step(state_class_list,
                #                                                                                  action_class_list,
                #                                                                                  step_count))

				# print("self.done:", self.done)
				# next_state_class_list, cost_array, reward_array, cost_array_max, done_list, _ = (self.empty_step() if self.done else self.env.step(
				# 	state_class_list,
				# 	action_class_list,
				# 	step_count))
				next_state_class_list, cost_array, reward_array, cost_array_max, done_list, _ = self.env.step(
						state_class_list,
						action_class_list,
						step_count)
				reward = reward_array[0]
				temp_reward = reward

				next_obs = []
				for mobile_device_id in range(len(self.env.base_station_set.all_mobile_device_list)):
					each_state = self.env.get_state_per_mobile_device(mobile_device_id)
					state_array = each_state.get_normalized_state_array()
					next_obs.append(state_array)
				next_state = np.concatenate(next_obs, -1)
				done = any(done_list)
				terminal = False
				terminal = (step_count + 1 >= self.env.global_config.train_config.step_num)
				done = terminal or done


				# if done and (not self.done):
				# 	observation = self.try_reset()
				# self.remote.send((observation, reward, done, self.task_id, info))
				self.remote.send((next_state, reward, done, self.task_id, cost_array[0], offload_count))

			elif command == 'reset':
				observation = self.try_reset()
				self.env.reset()
				o = []
				state_class_list = []
				for ue_id in range(len(self.env.base_station_set.all_mobile_device_list)):
					ue_state = self.env.get_state_per_mobile_device(ue_id)
					state_class_list.append(ue_state)
					# o.extend(ue_state.get_state_list())
					o.extend(ue_state.get_normalized_state_array())
				observation = np.array(o)
				self.remote.send((observation, state_class_list, self.task_id))
			elif command == 'reset_mectask':
				self.env.create_task_per_step()
				o = []
				state_class_list = []
				for ue_id in range(len(self.env.base_station_set.all_mobile_device_list)):
					ue_state = self.env.get_state_per_mobile_device(ue_id)
					state_class_list.append(ue_state)
					# o.extend(ue_state.get_state_list())
					o.extend(ue_state.get_normalized_state_array())
				observation = np.array(o)
				self.remote.send((observation, state_class_list, self.task_id))
			elif command == 'reset_task':
				env_pram = data
				if self.env.global_config.train_config.is_testbed:
					self.env.global_config.base_station_set_config.data_transmitting_rate = env_pram[0]  # [Mbps]
				else:
					self.env.global_config.base_station_set_config.bandwidth = env_pram[0]
				self.env.global_config.base_station_set_config.base_station_config.base_station_computing_ability = env_pram[1]  # [GHz]
				self.env.global_config.base_station_set_config.task_config.task_data_size_now = env_pram[2]  # [kb]
				self.env.reset()
				# self.env.unwrapped.reset_task(data)
				self.remote.send(True)
			elif command == 'close':
				self.remote.close()
				break
			elif command == 'get_spaces':
				o = []
				for ue_id in range(len(self.env.base_station_set.all_mobile_device_list)):
					ue_state = self.env.get_state_per_mobile_device(ue_id)
					# state_class_list.append(ue_state)
					o.extend(ue_state.get_normalized_state_array())
				o = np.array(o)
				self.env.observation_dim = observation_space = o.shape[0]
				self.env.action_dim = action_space = len(self.env.base_station_set.all_mobile_device_list) * self.env.global_config.agent_config.action_config.dimension
				self.remote.send((observation_space, action_space))
			else:
				raise NotImplementedError()


class SubprocVecEnv(gym.Env):

	def __init__(self, env_factorys, queue_):
		"""

		:param env_factorys: list of [lambda x: def p: envs.make(env_name), return p], len: num_workers
		:param queue:
		"""
		self.lock = mp.Lock()
		# remotes: all recv conn, len: 8, here duplex=True
		# works_remotes: all send conn, len: 8, here duplex=True
		self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_factorys])

		# queue and lock is shared.
		self.workers = [EnvWorker(remote, env_fn, queue_, self.lock)
		                    for (remote, env_fn) in zip(self.work_remotes, env_factorys)]
		# start 8 processes to interact with environments.
		for worker in self.workers:
			worker.daemon = True
			worker.start()
		for remote in self.work_remotes:
			remote.close()

		self.waiting = False # for step_async
		self.closed = False

		# Since the main process need talk to children processes, we need a way to comunicate between these.
		# here we use mp.Pipe() to send/recv data.
		self.remotes[0].send(('get_spaces', None))
		observation_space, action_space = self.remotes[0].recv()
		self.observation_space = observation_space
		self.action_space = action_space

	def step(self, state_class_list, actions, step_count):
		"""
		step synchronously
		:param actions:
		:return:
		"""
		self.step_async(state_class_list, actions, step_count)
		# wait until step state overdue
		return self.step_wait()

	def step_async(self, state_class_lists, actions, step_counts):
		"""
		step asynchronouly
		:param actions:
		:return:
		"""
		# let each sub-process step
		for remote, state_class_list, action, step_count in zip(self.remotes, state_class_lists, actions, step_counts):
			data = {}
			data['state_class_list'] = state_class_list
			data['action'] = action
			data['step_count'] = step_count
			remote.send(('step', data))
		self.waiting = True

	def step_wait(self):
		results = [remote.recv() for remote in self.remotes]
		self.waiting = False
		# observations, rewards, dones, task_ids, infos = zip(*results)
		next_state, rewards, dones, task_ids, cost_array, offload_count = zip(*results)
		return np.stack(next_state), np.stack(rewards), np.stack(dones), task_ids, np.stack(cost_array), np.stack(offload_count)

	def reset(self):
		"""
		reset synchronously
		:return:
		"""
		for remote in self.remotes:
			remote.send(('reset', None))
		results = [remote.recv() for remote in self.remotes]
		observations, state_class_list, task_ids = zip(*results)
		return np.stack(observations), np.stack(state_class_list), task_ids

	def reset_task(self, tasks):
		for remote, task in zip(self.remotes, tasks):
			print("task0-----------------------------------------:", task)
			remote.send(('reset_task', task))
		return np.stack([remote.recv() for remote in self.remotes])

	def create_mectask_per_step(self):
		for remote in self.remotes:
			remote.send(('reset_mectask', None))
		results = [remote.recv() for remote in self.remotes]
		observations, state_class_list, task_ids = zip(*results)
		return np.stack(observations), np.stack(state_class_list), task_ids

	def close(self):
		if self.closed:
			return
		if self.waiting: # cope with step_async()
			for remote in self.remotes:
				remote.recv()
		for remote in self.remotes:
			remote.send(('close', None))
		for worker in self.workers:
			worker.join()
		self.closed = True
