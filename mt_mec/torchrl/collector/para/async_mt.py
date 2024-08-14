import torch
import copy
import numpy as np

from .base import AsyncParallelCollector
import torch.multiprocessing as mp

import mt_mec.torchrl.policies.continuous_policy as policies_continuous_policy
import mt_mec.torchrl.policies.distribution as policies_distribution

from mt_mec.torchrl.env.continuous_wrapper import *

from mt_mec.metaworld_utils.meta_env import generate_single_mt_env

from mt_mec.metaworld_utils.meta_env import get_meta_env

from collections import OrderedDict
from offline_training_mec.mec_env.agent.action import Action

import time
import sys

class AsyncSingleTaskParallelCollector(AsyncParallelCollector):
    def __init__(
            self,
            reset_idx=False,
            **kwargs):
        self.reset_idx = reset_idx
        super().__init__(**kwargs)

    @staticmethod
    def eval_worker_process(
            shared_pf, env_info, shared_que, start_barrier, epochs, reset_idx):

        pf = copy.deepcopy(shared_pf).to(env_info.device)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)

        env_info.env.eval()
        env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch > epochs:
                break
            pf.load_state_dict(shared_pf.state_dict())

            eval_rews = []

            done = False
            success = 0
            for idx in range(env_info.eval_episodes):
                if reset_idx:
                    eval_ob = env_info.env.reset_with_index(idx)
                else:
                    eval_ob = env_info.env.reset()
                rew = 0
                current_success = 0
                while not done:
                    act = pf.eval_act(torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0))
                    eval_ob, r, done, info = env_info.env.step(act)
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()

                    current_success = max(current_success, info["success"])

                eval_rews.append(rew)
                done = False
                success += current_success

            shared_que.put({
                'eval_rewards': eval_rews,
                'success_rate': success / env_info.eval_episodes
            })

    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)

        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)

        self.env_info.env_cls = self.env_cls
        self.env_info.env_args = self.env_args

        for i in range(self.worker_nums):
            self.env_info.env_rank = i
            p = mp.Process(
                target=self.__class__.train_worker_process,
                args=(self.__class__, self.shared_funcs,
                      self.env_info, self.replay_buffer,
                      self.shared_que, self.start_barrier,
                      self.train_epochs))
            p.start()
            self.workers.append(p)

        for i in range(self.eval_worker_nums):
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf"],
                      self.env_info, self.eval_shared_que, self.eval_start_barrier,
                      self.eval_epochs, self.reset_idx))
            eval_p.start()
            self.eval_workers.append(eval_p)

    def eval_one_epoch(self):
        # self.eval_start_barrier.wait()
        eval_rews = []
        mean_success_rate = 0
        self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            eval_rews += worker_rst["eval_rewards"]
            mean_success_rate += worker_rst["success_rate"]

        return {
            'eval_rewards': eval_rews,
            'mean_success_rate': mean_success_rate / self.eval_worker_nums
        }


class AsyncMultiTaskParallelCollectorUniform(AsyncSingleTaskParallelCollector):

    def __init__(self, progress_alpha=0.1, logger=None, **kwargs):
        super().__init__(**kwargs)
        self.tasks = list(self.env_cls.keys())
        self.tasks_mapping = {}
        for idx, task_name in enumerate(self.tasks):
            self.tasks_mapping[task_name] = idx
        self.tasks_progress = [0 for _ in range(len(self.tasks))]
        self.progress_alpha = progress_alpha
        self.logger = logger
        self.tf_writer = self.logger.tf_writer
        self.load_weight_frequency_num = 1

    @classmethod
    def take_actions(cls, funcs, env_info, ob_info, replay_buffer, current_step):

        pf = funcs["pf"]

        o = []
        state_class_list = []
        goals_list = []
        o_human = []
        for ue_id in range(env_info.env.ue_num):
            ue_state = env_info.env.get_state_per_mobile_device(ue_id)
            state_class_list.append(ue_state)
            o_human.extend(ue_state.get_state_list())
            o.extend(ue_state.get_normalized_state_array())
            goals_list.extend(ue_state.goals)
        o = np.array(o)
        ob_info["ob"] = o
        ob_info["ob_class"] = state_class_list

        ob = ob_info["ob"]
        task_idx = env_info.env_rank
        idx_flag = isinstance(pf, policies_continuous_policy.MultiHeadGuassianContPolicy)

        embedding_flag = isinstance(pf, policies_continuous_policy.EmbeddingGuassianContPolicyBase)

        pf.eval()

        action_time = time.time()
        with torch.no_grad():
            if idx_flag:
                idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                if embedding_flag:
                    embedding_input = torch.zeros(env_info.num_tasks)
                    embedding_input[env_info.env_rank] = 1
                    embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                    out = pf.explore(torch.Tensor(ob).to(env_info.device).unsqueeze(0), embedding_input,
                                     [task_idx])
                else:
                    out = pf.explore(torch.Tensor(ob).to(env_info.device).unsqueeze(0),
                                     idx_input)
                act = out["action"]
                # act = act[0]
            else:
                if embedding_flag:
                    embedding_input = torch.zeros(env_info.num_tasks)
                    embedding_input[env_info.env_rank] = 1
                    embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                    out = pf.explore(torch.Tensor(ob).to(env_info.device).unsqueeze(0), embedding_input,
                                     return_weights=True)
                else:
                    out = pf.explore(torch.Tensor(ob).to(env_info.device).unsqueeze(0))
                act = out["action"]

        act = act.detach().cpu().numpy()

        if type(act) is not int:
            if np.isnan(act).any():
                print("NaN detected. BOOM")
                exit()

        path_length = current_step
        current_success = 0
        ue_state_list = ob_info["ob_class"]

        ue_queue_time_now = np.zeros(env_info.env.ue_num)
        offloading_count_list = []
        reward_list = []
        cost_avg_list = []
        cost_baseline_avg_list = []
        a_reshaped = act.reshape(env_info.env.ue_num, -1)
        a_reshaped = np.clip(a_reshaped, -0.99, 0.99)
        # print("a:", a_reshaped)
        action_class_list = []
        for ue_id, action in enumerate(a_reshaped):
            ue_action = Action(action, env_info.env.global_config)
            action_class_list.append(ue_action)

        offload_count = 0
        for ue_id in range(env_info.env.ue_num):
            if action_class_list[ue_id].get_whether_offload():
                offload_count += 1

        next_state_class_list, cost_array, reward_array, cost_array_max, done_list, _ = env_info.env.step(state_class_list,
                                                                                                 action_class_list,
                                                                                                 path_length)

        reward = reward_array[0]
        temp_reward = reward

        next_obs = []
        next_obs_human = []
        next_goal_list = []
        next_ue_state_list = []
        for mobile_device_id in range(len(env_info.env.base_station_set.all_mobile_device_list)):
            each_state = env_info.env.get_state_per_mobile_device(mobile_device_id)
            next_ue_state_list.append(each_state)
            state_array = each_state.get_normalized_state_array()
            next_goal_list.extend(each_state.goals)
            next_obs.append(state_array)
            next_obs_human.extend(each_state.get_state_list())
        next_state = np.concatenate(next_obs, -1)
        next_ob = next_state
        done = any(done_list)
        terminal = False
        assert current_step == path_length
        terminal = (path_length + 1 >= env_info.max_episode_frames)
        done = terminal or done

        reward_list.append(reward)
        cost_avg_list.append(cost_array[0])
        cost_baseline_avg_list.append(cost_array_max[0])

        offload_count_avg = offload_count
        assert len(reward_list) == 1
        sample_dict = {
            "obs": ob,
            "next_obs": next_ob,
            "acts": act,
            # "general_weights": general_weights_numpy,
            # "last_weights": last_weights,
            "task_idxs": [env_info.env_rank],
            "rewards": reward_list,
            "terminals": [done]
        }
        info = {}
        if embedding_flag:
            sample_dict["embedding_inputs"] = embedding_input.cpu().numpy()

        next_ob_class = next_ue_state_list
        if done or env_info.current_step >= env_info.max_episode_frames:
            env_info.finish_episode()
            env_info.start_episode()  # reset current_step

        replay_buffer.add_sample(sample_dict, env_info.env_rank)

        return next_ob, next_ob_class, done, reward_list[0], cost_array[0], offload_count_avg, info

    @staticmethod
    def train_worker_process(cls, shared_funcs, env_info,
                             replay_buffer, shared_que,
                             start_barrier, epochs, start_epoch, task_name, shared_dict, load_weight_frequency_num):

        replay_buffer.rebuild_from_tag()
        local_funcs = copy.deepcopy(shared_funcs)
        for key in local_funcs:
            local_funcs[key].to(env_info.device)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)

        norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        if norm_obs_flag:
            shared_dict[task_name] = {
                "obs_mean": env_info.env._obs_mean,
                "obs_var": env_info.env._obs_var
            }

        print("env_info.env:", env_info.env)
        env_info.env.reset()
        o = []
        state_class_list = []
        goals_list = []
        o_human = []
        for ue_id in range(env_info.env.ue_num):
            ue_state = env_info.env.get_state_per_mobile_device(ue_id)
            state_class_list.append(ue_state)
            o_human.extend(ue_state.get_state_list())
            o.extend(ue_state.get_normalized_state_array())
            goals_list.extend(ue_state.goals)
        o = np.array(o)

        c_ob = {
            "ob": o,
            "ob_class": state_class_list
        }
        train_rew = 0
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'train_rewards': None,
                    'train_epoch_reward': None
                })
                continue
            if current_epoch > epochs:
                break

            if current_epoch % load_weight_frequency_num == 0:
                for key in shared_funcs:
                    local_funcs[key].load_state_dict(shared_funcs[key].state_dict())

            train_rews = []
            train_cost = []
            train_epoch_reward = 0

            cost_list = []
            reward_list = []
            offload_count_avg_list = []
            current_step = 0
            other_time = time.time()
            for current_step in range(env_info.epoch_frames):
                env_info.env.create_task_per_step()

                next_ob, next_ob_class, done, reward, cost, offload_count_avg, _ = cls.take_actions(local_funcs,
                                                                                                    env_info, c_ob,
                                                                                                    replay_buffer,
                                                                                                    current_step)

                reward_list.append(reward)
                cost_list.append(cost)
                offload_count_avg_list.append(offload_count_avg)
                c_ob["ob"] = next_ob
                c_ob["ob_class"] = next_ob_class

                # train_rew += reward
                train_epoch_reward += reward
                if done:
                    train_rews.append(reward_list[-1])
                    train_cost.append(np.mean(cost_list))
                    break

            if norm_obs_flag:
                shared_dict[task_name] = {
                    "obs_mean": env_info.env._obs_mean,
                    "obs_var": env_info.env._obs_var
                }

            shared_que.put({
                'train_rewards': train_rews,
                'train_cost': train_cost,
                'train_path_length': [current_step + 1],
                'offload_count_avg_epoch': [np.mean(np.array(offload_count_avg_list))],
                'task_name': task_name,
                'train_epoch_reward': train_epoch_reward
            })

    @staticmethod
    def eval_worker_process(shared_pf,
                            env_info, shared_que, start_barrier, epochs, start_epoch, task_name, shared_dict, load_weight_frequency_num):

        eval_frequency_num = 1000
        pf = copy.deepcopy(shared_pf).to(env_info.device)
        idx_flag = isinstance(pf, policies_continuous_policy.MultiHeadGuassianContPolicy)
        embedding_flag = isinstance(pf, policies_continuous_policy.EmbeddingGuassianContPolicyBase)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)

        norm_obs_flag = env_info.env_args["env_params"]["obs_norm"]

        # env_info.env.eval()
        env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            # start_barrier_wait_time = time.time()
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch or not (current_epoch % eval_frequency_num == 0):
                shared_que.put({
                    'eval_rewards': None,
                    'eval_costs': None,
                    'eval_path_length': None,
                    'offload_count_avg_epoch': None,
                    'success_rate': None,
                    'task_name': task_name
                })
                continue
            if current_epoch > epochs:
                break
            pf.load_state_dict(shared_pf.state_dict())
            pf.eval()

            if norm_obs_flag:
                env_info.env._obs_mean = shared_dict[task_name]["obs_mean"]
                env_info.env._obs_var = shared_dict[task_name]["obs_var"]

            eval_rews = []
            cost_avg_list = []
            path_length = 0
            offload_count_avg_list = []

            done = False
            success = 0

            for idx in range(env_info.eval_episodes):

                done = False

                env_info.env.reset()
                o = []
                state_class_list = []
                goals_list = []
                o_human = []
                for ue_id in range(env_info.env.ue_num):
                    ue_state = env_info.env.get_state_per_mobile_device(ue_id)
                    state_class_list.append(ue_state)
                    o_human.extend(ue_state.get_state_list())
                    o.extend(ue_state.get_normalized_state_array())
                    goals_list.extend(ue_state.goals)
                o = np.array(o)

                eval_ob, ue_state_list = o, state_class_list
                rew = 0

                task_idx = env_info.env_rank
                current_success = 0

                ue_queue_time_now = np.zeros(env_info.env.ue_num)
                offloading_count_list = []
                eval_rews = []
                cost_avg_list = []
                cost_baseline_avg_list = []
                offload_count_avg_list = []

                path_length = 0
                while not done:
                    env_info.env.create_task_per_step()
                    o = []
                    state_class_list = []
                    goals_list = []
                    o_human = []
                    for ue_id in range(env_info.env.ue_num):
                        ue_state = env_info.env.get_state_per_mobile_device(ue_id)
                        state_class_list.append(ue_state)
                        o_human.extend(ue_state.get_state_list())
                        o.extend(ue_state.get_normalized_state_array())
                        goals_list.extend(ue_state.goals)
                    o = np.array(o)

                    eval_ob, ue_state_list = o, state_class_list

                    if idx_flag:
                        idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                        if embedding_flag:
                            embedding_input = torch.zeros(env_info.num_tasks)
                            embedding_input[env_info.env_rank] = 1
                            embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                            act = pf.eval_act(torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0),
                                              embedding_input, [task_idx])
                        else:
                            act = pf.eval_act(torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0), idx_input)
                    else:
                        if embedding_flag:
                            embedding_input = torch.zeros(env_info.num_tasks)
                            embedding_input[env_info.env_rank] = 1
                            embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                            act = pf.eval_act(torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0), embedding_input)
                        else:
                            act = pf.eval_act(torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0))

                    a_reshaped = act.reshape(env_info.env.ue_num, -1)
                    a_reshaped = np.clip(a_reshaped, -0.99, 0.99)
                    # print("a:", a_reshaped)
                    action_class_list = []
                    for ue_id, action in enumerate(a_reshaped):
                        ue_action = Action(action, env_info.env.global_config)
                        action_class_list.append(ue_action)

                    offload_count = 0
                    for ue_id in range(env_info.env.ue_num):
                        if action_class_list[ue_id].get_whether_offload():
                            offload_count += 1

                    next_state_class_list, cost_array, reward_array, cost_array_max, done_list, _ = env_info.env.step(
                        state_class_list,
                        action_class_list,
                        path_length)

                    reward = reward_array[0]
                    temp_reward = reward

                    next_obs = []
                    next_obs_human = []
                    next_goal_list = []
                    next_ue_state_list = []
                    for mobile_device_id in range(len(env_info.env.base_station_set.all_mobile_device_list)):
                        each_state = env_info.env.get_state_per_mobile_device(mobile_device_id)
                        next_ue_state_list.append(each_state)
                        state_array = each_state.get_normalized_state_array()
                        next_goal_list.extend(each_state.goals)
                        next_obs.append(state_array)
                        next_obs_human.extend(each_state.get_state_list())
                    next_state = np.concatenate(next_obs, -1)
                    next_ob = next_state
                    done = any(done_list)
                    terminal = False
                    terminal = (path_length + 1 >= env_info.max_episode_frames)
                    done = terminal or done

                    eval_rews.append(reward)
                    cost_avg_list.append(cost_array[0])
                    cost_baseline_avg_list.append(cost_array_max[0])

                success += current_success

                current_step = path_length

            que_put_eval = time.time()
            shared_que.put({
                'eval_rewards': [np.mean(np.array(eval_rews))],
                'eval_costs': [np.mean(np.array(cost_avg_list))],
                'eval_path_length': [path_length],
                'offload_count_avg_epoch': [np.mean(np.array(offload_count_avg_list))],
                'success_rate': success / env_info.eval_episodes,
                'task_name': task_name
            })

    def start_worker(self):
        self.workers = []
        self.shared_que = self.manager.Queue(self.worker_nums)
        self.start_barrier = mp.Barrier(self.worker_nums)

        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)

        self.shared_dict = self.manager.dict()
        self.load_weight_frequency_num = 1

        print("self.worker_nums:", self.worker_nums)
        print("self.env.num_tasks:", self.env.num_tasks)
        assert self.worker_nums == self.env.num_tasks
        # task_cls, task_args, env_params
        self.env_info.env = None
        self.env_info.num_tasks = self.env.num_tasks
        self.env_info.env_cls = generate_single_mt_env
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": self.env.num_tasks,
            "max_obs_dim": np.prod(self.env.mec_observation_space),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2]
        }

        tasks = list(self.env_cls.keys())

        for i, task in enumerate(tasks):
            env_cls = self.env_cls[task]

            self.env_info.env_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

            if "start_epoch" in self.env_info.env_args["task_args"]:
                start_epoch = self.env_info.env_args["task_args"]["start_epoch"]
                del self.env_info.env_args["task_args"]["start_epoch"]
            else:
                start_epoch = 0

            self.env_info.env_args["env_rank"] = i
            p = mp.Process(  # 每个任务并行多进程的调用train_worker_process
                target=self.__class__.train_worker_process,
                args=(self.__class__, self.shared_funcs,
                      self.env_info, self.replay_buffer,
                      self.shared_que, self.start_barrier,
                      self.train_epochs, start_epoch, task, self.shared_dict, self.load_weight_frequency_num))
            p.start()
            self.workers.append(p)
            # i += 1

        assert self.eval_worker_nums == self.env.num_tasks

        self.env_info.env = None
        self.env_info.num_tasks = self.env.num_tasks
        self.env_info.env_cls = generate_single_mt_env
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": self.env.num_tasks,
            "max_obs_dim": np.prod(self.env.mec_observation_space),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2]
        }

        for i, task in enumerate(tasks):
            env_cls = self.env_cls[task]

            self.env_info.env_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

            start_epoch = 0
            if "start_epoch" in self.env_info.env_args["task_args"]:
                # start_epoch = self.env_info.env_args["task_args"]["start_epoch"]
                del self.env_info.env_args["task_args"]["start_epoch"]
            # else:
            # start_epoch = 0

            self.env_info.env_args["env_rank"] = i
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf"],
                      self.env_info, self.eval_shared_que, self.eval_start_barrier,
                      self.eval_epochs, start_epoch, task, self.shared_dict, self.load_weight_frequency_num))
            eval_p.start()
            self.eval_workers.append(eval_p)

    def eval_one_epoch(self, epoch):

        eval_rews = []
        mean_success_rate = 0

        if epoch % self.load_weight_frequency_num == 0:
            self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())

        tasks_result = []

        active_task_counts = 0
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            if worker_rst["eval_rewards"] is not None:
                active_task_counts += 1
                # worker_rst["eval_rewards"] = worker_rst["eval_rewards"].tolist()
                eval_rews += worker_rst["eval_rewards"]
                mean_success_rate += worker_rst["success_rate"]
                tasks_result.append(
                    (worker_rst["task_name"], worker_rst["success_rate"], np.mean(worker_rst["eval_rewards"]),
                     np.mean(worker_rst["eval_costs"]), np.mean(worker_rst["eval_path_length"]),
                     np.mean(worker_rst["offload_count_avg_epoch"])))

        tasks_result.sort()

        dic = OrderedDict()
        for task_name, success_rate, eval_rewards, eval_costs, eval_path_length, offload_count_avg_epoch in tasks_result:
            dic[task_name + "_success_rate"] = success_rate
            dic[task_name + "_eval_rewards"] = eval_rewards
            dic[task_name + "_eval_costs"] = eval_costs
            dic[task_name + "_eval_path_length"] = eval_path_length
            dic[task_name + "_eval_offload_count_avg_epoch"] = offload_count_avg_epoch
            self.tasks_progress[self.tasks_mapping[task_name]] *= \
                (1 - self.progress_alpha)
            self.tasks_progress[self.tasks_mapping[task_name]] += \
                self.progress_alpha * success_rate

        dic['eval_rewards'] = eval_rews
        if active_task_counts == 0:
            active_task_counts = 1
        dic['mean_success_rate'] = mean_success_rate / active_task_counts

        return dic

    def train_one_epoch(self, epoch):
        train_rews = []
        train_costs = []
        train_path_lengths = []
        offload_counts_avg_epoch = []
        train_epoch_reward = 0
        task_result = []

        load_state_dict_time = time.time()
        if epoch % self.load_weight_frequency_num == 0:
            for key in self.shared_funcs:
                self.shared_funcs[key].load_state_dict(self.funcs[key].state_dict())

        active_worker_nums = 0
        for _ in range(self.worker_nums):
            worker_rst = self.shared_que.get()
            if worker_rst["train_rewards"] is not None:
                train_rews += worker_rst["train_rewards"]
                train_costs += worker_rst["train_cost"]
                train_path_lengths += worker_rst["train_path_length"]
                offload_counts_avg_epoch += worker_rst["offload_count_avg_epoch"]
                task_result.append(
                    (worker_rst["task_name"], np.mean(worker_rst["train_rewards"]), np.mean(worker_rst["train_cost"]),
                     np.mean(worker_rst["train_path_length"]), np.mean(worker_rst["offload_count_avg_epoch"]))
                )
                train_epoch_reward += worker_rst["train_epoch_reward"]
                active_worker_nums += 1
        self.active_worker_nums = active_worker_nums

        dic = OrderedDict()
        for task_name, train_rewards, train_cost, train_path_length, offload_count_avg_epoch in task_result:
            dic[task_name + "_train_rewards"] = train_rewards
            dic[task_name + "_train_cost"] = train_cost
            dic[task_name + "_train_path_length"] = train_path_length
            dic[task_name + "_train_offload_count_avg_epoch"] = offload_count_avg_epoch

        dic['train_rewards'] = train_rews
        dic['train_costs'] = train_costs
        dic['train_path_lengths'] = train_path_lengths
        dic['train_offload_counts_avg_epoch'] = offload_counts_avg_epoch
        dic['train_epoch_reward'] = train_epoch_reward

        return dic


class AsyncMultiTaskParallelCollectorUniformImitation(AsyncSingleTaskParallelCollector):

    def __init__(self, progress_alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.tasks = list(self.env_cls.keys())
        self.tasks_mapping = {}
        for idx, task_name in enumerate(self.tasks):
            self.tasks_mapping[task_name] = idx
        self.tasks_progress = [0 for _ in range(len(self.tasks))]
        self.progress_alpha = progress_alpha

    @staticmethod
    def eval_worker_process(shared_pf,
                            env_info, shared_que, start_barrier, epochs, start_epoch, task_name):

        pf = copy.deepcopy(shared_pf).to(env_info.device)
        idx_flag = isinstance(pf, policies_continuous_policy.MultiHeadGuassianContPolicy)
        embedding_flag = isinstance(pf, policies_continuous_policy.EmbeddingGuassianContPolicyBase) or isinstance(pf,
                                                                                                                  policies_continuous_policy.EmbeddingDetContPolicyBase)

        # Rebuild Env
        env_info.env = env_info.env_cls(**env_info.env_args)

        env_info.env.eval()
        env_info.env._reward_scale = 1
        current_epoch = 0
        while True:
            start_barrier.wait()
            current_epoch += 1
            if current_epoch < start_epoch:
                shared_que.put({
                    'eval_rewards': None,
                    'success_rate': None,
                    'task_name': task_name
                })
                continue
            if current_epoch > epochs:
                break
            pf.load_state_dict(shared_pf.state_dict())
            pf.eval()

            eval_rews = []

            done = False
            success = 0
            for idx in range(env_info.eval_episodes):

                eval_ob = env_info.env.reset()
                rew = 0

                task_idx = env_info.env_rank
                current_success = 0
                while not done:

                    if idx_flag:
                        idx_input = torch.Tensor([[task_idx]]).to(env_info.device).long()
                        if embedding_flag:
                            embedding_input = torch.zeros(env_info.num_tasks)
                            embedding_input[env_info.env_rank] = 1
                            embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                            act = pf.eval_act(torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0),
                                              embedding_input, [task_idx])
                        else:
                            act = pf.eval_act(torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0), idx_input)
                    else:
                        if embedding_flag:
                            embedding_input = torch.zeros(env_info.num_tasks)
                            embedding_input[env_info.env_rank] = 1
                            embedding_input = embedding_input.unsqueeze(0).to(env_info.device)
                            act = pf.eval_act(torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0), embedding_input)
                        else:
                            act = pf.eval_act(torch.Tensor(eval_ob).to(env_info.device).unsqueeze(0))

                    eval_ob, r, done, info = env_info.env.step(act)
                    rew += r
                    if env_info.eval_render:
                        env_info.env.render()
                    current_success = max(current_success, info["success"])

                eval_rews.append(rew)
                done = False
                success += current_success

            shared_que.put({
                'eval_rewards': eval_rews,
                'success_rate': success / env_info.eval_episodes,
                'task_name': task_name
            })

    def start_worker(self):
        self.eval_workers = []
        self.eval_shared_que = self.manager.Queue(self.eval_worker_nums)
        self.eval_start_barrier = mp.Barrier(self.eval_worker_nums)

        # task_cls, task_args, env_params
        tasks = list(self.env_cls.keys())

        assert self.worker_nums == 0
        assert self.eval_worker_nums == self.env.num_tasks

        self.env_info.env = None
        self.env_info.num_tasks = self.env.num_tasks
        self.env_info.env_cls = generate_single_mt_env
        single_mt_env_args = {
            "task_cls": None,
            "task_args": None,
            "env_rank": 0,
            "num_tasks": self.env.num_tasks,
            "max_obs_dim": np.prod(self.env.observation_space.shape),
            "env_params": self.env_args[0],
            "meta_env_params": self.env_args[2]
        }

        for i, task in enumerate(tasks):
            env_cls = self.env_cls[task]

            self.env_info.env_rank = i

            self.env_info.env_args = single_mt_env_args
            self.env_info.env_args["task_cls"] = env_cls
            self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

            start_epoch = 0
            if "start_epoch" in self.env_info.env_args["task_args"]:
                # start_epoch = self.env_info.env_args["task_args"]["start_epoch"]
                del self.env_info.env_args["task_args"]["start_epoch"]
            # else:
            # start_epoch = 0

            self.env_info.env_args["env_rank"] = i
            eval_p = mp.Process(
                target=self.__class__.eval_worker_process,
                args=(self.shared_funcs["pf"],
                      self.env_info, self.eval_shared_que, self.eval_start_barrier,
                      self.eval_epochs, start_epoch, task))
            eval_p.start()
            self.eval_workers.append(eval_p)

    def eval_one_epoch(self):

        eval_rews = []
        mean_success_rate = 0
        self.shared_funcs["pf"].load_state_dict(self.funcs["pf"].state_dict())

        tasks_result = []

        active_task_counts = 0
        for _ in range(self.eval_worker_nums):
            worker_rst = self.eval_shared_que.get()
            if worker_rst["eval_rewards"] is not None:
                active_task_counts += 1
                eval_rews += worker_rst["eval_rewards"]
                mean_success_rate += worker_rst["success_rate"]
                tasks_result.append(
                    (worker_rst["task_name"], worker_rst["success_rate"], np.mean(worker_rst["eval_rewards"])))

        tasks_result.sort()

        dic = OrderedDict()
        for task_name, success_rate, eval_rewards in tasks_result:
            dic[task_name + "_success_rate"] = success_rate
            dic[task_name + "_eval_rewards"] = eval_rewards
            self.tasks_progress[self.tasks_mapping[task_name]] *= \
                (1 - self.progress_alpha)
            self.tasks_progress[self.tasks_mapping[task_name]] += \
                self.progress_alpha * success_rate

        dic['eval_rewards'] = eval_rews
        dic['mean_success_rate'] = mean_success_rate / active_task_counts

        return dic
