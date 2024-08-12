import numpy as np
from offline_training_mec.mec_env.agent.action import Action


def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        num_epoch=0,
        is_expl=True,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    goals_all = []
    actions = []
    actions_all = []
    means = []
    means_all = []
    log_stds = []
    log_probs = []
    mixing_coefficients = []
    entropys = []
    stds = []
    stds_all = []
    mean_action_log_probs = []
    pre_tanh_values = []
    rewards = []
    terminals = []
    indexs = []
    agent_infos = []
    env_infos = []
    cost_avg_list = []
    data_transmission_rate_all = []
    if env.global_config.train_config.algorithm == 'PMOE':
        env.reset_for_pmoe(num_epoch, is_expl)
    else:
        env.reset()
    o = []
    state_class_list = []
    goals_list = []
    ue_queue_time_now = np.zeros(env.ue_num)

    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    episode_reward = 0
    episode_cost_list = []
    episode_cost_max_list = []
    eposide_reward_list = []
    while path_length < max_path_length:
        env.create_task_per_step()
        o = []
        state_class_list = []
        goals_list = []
        o_human = []
        for ue_id in range(env.ue_num):
            ue_state = env.get_state_per_mobile_device(ue_id)
            state_class_list.append(ue_state)
            o_human.extend(ue_state.get_state_list())
            o.extend(ue_state.get_normalized_state_array())
            goals_list.extend(ue_state.goals)
        o = np.array(o)
        g, a, a_all, mean, mean_all, log_std, log_prob, mixing_coefficient, entropy, std, std_all, mean_action_log_prob, pre_tanh_value, index, agent_info = \
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        if env.global_config.train_config.algorithm == 'PMOE':
            g = np.array(goals_list)
            a, a_all, mean, mean_all, log_std, locg_prob, mixing_coefficient, entropy, std, std_all, mean_action_log_prob, pre_tanh_value, index, agent_info = agent.get_action(
                o, g)
        else:
            a, agent_info = agent.get_action(o)
        a_reshaped = a.reshape(env.ue_num, -1)
        a_reshaped = np.clip(a_reshaped, -0.99, 0.99)
        # print("a:", a_reshaped)
        action_offload_mask = np.zeros((env.ue_num, int(env.action_dim)))
        action_offload_mask = action_offload_mask[:, 0]
        action_class_list = []
        for ue_id, action in enumerate(a_reshaped):
            ue_action = Action(action, env.global_config)
            action_class_list.append(ue_action)

        # init some list to save data before run one step
        ue_reward_list = []
        ue_next_state_list = []
        next_goal_list = []
        local_cost_baseline_list = []
        # ue_done_list = []
        ue_done_all = False
        ue_cost_list = []
        data_transmission_rate_list = np.ones(env.ue_num)
        offload_count = 0
        for ue_id in range(env.ue_num):
            if action_class_list[ue_id].get_whether_offload():
                offload_count += 1

        # print("list:", path_length)
        data_transmission_rate_list = data_transmission_rate_list * offload_count
        next_state_class_list, cost_array, reward_array, cost_array_max, done_list, _ = env.step(state_class_list,
                                                                                                 action_class_list,
                                                                                                 path_length)

        reward = reward_array[0]
        temp_reward = reward

        next_obs = []
        next_obs_human = []
        for mobile_device_id in range(len(env.base_station_set.all_mobile_device_list)):
            each_state = env.get_state_per_mobile_device(mobile_device_id)
            state_array = each_state.get_normalized_state_array()
            next_goal_list.extend(each_state.goals)
            next_obs.append(state_array)
            next_obs_human.extend(each_state.get_state_list())
        next_state = np.concatenate(next_obs, -1)
        done = any(done_list)
        terminal = False
        terminal = (path_length + 1 >= max_path_length)
        done = terminal or done

        episode_reward += reward
        episode_cost_list.append(cost_array[0])
        episode_cost_max_list.append(cost_array_max[0])
        eposide_reward_list.append(reward)

        next_o, next_g, r, d, env_info = next_state, np.array(next_goal_list), reward, done, {}
        observations.append(o)
        # cost_avg_list.append(cost_avg)
        data_transmission_rate_all.append(data_transmission_rate_list)
        if env.global_config.train_config.algorithm == 'PMOE':
            goals_all.append(g)
            actions_all.append(a_all)
            means.append(mean)
            means_all.append(mean_all)
            log_stds.append(log_std)
            log_probs.append(log_prob)
            mixing_coefficients.append(mixing_coefficient)
            entropys.append(entropy)
            stds.append(std)
            stds_all.append(std_all)
            mean_action_log_probs.append(mean_action_log_prob)
            pre_tanh_values.append(pre_tanh_value)
            indexs.append(index)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)

        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if path_length == max_path_length:
            d = True
        if d:
            break
        # else:
        #     env.update_the_hexagon_network()
        o = next_o
        g = next_g
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    actions_all = np.array(actions_all)
    if len(actions_all.shape) == 1:
        actions_all = np.expand_dims(actions_all, 1)
    observations = np.array(observations)
    goals_all = np.array(goals_all)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])

    mixing_coefficients = np.array(mixing_coefficients)
    if len(mixing_coefficients.shape) == 1:
        mixing_coefficients = np.expand_dims(mixing_coefficients, 1)
    if len(goals_all.shape) == 1:
        goals_all = np.expand_dims(goals_all, 1)
    # cost_avg_list = np.array(cost_avg_list)
    # if len(cost_avg_list.shape) == 1:
    #     cost_avg_list = np.expand_dims(cost_avg_list, 1)
    episode_cost_list = np.array(episode_cost_list)
    if len(episode_cost_list.shape) == 1:
        episode_cost_list = np.expand_dims(episode_cost_list, 1)
    data_transmission_rate_all = np.array(data_transmission_rate_all)
    if len(data_transmission_rate_all.shape) == 1:
        data_transmission_rate_all = np.expand_dims(data_transmission_rate_all, 1)
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    means = np.array(means)
    if len(means.shape) == 1:
        means = np.expand_dims(means, 1)
    means_all = np.array(means_all)
    if len(means_all.shape) == 1:
        means_all = np.expand_dims(means_all, 1)
    
    stds = np.array(stds)
    if len(stds.shape) == 1:
        stds = np.expand_dims(stds, 1)
    stds_all = np.array(stds_all)
    if len(stds_all.shape) == 1:
        stds_all = np.expand_dims(stds_all, 1)
    indexs = np.array(indexs)
    if len(indexs.shape) == 1:
        indexs = np.expand_dims(indexs, 1)
    return dict(
        observations=observations,
        goals_all=goals_all,
        actions=actions,
        actions_all=actions_all,
        means=means,
        means_all=means_all,
        log_stds=log_stds,
        log_probs=log_probs,
        mixing_coefficients=mixing_coefficients,
        entropys=entropys,
        stds=stds,
        stds_all=stds_all,
        mean_action_log_probs=mean_action_log_probs,
        pre_tanh_values=pre_tanh_values,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        indexs=indexs,
        agent_infos=agent_infos,
        env_infos=env_infos,
        path_length=path_length,
        cost_avg_list=episode_cost_list,
        data_transmission_rate_all=data_transmission_rate_all,
    )
