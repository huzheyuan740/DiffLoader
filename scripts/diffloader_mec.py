import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import diffuser.utils as utils
import numpy as np
from config import GlobalConfig
import shutil
import torch
import random
from transformers import BertTokenizer, BertModel


# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'test-'
    config: str = 'config_diffusion.locomotion'


def get_prompt_file(global_config, task_list):
    prompt_npy_path = global_config.train_config.save_prompt_npy_path
    buffer_npy_path = global_config.train_config.save_buffer_npy_path
    seed = global_config.train_config.seed
    os.makedirs(prompt_npy_path, exist_ok=True)
    start_episode = 1000
    end_episode = 10000
    for task_item in task_list:
        chose_episode = np.random.choice(np.arange(start_episode, end_episode), size=10, replace=False)  # 选择10个episode的数据作为prompt
        print("task_item:", task_item)
        for episode_item in chose_episode:
            print("episode_item:", episode_item)
            current_episode = episode_item
            src_dir = os.path.join(buffer_npy_path, 'seed_' + str(seed), task_item, str(episode_item) + '.npy')
            buffer_npy_load = np.load(src_dir, allow_pickle=True)
            path_length = buffer_npy_load.tolist()['terminals'].shape[0]
            while path_length < global_config.train_config.prompt_step:
                current_episode = np.random.choice(np.arange(start_episode, end_episode), size=1, replace=False)
                src_dir = os.path.join(buffer_npy_path, 'seed_' + str(seed), task_item,
                                       str(current_episode[0]) + '.npy')
                buffer_npy_load = np.load(src_dir, allow_pickle=True)
                path_length = buffer_npy_load.tolist()['terminals'].shape[0]

            dest_dir = os.path.join(prompt_npy_path, 'seed_' + str(seed), task_item, str(episode_item) + '.npy')
            os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
            # os.chmod(os.path.dirname(dest_dir), 0o754)
            command = f'cp {src_dir} {dest_dir}'
            os.system(command)
            # shutil.copy(src_dir, os.path.dirname(dest_dir))


def language_embedding(language_prompt_list):
    language_prompt_value_list = []
    max_len = 128
    for language_prompt in language_prompt_list:
        # print("language_prompt:", language_prompt)
        # print("len:", len(language_prompt.split()))
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")
        encoded_input = tokenizer(language_prompt, padding=True, truncation=True, max_length=max_len,  return_tensors='pt')

        # print("Token IDs:", encoded_input['input_ids'].shape)
        # tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
        # print("Tokens:", tokens)

        output = model(**encoded_input)
        language_prompt_value = output.last_hidden_state[:, 0, :].tolist()
        language_prompt_value_list.append(language_prompt_value)
        # print("language:", np.mean(np.array(language_prompt_value)))
    language_prompt_value_list = np.array(language_prompt_value_list)
    return language_prompt_value_list


def get_prompt_trajectories(global_config, task_list, test_env_list):
    prompt_npy_path = global_config.train_config.save_prompt_npy_path
    seed = global_config.train_config.seed
    trajectories_list = []
    language_prompt_list = []
    query_str = 'env_'
    for task_item in task_list:
        print("task_item:", task_item)
        index = task_item.find(query_str)
        env_par = task_item[index + len(query_str):]
        env_slice = env_par.split('_')
        language_prompt = f'Current MEC system has the bandwidth of {env_slice[0]} MHz, the computational capability is {env_slice[1]} GHz, and the task size is about {env_slice[2]} kb.'
        # language_prompt = language_prompt.split()
        language_prompt_list.append(language_prompt)
        each_task_list = []
        prompt_file_list = os.listdir(os.path.join(prompt_npy_path, 'seed_' + str(seed), task_item))
        for prompt_file in prompt_file_list:
            prompt_file_dir = os.path.join(prompt_npy_path, 'seed_' + str(seed), task_item, prompt_file)
            prompt_npy_load = np.load(prompt_file_dir, allow_pickle=True)
            prompt_npy_load = prompt_npy_load.tolist()
            prompt_npy_load['task'] = task_item
            each_task_list.append(prompt_npy_load)
        trajectories_list.append(each_task_list)
    for task_item in test_env_list:
        print("eval_task_item:", task_item)
        index = task_item.find(query_str)
        env_par = task_item[index + len(query_str):]
        env_slice = env_par.split('_')
        language_prompt_eval = f'Current MEC system has the bandwidth of {env_slice[0]} MHz, the computational capability is {env_slice[1]} GHz, and the task size is about {env_slice[2]} kb.'
        language_prompt_list.append(language_prompt_eval)
        each_task_list = []
        prompt_file_list = os.listdir(os.path.join(prompt_npy_path, 'eval', task_item))
        for prompt_file in prompt_file_list:
            prompt_file_dir = os.path.join(prompt_npy_path, 'eval', task_item, prompt_file)
            prompt_npy_load = np.load(prompt_file_dir, allow_pickle=True)
            prompt_npy_load = prompt_npy_load.tolist()
            prompt_npy_load['task'] = task_item
            each_task_list.append(prompt_npy_load)
        trajectories_list.append(each_task_list)
    return trajectories_list, language_prompt_list


args = Parser().parse_args('diffusion')
global_config = GlobalConfig()
args_seed = global_config.train_config.seed
torch.cuda.manual_seed(args_seed)
torch.cuda.manual_seed_all(args_seed)

torch.manual_seed(args_seed)
np.random.seed(args_seed)
random.seed(args_seed)
print("args:\n", args)
task_list = []
if global_config.train_config.save_buffer_npy_path is not None:
    # task_list = ['basketball-v2', 'bin-picking-v2']
    task_list_all = os.listdir(
        os.path.join(global_config.train_config.save_buffer_npy_path, 'seed_' + str(args_seed)))  # TODO 这里之后要删掉
    test_env_list = global_config.train_config.test_env_list
    task_list = [x for x in task_list_all if x not in test_env_list]

    task_list = task_list[1:3]  # TODO 正式实验时去掉

# -----------------------------------------------------------------------------#
# ---------------------------------- dataset ----------------------------------#
# -----------------------------------------------------------------------------#
if global_config.train_config.save_prompt_npy_path is not None and global_config.train_config.if_generate_prompt:
    print("get_prompt_file")
    get_prompt_file(global_config, task_list)
# exit()
test_env_list = global_config.train_config.test_env_list
print("task_list:", task_list)
prompt_trajectories, language_prompt_list = get_prompt_trajectories(global_config, task_list, test_env_list)
language_prompt_list = language_embedding(language_prompt_list)

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    replay_dir_list=global_config.train_config.save_buffer_npy_path,
    task_list=task_list,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    # max_n_episodes=global_config.train_config.episode_num,
    max_path_length=global_config.train_config.step_num,
    ## value-specific kwargs
    optimal=args.optimal,
    discount=args.discount,
    termination_penalty=args.termination_penalty,
    normed=args.normed,
    # meta_world=True,
    seq_length=2,
    global_config=global_config
)

dataset = dataset_config()
# renderer = render_config()
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim
reward_dim = 1

# -----------------------------------------------------------------------------#
# ------------------------------ model & trainer ------------------------------#
# -----------------------------------------------------------------------------#
model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + 1,  # + action_dim,# + reward_dim,
    cond_dim=observation_dim,
    num_tasks=50,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
    train_device=args.device,
    prompt_trajectories=prompt_trajectories,
    language_prompt_list=language_prompt_list,
    verbose=False,
    task_list=task_list,
    action_dim=action_dim,
    global_config=global_config
)
diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.MetaworldTrainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    envs=task_list,
    task_list=task_list,
    is_unet=args.is_unet,
    trainer_device=args.device,
    horizon=args.horizon,
    global_config=global_config
)

# -----------------------------------------------------------------------------#
# -------------------------------- instantiate --------------------------------#
# -----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)
renderer = None
trainer = trainer_config(diffusion, dataset, renderer)

# -----------------------------------------------------------------------------#
# ------------------------ test forward & backward pass -----------------------#
# -----------------------------------------------------------------------------#

utils.report_parameters(model)
print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
# loss, _ = diffusion.loss(*batch, device=args.device)
# loss.backward()
print('✓')

# -----------------------------------------------------------------------------#
# --------------------------------- main loop ---------------------------------#
# -----------------------------------------------------------------------------#
# args.n_train_steps = 5e4
# args.n_steps_per_epoch = 1000
n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    print("trainer.train:", trainer.train)
    trainer.train(n_train_steps=args.n_steps_per_epoch)
