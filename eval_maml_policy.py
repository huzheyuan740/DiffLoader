import maml_rl.envs
import gym
import numpy as np
import torch
import json

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from config import GlobalConfig

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


def total_rewards(episodes_rewards, aggregation=torch.mean):
	# print("episodes_rewards:", episodes_rewards)
	# print("xx:", torch.stack([aggregation(torch.sum(rewards, dim=0))
	#                                   for rewards in episodes_rewards], dim=0))
	# exit()
	rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
	                                  for rewards in episodes_rewards], dim=0))
	return rewards.item()


def main(args):

	args.output_folder = args.env_name
	global_config = GlobalConfig()

	# continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
	#                                         'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
	#                                         '2DNavigation-v0'])

	continuous_actions = True

	writer = SummaryWriter('./maml-based_tensorboard/{0}'.format(args.output_folder))
	save_folder = '/mnt/windisk/dataset_diffloader_mec/MAML-based/{0}'.format(args.output_folder)
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)

	with open(os.path.join(save_folder, 'config.json'), 'w') as f:
		# config = {k: v for (k, v) in vars(args).iteritems() if k != 'device'}
		config = {k: v for (k, v) in vars(args).items() if k != 'device'}
		config.update(device=args.device.type)
		json.dump(config, f, indent=2)
		print(config)

	sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size, num_workers=args.num_workers, global_config=global_config)

	if continuous_actions:
		policy = NormalMLPPolicy(
			int(np.prod(sampler.envs.observation_space)), # input shape
			int(np.prod(sampler.envs.action_space)), # output shape
			hidden_sizes=(args.hidden_size,) * args.num_layers) # [100, 100]
	else:
		policy = CategoricalMLPPolicy(
			int(np.prod(sampler.envs.observation_space.shape)),
			sampler.envs.action_space.n,
			hidden_sizes=(args.hidden_size,) * args.num_layers)

	# load trained policy
	policy.load_state_dict(torch.load(global_config.train_config.maml_policy_model_path))
	print("Load Policy Model!")

	baseline = LinearFeatureBaseline( int(np.prod(sampler.envs.observation_space)))

	metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
	                          fast_lr=args.fast_lr, tau=args.tau, device=args.device)

	for batch in range(args.num_batches): # number of epoches

		tasks = sampler.sample_eval_tasks(num_tasks=args.meta_batch_size, global_config=global_config)  # 采集出不同方向的任务
		print("tasks:", tasks)
		episodes = metalearner.sample(tasks, first_order=args.first_order, writer=writer, current_batch=batch)
		print("1?")

		# metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
		#                  cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
		#                  ls_backtrack_ratio=args.ls_backtrack_ratio)
		# print("2?")

		# Tensorboard
		writer.add_scalar('total_rewards/before_update',
		                  total_rewards([ep.rewards for ep, _ in episodes]), batch)
		writer.add_scalar('total_rewards/after_update',
		                  total_rewards([ep.rewards for _, ep in episodes]), batch)


		# # Save policy network
		# with open(os.path.join(save_folder, 'policy-{0}.pt'.format(batch)), 'wb') as f:
		# 	torch.save(policy.state_dict(), f)

		print(batch, total_rewards([ep.rewards for ep, _ in episodes]), total_rewards([ep.rewards for _, ep in episodes]))


if __name__ == '__main__':
	"""
	python main.py --env-name HalfCheetahDir-v1 --output-folder maml-halfcheetah-dir \
	--fast-lr 0.1 --meta-batch-size 30 --fast-batch-size 20 --num-batches 1000
	"""
	import argparse
	import os
	import multiprocessing as mp

	parser = argparse.ArgumentParser(description='Reinforcement learning with '
	                                             'Model-Agnostic Meta-Learning (MAML)')

	# General
	parser.add_argument('--env-name', type=str, default='mec_eval',
	                    help='name of the environment')
	parser.add_argument('--gamma', type=float, default=0.99,
	                    help='value of the discount factor gamma')
	parser.add_argument('--tau', type=float, default=1.0,
	                    help='value of the discount factor for GAE')
	parser.add_argument('--first-order', action='store_true',
	                    help='use the first-order approximation of MAML')

	# Policy network (relu activation function)
	parser.add_argument('--hidden-size', type=int, default=100,
	                    help='number of hidden units per layer')
	parser.add_argument('--num-layers', type=int, default=2,
	                    help='number of hidden layers')

	# Task-specific
	parser.add_argument('--fast-batch-size', type=int, default=1,
	                    help='batch size for each individual task')
	parser.add_argument('--fast-lr', type=float, default=0.1, # 0.5
	                    help='learning rate for the 1-step gradient update of MAML')

	# Optimization
	parser.add_argument('--num-batches', type=int, default=1000,
	                    help='number of batches, or number of epoches')
	parser.add_argument('--meta-batch-size', type=int, default=40,
	                    help='number of tasks per batch')
	parser.add_argument('--max-kl', type=float, default=1e-2,
	                    help='maximum value for the KL constraint in TRPO')
	parser.add_argument('--cg-iters', type=int, default=10,
	                    help='number of iterations of conjugate gradient')
	parser.add_argument('--cg-damping', type=float, default=1e-5,
	                    help='damping in conjugate gradient')
	parser.add_argument('--ls-max-steps', type=int, default=15,
	                    help='maximum number of iterations for line search')
	parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
	                    help='maximum number of iterations for line search')

	# Miscellaneous
	parser.add_argument('--output-folder', type=str, default='maml-mec-dir',
	                    help='name of the output folder')
	parser.add_argument('--num-workers', type=int, default=1,
	                    help='number of workers for trajectories sampling')
	parser.add_argument('--device', type=str, default='cuda',
	                    help='set the device (cpu or cuda)')

	args = parser.parse_args()

	# Create logs and saves folder if they don't exist
	if not os.path.exists('./logs'):
		os.makedirs('./logs')
	if not os.path.exists('./saves'):
		os.makedirs('./saves')
	# Device
	args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	# Slurm
	if 'SLURM_JOB_ID' in os.environ:
		args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

	main(args)
