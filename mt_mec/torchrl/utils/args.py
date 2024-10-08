import argparse
import json

import torch

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--worker_nums', type=int, default=4,
                        help='worker nums')

    parser.add_argument('--eval_worker_nums', type=int, default=2,
                        help='eval worker nums')

    parser.add_argument("--config", type=str,   default=None,
                        help="config file", )

    parser.add_argument('--save_dir', type=str, default='./snapshots',
                        help='directory for snapshots (default: ./snapshots)')

    parser.add_argument('--data_dir', type=str, default='./data',
                        help='directory for snapshots (default: ./snapshots)')

    parser.add_argument('--log_dir', type=str, default='./log_pre',
                        help='directory for tensorboard logs (default: ./log)')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument("--device", type=int, default=0,
                        help="gpu secification", )

    # tensorboard
    parser.add_argument("--id", type=str,   default=None,
                        help="id for tensorboard", )

    # policy snapshot
    parser.add_argument("--pf_snap", type=str,   default=None,
                        help="policy snapshot path", )
    # q function snapshot
    parser.add_argument("--qf1_snap", type=str,   default=None,
                        help="policy snapshot path", )
    # q function snapshot
    parser.add_argument("--qf2_snap", type=str,   default=None,
                        help="policy snapshot path", )

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def get_params(file_name):
    with open(file_name) as f:
        params = json.load(f)
    return params
