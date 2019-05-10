import os
import cv2
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import gzip
import pickle
import h5py

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data import load_data, prepare_input, normalize, denormalize
from models import DPINet
from utils import calc_box_init_FluidShake
from copy import deepcopy


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--pruning_perc', type=int, required=True)
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--time_step_clip', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./60.)
parser.add_argument('--nf_relation', type=int, default=300)
parser.add_argument('--nf_particle', type=int, default=200)
parser.add_argument('--nf_effect', type=int, default=200)
parser.add_argument('--outf', default='files')
parser.add_argument('--dataf', default='data/small/fluid_shake/')
parser.add_argument('--evalf', default='eval')
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--verbose_data', type=int, default=0)
parser.add_argument('--verbose_model', type=int, default=0)

parser.add_argument('--debug', type=int, default=0)

parser.add_argument('--n_instances', type=int, default=0)
parser.add_argument('--n_stages', type=int, default=0)
parser.add_argument('--n_his', type=int, default=0)

# shape state:
# [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
parser.add_argument('--shape_state_dim', type=int, default=14)

# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)
parser.add_argument('--position_dim', type=int, default=0)

# relation attr:
parser.add_argument('--relation_dim', type=int, default=0)

args = parser.parse_args()

phases_dict = dict()

if args.env == 'FluidFall':
    env_idx = 4
    data_names = ['positions', 'velocities']

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [fluid]
    args.attr_dim = 1

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.time_step = 121
    args.time_step_clip = 5
    args.n_instance = 1
    args.n_stages = 1

    args.neighbor_radius = 0.08

    phases_dict["instance_idx"] = [0, 189]
    phases_dict["root_num"] = [[]]
    phases_dict["instance"] = ['fluid']
    phases_dict["material"] = ['fluid']

    args.outf = 'dump_FluidFall/' + args.outf
    args.evalf = 'dump_FluidFall/' + args.evalf

elif args.env == 'BoxBath':
    resume_epoch = 4
    resume_iter = 370000
    env_idx = 1
    data_names = ['positions', 'velocities', 'clusters']

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [rigid, fluid, root_0]
    args.attr_dim = 3

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.time_step = 151
    args.time_step_clip = 0
    args.n_instance = 2
    args.n_stages = 4

    args.neighbor_radius = 0.08

    # ball, fluid
    phases_dict["instance_idx"] = [0, 64, 1024]
    phases_dict["root_num"] = [[8], []]
    phases_dict["root_sib_radius"] = [[0.4], []]
    phases_dict["root_des_radius"] = [[0.08], []]
    phases_dict["root_pstep"] = [[args.pstep], []]
    phases_dict["instance"] = ['cube', 'fluid']
    phases_dict["material"] = ['rigid', 'fluid']

    args.outf = 'dump_BoxBath/' + args.outf
    args.evalf = 'dump_BoxBath/' + args.evalf

elif args.env == 'FluidShake':
    resume_epoch = 4
    resume_iter = 500000
    env_idx = 6
    data_names = ['positions', 'velocities', 'shape_quats', 'scene_params']

    height = 1.0
    border = 0.025

    # object states:
    # [x, y, z, xdot, ydot, zdot]
    args.state_dim = 6
    args.position_dim = 3

    # object attr:
    # [fluid, wall_0, wall_1, wall_2, wall_3, wall_4]
    # wall_0: floor
    # wall_1: left
    # wall_2: right
    # wall_3: back
    # wall_4: front
    args.attr_dim = 6

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = 1
    args.time_step = 301
    args.time_step_clip = 0
    args.n_stages = 2

    args.neighbor_radius = 0.08

    phases_dict["root_num"] = [[]]
    phases_dict["instance"] = ["fluid"]
    phases_dict["material"] = ["fluid"]

    args.outf = 'dump_FluidShake/' + args.outf
    args.evalf = 'dump_FluidShake/' + args.evalf

elif args.env == 'RiceGrip':
    resume_epoch = 18
    resume_iter = 130000
    env_idx = 5
    data_names = ['positions', 'velocities', 'shape_quats', 'clusters', 'scene_params']

    args.n_his = 3

    # object state:
    # [rest_x, rest_y, rest_z, rest_xdot, rest_ydot, rest_zdot,
    #  x, y, z, xdot, ydot, zdot, quat.x, quat.y, quat.z, quat.w]
    args.state_dim = 16 + 6 * args.n_his
    args.position_dim = 6

    # object attr:
    # [fluid, root, gripper_0, gripper_1,
    #  clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep]
    args.attr_dim = 7

    # relation attr:
    # [none]
    args.relation_dim = 1

    args.n_instance = 1
    args.time_step = 41
    args.time_step_clip = 0
    args.n_stages = 4

    args.neighbor_radius = 0.08
    args.n_roots = 30

    phases_dict["root_num"] = [[args.n_roots]]
    phases_dict["root_sib_radius"] = [[5.0]]
    phases_dict["root_des_radius"] = [[0.2]]
    phases_dict["root_pstep"] = [[args.pstep]]
    phases_dict["instance"] = ["rice"]
    phases_dict["material"] = ["fluid"]

    args.outf = 'dump_RiceGrip/' + args.outf
    args.evalf = 'dump_RiceGrip/' + args.evalf

else:
    raise AssertionError("Unsupported env")


args.outf = args.outf + '_' + args.env
args.evalf = args.evalf + '_' + args.env
# args.dataf = 'data/' + args.dataf + '_' + args.env
print(args)


print("Loading stored stat from %s" % args.dataf)
stat_path = os.path.join(args.dataf, 'stat.h5')
stat = load_data(data_names[:2], stat_path)
for i in range(len(stat)):
    stat[i] = stat[i][-args.position_dim:, :]
    # print(data_names[i], stat[i].shape)


use_gpu = torch.cuda.is_available()

model = DPINet(args, stat, phases_dict, residual=True, use_gpu=use_gpu)

model_file = './dump_{}/files_{}/net_epoch_{}_iter_{}.pth'.format(args.env, args.env, resume_epoch, resume_iter)
used_vars = np.load('./{}_used_vars.npy'.format(args.env))
print("Loading network from %s" % model_file)
model.load_state_dict(torch.load(model_file))
model.eval()


pruning_perc = args.pruning_perc

# Calculate threshold
all_parameters = []
for name, param in model.named_parameters():
    if name in used_vars and param.requires_grad:
        all_parameters += list(param.cpu().data.abs().numpy().flatten())
threshold = np.percentile(np.array(all_parameters), pruning_perc)
print("Threshold: {}".format(threshold))

# Copy parameters
original_params = deepcopy(model.state_dict())
pruned_params = deepcopy(model.state_dict())

# Generate pruned params
num_params = 0
for name, param in pruned_params.items():
    if name in used_vars:
        param *= (param.data.abs() > threshold).float()
        num_params += param.numel()

assert len(all_parameters) == num_params

model.load_state_dict(pruned_params)
num_zeros = 0
for name, param in model.named_parameters():
    if name in used_vars:
        num_zeros += np.sum((param == 0).cpu().numpy())
        # print(param)

print("Num zeros: {}, Num params: {}".format(num_zeros, num_params))
print("Sparsity: {}".format(num_zeros / num_params))
torch.save(model.state_dict(), './dump_{}/files_{}/net_epoch_{}_iter_{}_pruning_{}.pth'.format(args.env, args.env, resume_epoch, resume_iter, pruning_perc))
