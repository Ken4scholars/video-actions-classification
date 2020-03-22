#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# parser = argparse.ArgumentParser()
# parser.add_argument('--action', default='train')
# parser.add_argument('--dataset', default="gtea")
# parser.add_argument('--split', default='1')
#
# args = parser.parse_args()
split = '1'
action = 'predict'
dataset = 'breakfast'
args = type('args', (), {'split': split, 'action': action, 'dataset': dataset})()

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 400
bz = 1
lr = 0.0005
num_epochs = 16

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2
COMP_PATH = 'data'

vid_list_file = os.path.join(COMP_PATH, "splits/train.split"+args.split+".bundle")
vid_list_file_tst = os.path.join(COMP_PATH, "splits/test.split"+args.split+".bundle")
# features_path = os.path.join(COMP_PATH, "train_features/")
features_path = os.path.join(COMP_PATH, "test_features/")
gt_path = os.path.join(COMP_PATH, "groundTruth/")

mapping_file = os.path.join(COMP_PATH, "splits/mapping_bf.txt")

model_dir = os.path.join(COMP_PATH, "models/split_"+args.split)
results_dir = os.path.join(COMP_PATH, "results/split_"+args.split)
segment_file = 'test_segment.txt'
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)

trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate, segment_file)
