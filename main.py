import torch
from model import Runner, BatchLoader
import os
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

split = '1'
action = 'predict'

num_blocks = 1
num_layers = 2
num_channels = 2
i3d_dim = 400
bz = 1
lr = 0.0005
end_epoch = 1
start_epoch = 0
weight_decay = 0.0001

COMP_PATH = 'data'

vid_list_file = os.path.join(COMP_PATH, "splits/train.split"+split+".bundle")
vid_list_file_tst = os.path.join(COMP_PATH, "splits/test.split"+split+".bundle")
train_features_path = os.path.join(COMP_PATH, "train_features/")
test_features_path = os.path.join(COMP_PATH, "test_features/")
gt_path = os.path.join(COMP_PATH, "groundTruth/")

mapping_file = os.path.join(COMP_PATH, "splits/mapping_bf.txt")

model_dir = os.path.join(COMP_PATH, "models")
results_dir = os.path.join(COMP_PATH, "results")
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

trainer = Runner(num_blocks, num_layers, num_channels, i3d_dim, num_classes)
if action == "train":
    batch_gen = BatchLoader(num_classes, actions_dict, gt_path, train_features_path)
    batch_gen.read_vid_list(vid_list_file)
    trainer.train(model_dir, batch_gen, end_epoch=end_epoch, batch_size=bz, learning_rate=lr, device=device, start_epoch=start_epoch, weight_decay=weight_decay)

if action == "predict":
    trainer.predict(model_dir, results_dir, test_features_path, vid_list_file_tst, end_epoch, device, segment_file)
