import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch import optim
import copy
import numpy as np
import pandas as pd


class BatchLoader:
    def __init__(self, num_classes, actions_dict, gt_path, features_path):
        self.videos = []
        self.current_index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_folder = gt_path
        self.features_folder = features_path

    def reset(self):
        self.current_index = 0
        random.shuffle(self.videos)

    def has_next(self):
        return self.current_index < len(self.videos)

    def read_vid_list(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.videos = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.videos)

    def next_batch(self, batch_size):
        batch = self.videos[self.current_index:self.current_index + batch_size]
        self.current_index += batch_size

        batch_input = []
        batch_labels = []
        for vid in batch:
            vid_name = vid.split('/')[3]
            features = np.load(os.path.join(self.features_folder, vid_name.split('.')[0] + '.npy')).T
            file_ptr = open(self.gt_folder + vid_name, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input.append(features[:, ::1])
            batch_labels.append(classes[::1])

        length_of_sequences = list(map(len, batch_labels))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_label_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_label_tensor[i, :np.shape(batch_labels[i])[0]] = torch.from_numpy(batch_labels[i])
            mask[i, :, :np.shape(batch_labels[i])[0]] = torch.ones(self.num_classes, np.shape(batch_labels[i])[0])

        return batch_input_tensor, batch_label_tensor, mask


class VideoActionsModel(nn.Module):
    def __init__(self, num_blocks, num_layers, num_channels, i3d_dim, num_classes):
        super(VideoActionsModel, self).__init__()
        self.block0 = SingleBlock(num_layers, num_channels, i3d_dim, num_classes)
        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(copy.deepcopy(SingleBlock(num_layers, num_channels, num_classes, num_classes)))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x, mask):
        out = self.block0(x, mask)
        outputs = out.unsqueeze(0)
        for block in self.blocks:
            out = block(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleBlock(nn.Module):
    def __init__(self, num_layers, num_channels, i3d_dim, num_classes):
        super(SingleBlock, self).__init__()
        self.conv_1x1 = nn.Conv1d(i3d_dim, num_channels, 1)
        self.pool_1x3 = nn.MaxPool1d(kernel_size=3, padding=1, stride=1)

        self.layers = []
        for i in range(num_layers):
            self.layers.append(copy.deepcopy(ResidualLayer(2 ** i, num_channels)))
        self.layers = nn.ModuleList(self.layers)
        self.conv_out = nn.Conv1d(num_channels, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        out = self.pool_1x3(out)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class ResidualLayer(nn.Module):
    def __init__(self, dilation, num_channels):
        super(ResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(num_channels, num_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(num_channels, num_channels, 1)
        self.pool_1x3 = nn.MaxPool1d(kernel_size=3, padding=1, stride=1)
        self.batch_norm = nn.BatchNorm1d(num_channels)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.pool_1x3(out)
        out = self.batch_norm(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Runner:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = VideoActionsModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, loader, end_epoch, batch_size, learning_rate, weight_decay, device, start_epoch):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if start_epoch > 0:
            self.model.load_state_dict(torch.load(save_dir + "/epoch-" + str(start_epoch) + ".model"))
            optimizer.load_state_dict(torch.load(save_dir + "/epoch-" + str(start_epoch) + ".opt"))
        for epoch in range(start_epoch, end_epoch):
            epoch_loss = 0
            correct = 0
            total = 0
            while loader.has_next():
                batch_input, batch_target, mask = loader.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            loader.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(loader.videos),
                                                               float(correct)/total))

    def segment_result(self, result, segment_file_path, result_path):

        file_ptr = open(segment_file_path, 'r')
        test_all = file_ptr.read().split('\n')[:-1]

        pred_im = []
        for x in range(len(result)):
            segment_file_path = test_all[x].split(' ')
            image_arr = result[x]
            f = int(segment_file_path[0])
            for s in range(1, len(segment_file_path)):
                subsec = image_arr[f:int(segment_file_path[s])]
                pred = stats.mode(subsec)
                pred_im = np.append(pred_im, pred[0])
                f = int(segment_file_path[s]) + 1

        df = pd.DataFrame(data=pred_im.flatten().astype(int))
        df.columns = ['Category']
        df.index.name = 'Id'
        df.to_csv(os.path.join(result_path, 'res.csv'), index=True)

    def predict(self, model_folder, results_folder, features_path, vid_list_file, epoch, device, segment_file):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_folder + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            result = []
            for vid in list_of_vids:
                vid_name = vid.split('/')[3]
                features = np.load(os.path.join(features_path, vid_name.split('.')[0] + '.npy')).T
                features = features[:, ::1]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                result.append(predicted.tolist())
            self.segment_result(result, segment_file, results_folder)

