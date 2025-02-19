import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, os
from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import warnings
import wandb
from datetime import datetime
import json

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_grad_norm(model, layer_name):
    norm = 0.0
    for name, param in model.named_parameters():
        if layer_name in name and param.grad is not None:
            norm += torch.norm(param.grad).item()
    return norm

# Define the EvalMetric class to evaluate the model performance
class EvalMetric:
    def __init__(self, multilabel=False):
        self.multilabel = multilabel
        self.pred_list = []  
        self.audio_pred_list = []
        self.video_pred_list = []
        self.mm_pred_list = []
        self.truth_list = []  
        self.loss_list = []  
        self.audio_ME_loss_list = []
        self.video_ME_loss_list = []
        self.calc_audio_ME_loss_list = []
        self.calc_video_ME_loss_list = []
        self.total_loss_list = []
        self.contribution_list = []

    def append_classification_results(self, labels, preds, audio_preds, video_preds, mm_preds, loss, audio_ME_loss, video_ME_loss, total_loss, contribution, coef):
        predictions = np.argmax(preds.detach().cpu().numpy(), axis=1)
        audio_predictions = np.argmax(audio_preds.detach().cpu().numpy(), axis=1)
        video_predictions = np.argmax(video_preds.detach().cpu().numpy(), axis=1)
        mm_predictions = np.argmax(mm_preds.detach().cpu().numpy(), axis=1)
        for idx in range(len(predictions)):
            self.pred_list.append(predictions[idx])
            self.audio_pred_list.append(audio_predictions[idx])
            self.video_pred_list.append(video_predictions[idx])
            self.mm_pred_list.append(mm_predictions[idx])
            self.truth_list.append(labels.detach().cpu().numpy()[idx])
        
        self.contribution_list.append(contribution)
        self.loss_list.append(loss.item())
        self.audio_ME_loss_list.append(audio_ME_loss.item())
        self.video_ME_loss_list.append(video_ME_loss.item())
        self.calc_audio_ME_loss_list.append(coef*contribution['audio']*audio_ME_loss.item())
        self.calc_video_ME_loss_list.append(coef*contribution['video']*video_ME_loss.item())
        self.total_loss_list.append(total_loss.item())
        
    def classification_summary(self, return_auc=False):
        result_dict = {}
        result_dict['acc'] = accuracy_score(self.truth_list, self.pred_list) * 100
        result_dict['audio_acc'] = accuracy_score(self.truth_list, self.audio_pred_list) * 100
        result_dict['video_acc'] = accuracy_score(self.truth_list, self.video_pred_list) * 100
        result_dict['mm_acc'] = accuracy_score(self.truth_list, self.mm_pred_list) * 100
        
        result_dict['uar'] = recall_score(self.truth_list, self.pred_list, average="macro") * 100
        result_dict['audio_uar'] = recall_score(self.truth_list, self.audio_pred_list, average="macro") * 100
        result_dict['video_uar'] = recall_score(self.truth_list, self.video_pred_list, average="macro") * 100
        result_dict['mm_uar'] = recall_score(self.truth_list, self.mm_pred_list, average="macro") * 100
        
        result_dict["loss"] = np.mean(self.loss_list)
        result_dict["audio_ME_loss"] = np.mean(self.audio_ME_loss_list)
        result_dict["video_ME_loss"] = np.mean(self.video_ME_loss_list)
        result_dict["calc_audio_ME_loss"] = np.mean(self.calc_audio_ME_loss_list)
        result_dict["calc_video_ME_loss"] = np.mean(self.calc_video_ME_loss_list)
        result_dict["total_loss"] = np.mean(self.total_loss_list)
        
        result_dict["contributions"] = {
            "audio": np.mean([item['audio'] for item in self.contribution_list]),
            "video": np.mean([item['video'] for item in self.contribution_list]),
            "multimodal": np.mean([item['multimodal'] for item in self.contribution_list])
        }
        # result_dict['top5_acc'] = (np.sum(np.array(self.top_k_list) == np.array(self.truth_list).reshape(len(self.truth_list), 1)) / len(self.truth_list)) * 100
        result_dict['conf'] = np.round(confusion_matrix(self.truth_list, self.pred_list, normalize='true') * 100, decimals=2)
        result_dict["sample"] = len(self.truth_list)
        result_dict['f1'] = f1_score(self.truth_list, self.pred_list, average='macro') * 100
        
        if return_auc:
            result_dict['auc'] = roc_auc_score(self.truth_list, self.pred_list) * 100
        
        return result_dict
    
# Function to pad tensors for consistent input sizes
def pad_tensor(vec, pad):
    pad_size = list(vec.shape)
    pad_size[0] = pad - vec.size(0)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=0)

# Collate function for multimodal data padding
def collate_mm_fn_padd(batch):
    if batch[0][0] is not None:
        max_a_len = max(map(lambda x: x[0].shape[0], batch))
    if batch[0][1] is not None:
        max_b_len = max(map(lambda x: x[1].shape[0], batch))

    x_a, x_b, len_a, len_b, ys = list(), list(), list(), list(), list()
    for idx in range(len(batch)):
        x_a.append(pad_tensor(batch[idx][0], pad=max_a_len))
        x_b.append(pad_tensor(batch[idx][1], pad=max_b_len))

        len_a.append(torch.tensor(batch[idx][2]))
        len_b.append(torch.tensor(batch[idx][3]))

        ys.append(batch[idx][-1])

    x_a = torch.stack(x_a, dim=0)
    x_b = torch.stack(x_b, dim=0)
    len_a = torch.stack(len_a, dim=0)
    len_b = torch.stack(len_b, dim=0)
    ys = torch.stack(ys, dim=0)
    return x_a, x_b, len_a, len_b, ys

class MMDatasetGenerator(Dataset):
    def __init__(self, modalityA, modalityB, default_feat_shape_a, default_feat_shape_b, data_len: int, dataset: str = ''):
        self.data_len = data_len
        # Each element in modalityA and modalityB is now a tuple: (label, sample)
        self.modalityA = modalityA  # Expected audio shape: (257, 299) -> will become (1, 257, 299)
        self.modalityB = modalityB  # Expected video shape: (3, 3, 224, 224)
        self.default_feat_shape_a = default_feat_shape_a  # should be np.array([1, 257, 299])
        self.default_feat_shape_b = default_feat_shape_b  # remains np.array([3, 3, 224, 224])
        self.dataset = dataset

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # Unpack the tuple for audio and video
        label_a, audio_sample = self.modalityA[index]
        label_b, video_sample = self.modalityB[index]
        # Ensure the labels are the same
        assert label_a == label_b, f"Label mismatch at index {index}: {label_a} vs {label_b}"
        label = torch.tensor(label_a)
        # Convert numpy arrays to torch tensors
        # For audio, add a channel dimension to get shape (1, 257, 299)
        audio_tensor = torch.tensor(audio_sample).unsqueeze(0)
        video_tensor = torch.tensor(video_sample)  # Already should be (3, 3, 224, 224)
        # For audio, use the time dimension (299) as length; for video, number of frames (3)
        len_a = audio_tensor.shape[2]  # 299
        len_b = video_tensor.shape[0]  # 3
        return audio_tensor, video_tensor, len_a, len_b, label

class DataloadManager:
    def __init__(self, args):
        self.args = args
        self.label_dist_dict = dict()
        self.audio_feat_path = Path(self.args.data_dir)
        self.video_feat_path = Path(self.args.data_dir)

    def get_client_ids(self, fold_idx: int = 1):
        data_path = self.video_feat_path.joinpath(f'fold{fold_idx}')
        # self.client_ids = [f.replace("video.pkl", "") for f in os.listdir(str(data_path)) if f.endswith("video.pkl")]
        self.client_ids = ['1']
        self.client_ids.sort()

    def load_audio_feat(self, client_id: str, fold_idx: int = 1) -> list:
        # For test files, load testtaudio.pkl; otherwise load the client file.
        if client_id == "testaudio":
            data_path = self.audio_feat_path.joinpath(f'fold{fold_idx}', "testtaudio.pkl")
        else:
            data_path = self.audio_feat_path.joinpath(f'fold{fold_idx}', f'{client_id}audio.pkl')
        with open(str(data_path), "rb") as f:
            data = pickle.load(f)
        print(f"Loaded audio data from: {data_path}")
        return data

    def load_video_feat(self, client_id: str, fold_idx: int = 1) -> list:
        if client_id == "testvideo":
            data_path = self.video_feat_path.joinpath(f'fold{fold_idx}', "testtvideo.pkl")
        else:
            data_path = self.video_feat_path.joinpath(f'fold{fold_idx}', f'{client_id}video.pkl')
        with open(str(data_path), "rb") as f:
            data = pickle.load(f)
        print(f"Loaded video data from: {data_path}")
        return data

    def set_dataloader(self, data_a: list, data_b: list,
                       default_feat_shape_a: np.array = np.array([1, 257, 299]),
                       default_feat_shape_b: np.array = np.array([3, 3, 224, 224]),
                       shuffle: bool = False) -> DataLoader:
        data_ab = MMDatasetGenerator(data_a, data_b, default_feat_shape_a, default_feat_shape_b, len(data_a), self.args.dataset)
        dataloader = DataLoader(data_ab, batch_size=64, num_workers=0, shuffle=shuffle, collate_fn=collate_mm_fn_padd)
        return dataloader


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, modality, num_classes=1000, pool='avgpool', zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self.modality = modality
        self.pool = pool
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if modality == 'audio':
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        elif modality == 'visual':
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            raise NotImplementedError('Incorrect modality, should be audio or visual but got {}'.format(modality))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        if self.modality == 'visual':
            (B, C, T, H, W) = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(B * T, C, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)  # audio: bsz x 128 x 33 x 38  visual: bsz x 128 x 28 x 28
        x_shape = x.shape

        x = self.layer3(x)
        x = self.layer4(x)
        out = x
        # print(out.shape)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def _resnet(arch, block, layers, modality, progress, **kwargs):
    model = ResNet(block, layers, modality, **kwargs)
    return model


def resnet18(modality, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], modality, progress,
                   **kwargs)


def resnet34(modality, progress=True, **kwargs):  # nclasses, nf=64, bias=True
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], modality, progress, **kwargs)


def resnet101(modality, progress=True, **kwargs):  # nclasses, nf=64, bias=True
    return _resnet('resnet101', BasicBlock, [3, 4, 23, 3], modality, progress, **kwargs)



class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return output

    def forward_uni_audio(self, x):
        y = torch.zeros_like(x).to(x.device)
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return output

    def forward_uni_visual(self, y):
        x = torch.zeros_like(y).to(y.device)
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return output

class AVClassifierImproved(nn.Module):
    def __init__(self, args):
        super(AVClassifierImproved, self).__init__()
        self.dropout_p = 0.1
        num_classes = 6
        self.classifier = nn.Sequential(
            nn.Linear(1024, 64),  
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, num_classes)
        )
        self.a_classifier = nn.Sequential(
            nn.Linear(512, 64),  
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, num_classes)
        )
        self.v_classifier = nn.Sequential(
            nn.Linear(512, 64),  
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, num_classes)
        )
        self.gate = nn.Sequential(
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, 3), 
            nn.Softmax(dim=1)  
        )
        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

    def forward(self, audio, visual, bsz=None):
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)
        av = torch.cat((a, v), dim=1)
        
        av_preds = self.classifier(av)
        a_preds = self.a_classifier(a)
        v_preds = self.v_classifier(v)
        self.gate_weights = self.gate(av)  # Shape: (batch_size, 3)
        preds = (
            self.gate_weights[:, 0].unsqueeze(1) * a_preds +
            self.gate_weights[:, 1].unsqueeze(1) * v_preds +
            self.gate_weights[:, 2].unsqueeze(1) * av_preds
        )

        return (preds, a_preds, v_preds, av_preds), (a, v)
    
    def get_contribution(self):
        return {
            "audio": self.gate_weights[:, 0].mean().item(),  
            "video": self.gate_weights[:, 1].mean().item(),  
            "multimodal": self.gate_weights[:, 2].mean().item(),  
        }
        
class ClientFedAvg(object):
    def __init__(self, args, device, criterion, dataloader, model):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1e-5)

    def get_parameters(self):
        return self.model.state_dict()
    
    def set_model_weights(self, global_model):
        # load server state dict to client model
        state_dict = self.model.state_dict()
        for k in state_dict.keys():
            state_dict[k] = global_model.state_dict()[k].clone().detach()
        self.model.load_state_dict(state_dict)
        

    def update_weights(self, global_audio_prototype, global_video_prototype):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1e-5)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        
        client_loss = 0
        client_acc = 0
        num_classes = 6  # Assuming 6 emotion classes

        # Initialize local prototypes as tensors
        self.local_audio_prototype = torch.zeros((num_classes, 512), device=self.device)
        self.local_video_prototype = torch.zeros((num_classes, 512), device=self.device)
        # x_audio.shape[1] = x_video.shape[1] = 64
        
        # print(f"Begining: {self.model.classifier[0].weight.norm()}")
        for epoch in range(int(self.args.local_epochs)):
            total_train = 0
            correct_train = 0
            batch_loss = 0
            
            for batch_idx, batch_data in enumerate(self.dataloader):
                
                self.model.zero_grad()
                self.optimizer.zero_grad()
                
                
                x_a, x_b, l_a, l_b, y = batch_data
                # print(x_a.shape, x_b.shape)
                x_a.unsqueeze(0)
                x_b.unsqueeze(0)
                # print(x_a.shape, x_b.shape)
                x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                
                # outputs, x = self.model(x_a.float(), x_b.float(), l_a, l_b)
                outputs, x = self.model(x_a.float(), x_b.float())
                
                preds, audio_preds, video_preds, mm_preds = outputs
                x_audio, x_video = x
                contribution = self.model.get_contribution()
                # ========================
                # Compute Local Prototypes
                # ========================
                class_counts = torch.zeros((num_classes,), device=self.device)

                for label in range(num_classes):
                    label_indices = (y == label).nonzero(as_tuple=True)[0]
                    if len(label_indices) > 0:
                        self.local_audio_prototype[label] += x_audio[label_indices].mean(dim=0).detach()
                        self.local_video_prototype[label] += x_video[label_indices].mean(dim=0).detach()
                        class_counts[label] += 1

                # Normalize local prototypes (prevent division by zero)
                for label in range(num_classes):
                    if class_counts[label] > 0:
                        self.local_audio_prototype[label] /= class_counts[label]
                        self.local_video_prototype[label] /= class_counts[label]
                        
                # ========================
                # Compute Modal Enhancement Loss (ME Loss)
                # ========================
                audio_modal_enhancement_loss = 0.0
                video_modal_enhancement_loss = 0.0
                if global_audio_prototype is not None and global_video_prototype is not None:
                    for label in range(num_classes):
                        if torch.norm(global_audio_prototype[label]) > 0:
                            audio_modal_enhancement_loss += torch.norm(
                                x_audio[y == label] - global_audio_prototype[label], p=2
                            ).mean()

                        if torch.norm(global_video_prototype[label]) > 0:
                            video_modal_enhancement_loss += torch.norm(
                                x_video[y == label] - global_video_prototype[label], p=2
                            ).mean()
                            
                loss = self.criterion(preds, y)
                total_loss = loss + 0.01*(contribution['audio']*audio_modal_enhancement_loss + contribution['video']*video_modal_enhancement_loss)
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
                batch_loss += total_loss.item()
                _, predicted = torch.max(preds.data, 1)
                total_train += y.size(0)
                correct_train += (predicted == y).sum().item()
            client_loss = batch_loss / total_train # Average loss per batch
            client_acc = 100 * correct_train / total_train
            
            print(f'Epoch [{epoch+1}], Acc: {100 * correct_train / total_train}, Loss: {batch_loss / len(self.dataloader)}')
            
        # print(f"After: {self.model.classifier[0].weight.norm()}")
        # import IPython; IPython.embed(); exit(0)
        
        return client_loss, client_acc
        # return client_loss / int(self.args.local_epochs), client_acc / int(self.args.local_epochs)  # Average loss per epoch

class Server(object):
    def __init__(self, args, model, device, criterion, client_ids, fold_idx):
        self.args = args
        self.device = device
        self.criterion = criterion
        self.client_ids = [cid for cid in client_ids if cid not in ['dev', 'test']]
        self.fold_idx = fold_idx  
        self.multilabel = False 
        self.clients = []
        self.result_dict = dict()
        self.global_model = model.to(self.device)
        self.global_optimizer = None  
        self.best_epoch = -1
        self.best_uar = 0.0
        self.best_acc = 0.0
        self.best_loss = float('inf')
        self.global_audio_prototype = None
        self.global_video_prototype = None
        self.all_result = dict()

    def initialize_clients(self, dataload_manager):
        self.clients = []
        for client_id in self.client_ids:
            print(f"Loading data for client: {client_id}...")
            audio_feat = dataload_manager.load_audio_feat(client_id)
            video_feat = dataload_manager.load_video_feat(client_id)
            
            print(f"Number of audio samples: {len(audio_feat)}, video samples: {len(video_feat)}")
            dataloader = dataload_manager.set_dataloader(audio_feat, video_feat, 
                                                         default_feat_shape_a=np.array([257, 299]), 
                                                         default_feat_shape_b=np.array([3, 3, 224, 224]), 
                                                         shuffle=True)
            
            if dataloader is not None:
                self.clients.append(ClientFedAvg(self.args, self.device, self.criterion, dataloader, 
                                                 AVClassifierImproved(self.args).to(device)))
            else:
                print(f"No data loaded for client: {client_id}. Skipping...")

        print(f"Total clients initialized for training: {len(self.clients)}")

    def aggregate_weights(self):
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client.get_parameters()[k].float() for client in self.selected_clients], 0).mean(0)
        self.global_model.load_state_dict(global_dict)
        
        # Ensure clients have valid prototypes and convert them to tensors
        audio_prototypes = [client.local_audio_prototype for client in self.selected_clients]
        video_prototypes = [client.local_video_prototype for client in self.selected_clients]

        # Stack tensors along a new dimension and compute mean across clients
        self.global_audio_prototype = torch.mean(torch.stack(audio_prototypes), dim=0)
        self.global_video_prototype = torch.mean(torch.stack(video_prototypes), dim=0)

        
    
    def train(self, dataload_manager, test_dataloader):
        self.initialize_clients(dataload_manager)
        
        for epoch in range(int(self.args.num_epochs)):  # global epoch
            print(f"Training Epoch {epoch + 1}/{self.args.num_epochs}...")
            epoch_losses = []
            client_acces = []
            grad_norm_audio_total = 0.0
            grad_norm_video_total = 0.0
            # Log per-client metrics at the final local epoch of each global epoch
            client_metrics = {}
            self.selected_clients = random.sample(self.clients, max(1, int(len(self.clients) * 0.1)))
            for i, client in enumerate(self.selected_clients):
                client_id_str = f"Client_{str(i+1).zfill(2)}"  # Zero-padding for sorting
                print(f"Training client {i + 1} of {len(self.selected_clients)}")
                
                # Update the weights for each client
                client.set_model_weights(self.global_model)
                client_loss, client_acc = client.update_weights(self.global_audio_prototype, self.global_video_prototype)
                epoch_losses.append(client_loss)
                client_acces.append(client_acc)

                # Accumulate gradient norms
                grad_norm_audio_total += compute_grad_norm(client.model, 'audio_proj')
                grad_norm_video_total += compute_grad_norm(client.model, 'video_proj')

                # Store the metrics for each client to log later
                client_metrics[f"Accuracy/{client_id_str}"] = client_acc
                client_metrics[f"Loss/{client_id_str}"] = client_loss


            # Perform aggregation only on the selected subset
            self.aggregate_weights()

            grad_norm_audio_avg = grad_norm_audio_total / len(self.selected_clients)
            grad_norm_video_avg = grad_norm_video_total / len(self.selected_clients)

            print(f"Global Epoch {epoch + 1}: Grad Norm Audio (avg): {grad_norm_audio_avg:.4f}, "
                f"Grad Norm Video (avg): {grad_norm_video_avg:.4f}")
            if self.args.use_wandb:
                wandb.log({
                    f"global_epoch": epoch + 1,
                    "Avg Grad Norm/Audio Projection": grad_norm_audio_avg,
                    "Avg Grad Norm/Video Projection": grad_norm_video_avg
                })
            # Print average loss and accuracy for the epoch
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            avg_client_acc = sum(client_acces) / len(client_acces)
            print(f"Average Training Loss for Epoch {epoch + 1}: {avg_epoch_loss:.4f}, Avg Acc: {avg_client_acc:.2f}%")
            
            # Perform inference on the test set after each epoch
            print(f"Performing inference on test set after Epoch {epoch + 1}...")
            test_result = self.inference(test_dataloader)
            self.save_result(epoch)
            # Print the results of the test set for this epoch
            print(f"Epoch {epoch + 1} Test Set Results: Loss: {test_result['loss']:.4f}, UAR: {test_result['uar']:.2f}%, Top-1 Acc: {test_result['acc']:.2f}%")
            # Log the aggregated metrics for all clients after each global epoch
            if self.args.use_wandb:
                wandb.log({
                    "global_epoch": epoch + 1,
                    "Train/Avg Client Accuracy": avg_client_acc,
                    "Train/Avg Client Loss": avg_epoch_loss,
                    "Test/Global Model Loss": test_result['loss'],
                    "global_test_UAR": test_result['uar'],
                    "global_test_accuracy": test_result['acc'],
                    **client_metrics
                })
            # Track the best test UAR and epoch
            if test_result['uar'] > self.best_uar:
                self.best_epoch = epoch + 1
                self.best_uar = test_result['uar']
                self.best_acc = test_result['acc']
                self.best_loss = test_result['loss']
                print(f"New best epoch {self.best_epoch} with UAR: {self.best_uar:.2f}%, Acc: {self.best_acc:.2f}%")

        # After training is done, print the best epoch details
        print("--------------------------------------------------")
        print(f"Best epoch {self.best_epoch}")
        print(f"Best test UAR {self.best_uar:.2f}%, Top-1 Acc {self.best_acc:.2f}%")
    def inference(self, dataloader):
        self.global_model.eval()
        self.eval = EvalMetric(self.multilabel)

        print("Starting inference...")
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Inference Progress")):
            try:
                self.global_model.zero_grad()
                x_a, x_b, l_a, l_b, y = batch_data
                x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                l_a, l_b = l_a.to(self.device), l_b.to(self.device)

                # Forward pass
                outputs, x = self.global_model(x_a.float(), x_b.float())
                x_audio, x_video = x  # Unpack tuple
                preds, audio_preds, video_preds, mm_preds = outputs
                contribution = self.global_model.get_contribution()

                # ========================
                # Compute Modal Enhancement Loss (ME Loss)
                # ========================
                audio_modal_enhancement_loss = 0.0
                video_modal_enhancement_loss = 0.0
                if self.global_audio_prototype is not None and self.global_video_prototype is not None:
                    num_classes = self.global_audio_prototype.shape[0]

                    for label in range(num_classes):
                        label_indices = (y == label).nonzero(as_tuple=True)[0]
                        if len(label_indices) > 0:
                            if torch.norm(self.global_audio_prototype[label]) > 0:
                                audio_modal_enhancement_loss += torch.norm(
                                    x_audio[label_indices] - self.global_audio_prototype[label], p=2
                                ).mean()
                            if torch.norm(self.global_video_prototype[label]) > 0:
                                video_modal_enhancement_loss += torch.norm(
                                    x_video[label_indices] - self.global_video_prototype[label], p=2
                                ).mean()

                loss = self.criterion(preds, y)
                
                coef = 0.01
                total_loss = loss + coef*(contribution['audio']*audio_modal_enhancement_loss+contribution['video']*video_modal_enhancement_loss)

                self.eval.append_classification_results(y, preds, audio_preds, video_preds, mm_preds, loss, audio_modal_enhancement_loss, video_modal_enhancement_loss, total_loss, contribution, coef)
            except Exception as e:
                print(f"Error during batch {batch_idx} processing: {e}")
                continue

        print("Inference completed. Calculating results...")
        self.result = self.eval.classification_summary()
        return self.result
    
    def save_result(self,epoch):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        # Define directory path
        result_dir = f"/home/ubuntu/an.nd/an_fed_multimodal/result/raw_result/{current_time}{self.args.name}"
        os.makedirs(result_dir, exist_ok=True)  # Ensure directory exists
        
        contributions = self.global_model.get_contribution()
        jsonStringResult = json.dumps(self.result, cls=NumpyEncoder, indent=4)
        
        # Generate file name with timestamp
        file_path_result = os.path.join(result_dir, f"result_ep{epoch}.json")
        
        
        # Use context manager to safely handle file writing
        with open(file_path_result, "w") as jsonFile:
            jsonFile.write(jsonStringResult)
            

def main():
    parser = argparse.ArgumentParser(description='Federated Learning on CREMA-D')
    parser.add_argument('--data_dir', type=str, default='/home/ubuntu/shared/tuan.nm/emotion-data/CREMA-D/customFL_CREMA_D')
    parser.add_argument('--audio_feat', type=str, default='mfcc')
    parser.add_argument('--video_feat', type=str, default='mobilenet_v2')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=180)
    parser.add_argument('--local_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='crema_d')
    parser.add_argument('--use_wandb', type=bool, default=False)
    parser.add_argument('--name', type=str, default='temp')
    
    args = parser.parse_args()

    # Start W&B
    if args.use_wandb:
        wandb.login(key='710928f0b4c2130352c6a4f15630536ca8a018ed')
        wandb.init(project="fed_multimodal_improved", entity="nguyendaian1202-vinuniversity", config=args, name=f"{current_time} {args.name}")
    
    # Initialize dataload manager and client ids
    dataload_manager = DataloadManager(args)
    dataload_manager.get_client_ids(fold_idx=1)

    # Filter out 'dev', 'test' from the client ids for training
    train_client_ids = [client_id for client_id in dataload_manager.client_ids if client_id not in ['dev', 'test']]

    # Initialize model and criterion
    model = AVClassifierImproved(args)
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

    # Initialize server with filtered client ids
    server = Server(args, model, device, criterion, train_client_ids, fold_idx=1)

    # Load test data for evaluation
    test_dataloader = dataload_manager.set_dataloader(
        dataload_manager.load_audio_feat('testaudio'),
        dataload_manager.load_video_feat('testvideo'),
        default_feat_shape_a=np.array([1, 257, 299]),
        default_feat_shape_b=np.array([3, 3, 224, 224]),
        shuffle=False
    )

    # Train and evaluate
    server.train(dataload_manager, test_dataloader)

if __name__ == '__main__':
    main()
