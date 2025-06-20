import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class VisualCNN(nn.Module):
    def __init__(self, num_rays):
        super().__init__()
        # self.cnn = nn.Sequential(
        #     nn.Conv1d(1, 16, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(16, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool1d(1)
        # )
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output x, z
        )

        # self.fc = nn.Linear(32, 2)  # Predict x, y position

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, R)
        features = self.cnn(x).squeeze(-1)  # (B, 128), (B, 32)
        return self.fc(features)  # (B, 2)

class MotionLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 4)  # Predict x, y, vx, vy

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, H)
        return self.fc(out[:, -1])  # (B, 4)

class GatingNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, lstm_feat):
        x = torch.cat([cnn_feat, lstm_feat], dim=-1)
        return self.fc(x)  # scalar gate between 0 and 1

class GatedFusionModel(nn.Module):
    def __init__(self, num_rays):
        super().__init__()
        self.visual_cnn = VisualCNN(num_rays)
        self.motion_lstm = MotionLSTM()
        self.gating_net = GatingNetwork(input_size=6)  # [x,y] from CNN + [x,y,vx,vy] from LSTM

    def forward(self, ray_seq):
        cnn_input = ray_seq[:, -1, :]  # (B, R)
        cnn_pos = self.visual_cnn(cnn_input)  # (B, 2)

        # Build synthetic position history from CNN predictions
        # cnn_pos_seq = cnn_pos.unsqueeze(1).repeat(1, ray_seq.shape[1], 1)  # (B, T, 2)
        cnn_pos_seq = torch.stack([self.visual_cnn(ray_seq[:, t, :]) for t in range(ray_seq.shape[1])], dim=1)  # (B, T, 2)

        lstm_out = self.motion_lstm(cnn_pos_seq)  # (B, 4)

        # B, T, R = ray_seq.shape
        # cnn_input = ray_seq.view(B * T, R)
        # cnn_pos = self.visual_cnn(cnn_input).view(B, T, -1)  # (B, T, 2)
        # lstm_out = self.motion_lstm(cnn_pos)  # (B, 4)

        gate = self.gating_net(cnn_pos, lstm_out)  # (B, 1)

        # Blend positions
        blended_pos = gate * cnn_pos + (1 - gate) * lstm_out[:, :2]  # (B, 2)
        velocity = lstm_out[:, 2:]  # (B, 2)

        # return torch.cat([blended_pos, velocity], dim=-1)  # (B, 4)
        # print(cnn_pos_seq.shape)

        return torch.cat([blended_pos, velocity, cnn_pos_seq[:, -1, :]], dim=-1)
        # return torch.cat([blended_pos, velocity], dim=-1), cnn_pos
    

    