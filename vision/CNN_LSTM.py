import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class VisualCNN(nn.Module):
    def __init__(self, num_rays):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(32, 2)  # Predict x, y position

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, R)
        features = self.cnn(x).squeeze(-1)  # (B, 32)
        return self.fc(features)  # (B, 2)

class MotionLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32):
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

    def forward(self, rays_seq, pos_seq):
        cnn_input = rays_seq[:, -1, :]  # last frame (B, R)
        lstm_input = pos_seq  # (B, T, 2)

        v_spatial = self.visual_cnn(cnn_input)  # (B, 2)
        l_spatial = self.motion_lstm(lstm_input)  # (B, 4)

        g = self.gating_net(v_spatial, l_spatial)  # (B, 1)
        s_pos = g * v_spatial + (1 - g) * l_spatial[:, :2]  # final position
        s_vel = l_spatial[:, 2:]  # only LSTM predicts velocity
        return torch.cat([s_pos, s_vel], dim=-1)  # (B, 4)