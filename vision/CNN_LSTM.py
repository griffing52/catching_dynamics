import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# class VisualCNN(nn.Module):
#     def __init__(self, num_rays):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv1d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         self.fc = nn.Linear(32, 2)  # Predict x, y position

#     def forward(self, x):
#         x = x.unsqueeze(1)  # (B, 1, R)
#         features = self.cnn(x).squeeze(-1)  # (B, 32)
#         return self.fc(features)  # (B, 2)
    
class VisualCNN(nn.Module):
    def __init__(self, n_raycasts=107):
        """
        Args:
            n_raycasts (int): The number of raycasts in the input vector (n).
        """
        super(VisualCNN, self).__init__()
        
        # We define the convolutional part of the network
        self.convolutional_layers = nn.Sequential(
            # Input shape: (batch_size, 1, n_raycasts)
            # The '1' is the number of input channels.
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            # Shape: (batch_size, 16, n_raycasts)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Shape: (batch_size, 16, n_raycasts / 2)

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # Shape: (batch_size, 32, n_raycasts / 2)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # Shape: (batch_size, 32, n_raycasts / 4)
        )
        
        # Calculate the size of the flattened layer after the convolutions
        # This is a robust way to do it without manual calculation
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_raycasts)
            flattened_size = self.convolutional_layers(dummy_input).flatten(1).shape[1]

        # We define the fully connected (regression) part of the network
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=flattened_size, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout for regularization
            nn.Linear(in_features=64, out_features=2) # Output layer for (x, z)
        )

    def forward(self, x):
        """
        The forward pass of the model.
        Args:
            x (torch.Tensor): The input tensor of raycasts. 
                              Expected shape: (batch_size, n_raycasts)
        """
        # Add a channel dimension for the Conv1d layers. Shape becomes (batch_size, 1, n_raycasts)
        x = x.unsqueeze(1)
        
        # Pass through convolutional layers
        x = self.convolutional_layers(x)
        
        # Flatten the output for the linear layers. Shape becomes (batch_size, flattened_size)
        x = x.flatten(1)
        
        # Pass through linear layers to get the final prediction
        prediction = self.linear_layers(x)
        
        return prediction

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
    

    