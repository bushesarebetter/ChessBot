import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class ChessBot(nn.Module):
    def __init__(self, input_channels=17, board_size=8, num_squares=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.policy_start_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_start_fc = nn.Linear(2 * board_size * board_size, num_squares)

        self.policy_end_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_end_fc = nn.Linear(2 * board_size * board_size, num_squares)


        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv(x)

        policy_start = F.relu(self.policy_start_conv(x))
        policy_start = policy_start.view(policy_start.size(0), -1)
        policy_start = self.policy_start_fc(policy_start)

        policy_end = F.relu(self.policy_end_conv(x))
        policy_end = policy_end.view(policy_end.size(0), -1)
        policy_end = self.policy_end_fc(policy_end)

        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_start, policy_end, value

