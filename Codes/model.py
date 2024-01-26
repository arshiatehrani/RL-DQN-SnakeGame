import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Define the third fully connected layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Define the output layer
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        # Define the ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass data through fc1
        l1 = self.relu(self.fc1(x))
        # Pass data through fc2
        l2 = self.relu(self.fc2(l1))
        # Pass data through fc3
        l3 = self.relu(self.fc3(l2))
        # Pass data through fc4
        l4 = self.fc4(l3)
        return l4


def get_network_input(player, apple):
    proximity = player.get_proximity()
    x = torch.cat(
        [
            torch.from_numpy(player.pos).float(),
            torch.from_numpy(apple.pos).float(),
            torch.from_numpy(player.dir).float(),
            torch.tensor(proximity).float(),
        ]
    )
    return x


# def get_network_input(player, apple):
#     proximity = player.get_proximity()
#     x = torch.cat(
#         [
#             torch.from_numpy(player.pos).double(),
#             torch.from_numpy(apple.pos).double(),
#             torch.from_numpy(player.dir).double(),
#             torch.tensor(proximity).double(),
#         ]
#     )
#     return x
