import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, layer_units=[64, 64], seed=0, dueling=False):
        super(QNetwork, self).__init__()
        self._state_size = state_size
        self._action_size = action_size
        self._seed = torch.manual_seed(seed)
        self._dueling = dueling

        # Arquitecture (Dueling arq)
        num_units = len(layer_units)
        layers = [nn.Linear(state_size, layer_units[0])]
        for i in range(1, num_units):
            layers.append(nn.Linear(layer_units[i-1], layer_units[i]))

        self.fc_layers = nn.ModuleList(layers)
        self.V_layer = nn.Linear(layer_units[num_units-1], 1)
        self.A_layer = nn.Linear(layer_units[num_units-1], action_size)

        # Init layers
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = state
        for layer in self.fc_layers:
            x = F.relu(layer(x))

        A_values = self.A_layer(x)
        if not self._dueling: return A_values

        V_value = self.V_layer(x)
        Q_values = V_value + (A_values - A_values.mean(1, keepdim=True))
        return Q_values
