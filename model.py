from collections import OrderedDict
from typing import OrderedDict as OD

from torch import nn

StateDict = OD

first_run = True
global_model = nn.Sequential(nn.Linear(784, 128),
                             nn.ReLU(),
                             nn.Linear(128, 10),
                             nn.LogSoftmax(dim=1)
                             )


def init_layers(local_model: StateDict):
    global_model.load_state_dict(local_model)


def compute_deltas(local_model: StateDict, alpha: float):
    deltas = OrderedDict()
    glob_m = global_model.state_dict()
    for key, val in local_model.items():
        deltas[key] = alpha * (val - glob_m[key])
    return deltas


def update_global(deltas: StateDict):
    new_state_dict = OrderedDict()
    for key, val in global_model.state_dict().items():
        new_state_dict[key] = val + deltas[key]
    global_model.load_state_dict(new_state_dict)
