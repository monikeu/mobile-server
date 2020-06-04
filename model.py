from typing import List

import torch
import torch.optim as optim
from torch import nn, Tensor


class Mobilenet(nn.Module):
    def __init__(self, input_size: int, lr: float = 0.005, momentum: float = 0.9):

        super().__init__()
        self.inner_net = nn.Sequential(nn.Linear(input_size, 32),  # 8?
                                       nn.ReLU(),
                                       nn.Linear(32, 1),
                                       nn.ReLU())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.inner_net.parameters(), lr=lr, momentum=momentum)
        self.losses = []
        self.cloud_results = []
        self.local_results = []

    def _with_cloud(self, tensor):
        ones = torch.ones(tensor.size(0), tensor.size(1), 1, requires_grad=True)
        x = torch.cat([tensor, ones], axis=2)
        # x.requires_grad_(True)
        return x

    def _with_local(self, tensor):
        zeros = torch.zeros(tensor.size(0), tensor.size(1), 1, requires_grad=True)
        x = torch.cat([tensor, zeros], axis=2)
        # x.requires_grad_(True)
        return x

    def _prepare_target(self,
                        result: float):  # chwilowo size (1, 1, result_size) a result size to tylko czas wiec tez 1
        return torch.tensor([result]).view((1, 1, 1))  # todo ogarnąć też inne rzeczy, tj nie tylko czas

    def updateModel(self, input: List[float], result: float, mode: bool):

        input = torch.tensor(input).view(1, 1, 7)
        self.optimizer.zero_grad()
        output = self.inner_net(self._with_cloud(input)) if mode == 1 else self.inner_net(self._with_local(input))

        loss = self.criterion(output, self._prepare_target(result))
        loss.backward()
        self.optimizer.step()
        l = loss.item()
        print(l)
        self.losses.append(l)
        if mode == 1:
            self.cloud_results.append((input, result))
        else:
            self.local_results.append((input, result))

    def forward(self, input: Tensor, grad=False):
        def _forward():
            cloud_cost = self.inner_net(self._with_cloud(input))
            local_cost = self.inner_net(self._with_local(input))
            # 1. -cloud, 0 - local

            return (cloud_cost < local_cost).double()

        if grad:
            return _forward()
        else:
            with torch.no_grad():
                return _forward()

# StateDict = OD
#
# first_run = True
# global_model = nn.Sequential(nn.Linear(784, 128),
#                              nn.ReLU(),
#                              nn.Linear(128, 10),
#                              nn.LogSoftmax(dim=1)
#                              )
#
#
# def init_layers(local_model: StateDict):
#     global_model.load_state_dict(local_model)
#
#
# def compute_deltas(local_model: StateDict, alpha: float):
#     deltas = OrderedDict()
#     glob_m = global_model.state_dict()
#     for key, val in local_model.items():
#         deltas[key] = alpha * (val - glob_m[key])
#     return deltas
#
#
# def update_global(deltas: StateDict):
#     new_state_dict = OrderedDict()
#     for key, val in global_model.state_dict().items():
#         new_state_dict[key] = val + deltas[key]
#     global_model.load_state_dict(new_state_dict)
