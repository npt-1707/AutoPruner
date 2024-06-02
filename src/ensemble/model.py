from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import torch
class BaggingModel(nn.Module):
    def __init__(self, base_model, n_estimators=10, estimator_params=None):
        super(BaggingModel, self).__init__()
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params if estimator_params is not None else {}
        self.estimators = [self.base_model(**self.estimator_params) for _ in range(n_estimators)]

    def __len__(self):
        return self.n_estimators

    def __getitem__(self, index):
        return self.estimators[index]

    def to(self, device):
        for estimator in self.estimators:
            estimator.to(device)
        return self

    def set_adam_optimizer(self, **kwargs):
        self.optimizers = [
            Adam(estimator.parameters(), **kwargs) for estimator in self.estimators
        ]

    def forward(self, code, struct):
        pred = [estimator(code, struct)[:, 1] for estimator in self.estimators]
        pred = sum(pred)/len(pred)
        return pred
    
    def state_dict(self):
        return {
            idx: estimator.state_dict() for idx, estimator in enumerate(self.estimators)
        }
        
    def load_state_dict(self, state_dict):
        assert len(state_dict) == self.n_estimators, "Invalid state dict: number of estimators mismatch"
        for idx, estimator in enumerate(self.estimators):
            estimator.load_state_dict(state_dict[idx])
            
    def eval(self):
        for estimator in self.estimators:
            estimator.eval()
    
    def train(self):
        for estimator in self.estimators:
            estimator.train()