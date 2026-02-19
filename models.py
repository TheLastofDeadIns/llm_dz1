import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, is_critic=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.is_critic = is_critic

    def forward(self, x, return_logits=False):
        x = self.net(x)
        if self.is_critic:
            return x  # для value network выход без активации
        else:
            if return_logits:
                return x  # логиты для CrossEntropyLoss
            else:
                return F.softmax(x, dim=-1)  # вероятности для Categorical