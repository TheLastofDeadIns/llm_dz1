import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from models import MLP

class PGAgent:
    def __init__(self, state_dim, action_dim, lr_policy=1e-3, lr_value=1e-3, gamma=0.99, ent_coef_init=0.01):
        self.policy = MLP(state_dim, action_dim)
        self.value_net = MLP(state_dim, 1, is_critic=True)
        self.opt_p = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.opt_v = torch.optim.Adam(self.value_net.parameters(), lr=lr_value)
        self.gamma = gamma
        self.ent_coef_init = ent_coef_init

    def get_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.policy(state)
        dist = Categorical(probs)
        if deterministic:
            action = torch.argmax(probs)
            # Для детерминированного режима log_prob и энтропия не нужны, вернём нули
            return action.item(), torch.tensor(0.0), torch.tensor(0.0)
        else:
            action = dist.sample()
            return action.item(), dist.log_prob(action), dist.entropy()

    def update(self, rewards, log_probs, entropies, states, method="vanilla", ent_coef=None, normalize_adv=True, n_value_updates=5):
        """
        method: 'vanilla', 'mean_baseline', 'critic', 'rloo'
        ent_coef: коэффициент энтропии (если None, используется self.ent_coef_init)
        normalize_adv: нормализовать advantage после вычитания бейзлайна
        n_value_updates: количество обновлений критика на один вызов (для метода 'critic')
        """
        # 1. Расчёт returns (ненормализованных)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # 2. Вычисление advantage
        if method == "vanilla":
            advantage = returns
        elif method == "mean_baseline":
            baseline = returns.mean()  # среднее по эпизоду как бейзлайн
            advantage = returns - baseline
        elif method == "critic":
            states_t = torch.tensor(np.array(states), dtype=torch.float32)
            # Обучаем критика: делаем несколько шагов для улучшения оценки
            for _ in range(n_value_updates):
                values = self.value_net(states_t).squeeze()
                v_loss = F.mse_loss(values, returns)
                self.opt_v.zero_grad()
                v_loss.backward()
                self.opt_v.step()
            # После обучения используем detach для advantage
            with torch.no_grad():
                values = self.value_net(states_t).squeeze()
            advantage = returns - values
        elif method == "rloo":
            # Leave-one-out по временным шагам эпизода
            total = returns.sum()
            n = len(returns)
            advantage = returns - (total - returns) / (n - 1)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 3. Опциональная нормализация advantage
        if normalize_adv:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # 4. Policy loss + энтропийная регуляризация
        if ent_coef is None:
            ent_coef = self.ent_coef_init

        pg_loss = -(torch.stack(log_probs) * advantage).mean()
        entropy_loss = -ent_coef * torch.stack(entropies).mean()
        loss = pg_loss + entropy_loss

        self.opt_p.zero_grad()
        loss.backward()
        self.opt_p.step()

        return loss.item()