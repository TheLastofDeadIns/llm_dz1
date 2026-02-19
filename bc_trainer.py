import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical

class StudentAgent:
    """Оборачивает модель политики, чтобы она имела метод get_action, как у PGAgent"""
    def __init__(self, policy):
        self.policy = policy

    def get_action(self, state, deterministic=False):
        state_t = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            probs = self.policy(state_t)  # уже softmax
        if deterministic:
            action = torch.argmax(probs).item()
            # Для совместимости возвращаем нулевые log_prob и entropy
            return action, torch.tensor(0.0), torch.tensor(0.0)
        else:
            dist = Categorical(probs)
            action = dist.sample().item()
            return action, dist.log_prob(torch.tensor(action)), dist.entropy()

def run_behavior_cloning(expert, env, epochs=20, n_episodes=50):
    # 1. Сбор данных с эксперта (детерминированные действия)
    states, actions = [], []
    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a, _, _ = expert.get_action(s, deterministic=True)
            states.append(s)
            actions.append(a)
            s, _, term, trunc, _ = env.step(a)
            done = term or trunc

    # 2. Создание студента (новая сеть)
    student = expert.policy.__class__(4, 2)  # вход 4, выход 2
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

    # 3. Подготовка данных
    states_t = torch.tensor(np.array(states), dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    dataset = TensorDataset(states_t, actions_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 4. Обучение
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            logits = student(xb, return_logits=True)  # логиты для CrossEntropyLoss
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, loss: {total_loss/len(loader):.4f}")

    # Возвращаем обёрнутого студента, чтобы он имел метод get_action
    return StudentAgent(student)

def evaluate_agent(agent, env, n_episodes=100, deterministic=True):
    """Оценка средней награды агента в среде"""
    rewards = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            a, _, _ = agent.get_action(s, deterministic=deterministic)
            s, r, term, trunc, _ = env.step(a)
            total_reward += r
            done = term or trunc
        rewards.append(total_reward)
    return np.mean(rewards)