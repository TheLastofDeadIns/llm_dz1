import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core import PGAgent
from bc_trainer import run_behavior_cloning, evaluate_agent


def train_and_compare():
    methods = ["vanilla", "mean_baseline", "critic", "rloo"]
    env = gym.make("CartPole-v1")
    results = {m: [] for m in methods}
    agents = {}

    n_episodes = 1000  # Увеличил до 1000, чтобы все сошлись
    ent_coef_init = 0.01

    for m in methods:
        print(f"Training: {m}")
        agent = PGAgent(4, 2, ent_coef_init=ent_coef_init, lr_value=1e-3, lr_policy=1e-3)
        rewards_history = []
        ent_coef = ent_coef_init
        decay = ent_coef / n_episodes

        for ep in range(n_episodes):
            s, _ = env.reset()
            states, rewards, logs, ents = [], [], [], []
            done = False
            while not done:
                a, log_p, ent = agent.get_action(s)
                s_next, r, term, trunc, _ = env.step(a)
                states.append(s)
                rewards.append(r)
                logs.append(log_p)
                ents.append(ent)
                s = s_next
                done = term or trunc

            agent.update(rewards, logs, ents, states, method=m, ent_coef=ent_coef, normalize_adv=True)
            rewards_history.append(sum(rewards))
            ent_coef = max(0, ent_coef - decay)

        results[m] = rewards_history
        agents[m] = agent

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index_label="episode")
    print("Results saved to results.csv")

    plt.figure(figsize=(12, 6))
    for m in methods:
        plt.plot(results[m], label=m, alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Policy Gradient Methods on CartPole (1000 episodes)")
    plt.legend()
    plt.grid()
    plt.savefig("comparison.png", dpi=150)
    plt.show()
    print("Plot saved to comparison.png")

    # Выбираем лучшего эксперта
    best_method = None
    best_avg = -float('inf')
    for m in methods:
        avg_last_100 = np.mean(results[m][-100:])
        print(f"{m}: average last 100 episodes = {avg_last_100:.2f}")
        if avg_last_100 > best_avg:
            best_avg = avg_last_100
            best_method = m
    print(f"Best method: {best_method}")

    return agents[best_method], results


if __name__ == "__main__":
    expert_agent, results = train_and_compare()

    # Behavior Cloning
    env = gym.make("CartPole-v1")
    student_model = run_behavior_cloning(expert_agent, env, epochs=20, n_episodes=50)

    # оценка
    print("\nBehavior Cloning Evaluation:")
    student_avg = evaluate_agent(student_model, env, n_episodes=100, deterministic=True)
    print(f"Student average reward: {student_avg:.2f}")

    # сравнение
    expert_avg = evaluate_agent(expert_agent, env, n_episodes=100, deterministic=True)
    print(f"Expert average reward: {expert_avg:.2f}")

    # BC с плохим экспертом
    print("\n--- BC with Poor Expert ---")
    poor_agent = PGAgent(4, 2)
    for ep in range(50):
        s, _ = env.reset()
        states, rewards, logs, ents = [], [], [], []
        done = False
        while not done:
            a, log_p, ent = poor_agent.get_action(s)
            s_next, r, term, trunc, _ = env.step(a)
            states.append(s);
            rewards.append(r);
            logs.append(log_p);
            ents.append(ent)
            s = s_next;
            done = term or trunc
        poor_agent.update(rewards, logs, ents, states, method="critic")

    poor_avg = evaluate_agent(poor_agent, env, 100, True)
    print(f"Poor expert (50 episodes) average: {poor_avg:.2f}")

    student_poor = run_behavior_cloning(poor_agent, env, epochs=20, n_episodes=50)
    student_poor_avg = evaluate_agent(student_poor, env, 100, True)
    print(f"Student cloned from poor expert: {student_poor_avg:.2f}")