from solver import solver
import matplotlib.pyplot as plt
import numpy as np


def single_run():
    s = solver()
    rewards, steps = s.q_learning_agent(learning_rate=0.9, discount_rate=0.8, epsilon=1.0, decay=0.005, episodes=1000, max_steps=100)
    plt.figure(1)
    episodes = np.arange(0, len(rewards))
    plt.scatter(episodes, rewards, linewidths=0.1, edgecolors='black')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Q-learning rewards')
    plt.figure(2)
    plt.scatter(episodes, steps, linewidths=0.1, edgecolors='black')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title('Q-learning steps')
    plt.show()
    s.play()


if __name__=="__main__":
    single_run()
