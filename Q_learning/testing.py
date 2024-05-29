from solver import solver
import matplotlib.pyplot as plt
import numpy as np
import gym


def test_learning_rate():
    env = gym.make('Taxi-v3')
    env.reset()
    state_size = env.observation_space.n
    action_size = env.action_space.n
    s = solver(state_size, action_size)
    rewards = []
    steps = []
    learning_rates = np.arange(0.1, 2.1, 0.1)
    for learning_rate in learning_rates:
        r, st = s.q_learning(
            action_info=env.step, reset_func=env.reset, learning_rate=learning_rate,
            discount_rate=0.8, epsilon=1.0, decay=0.005, episodes=1000, max_steps=100
        )
        rewards.append(np.mean(r))
        steps.append(np.mean(st))
    plt.figure(1)
    plt.plot(learning_rates, rewards)
    plt.xlabel('Learning rates')
    plt.ylabel('Mean of rewards')
    plt.title('Q-learning rewards')
    plt.figure(2)
    plt.plot(learning_rates, steps)
    plt.xlabel('Learning rates')
    plt.ylabel('Mean of steps')
    plt.title('Q-learning steps')
    plt.show()
    env.close()

def test_discount_rate():
    env = gym.make('Taxi-v3')
    env.reset()
    state_size = env.observation_space.n
    action_size = env.action_space.n
    s = solver(state_size, action_size)
    rewards = []
    steps = []
    discount_rates = np.arange(0.1, 2.1, 0.1)
    for discount_rate in discount_rates:
        r, st = s.q_learning(
            action_info=env.step, reset_func=env.reset, learning_rate=0.9,
            discount_rate=discount_rate, epsilon=1.0, decay=0.005, episodes=1000, max_steps=100
        )
        rewards.append(np.mean(r))
        steps.append(np.mean(st))
    plt.figure(1)
    plt.plot(discount_rates, rewards)
    plt.xlabel('Discount rates')
    plt.ylabel('Mean of rewards')
    plt.title('Q-learning rewards')
    plt.figure(2)
    plt.plot(discount_rates, steps)
    plt.xlabel('Discount rates')
    plt.ylabel('Mean of steps')
    plt.title('Q-learning steps')
    plt.show()
    env.close()

def test_episode():
    env = gym.make('Taxi-v3')
    env.reset()
    state_size = env.observation_space.n
    action_size = env.action_space.n
    s = solver(state_size, action_size)
    rewards = []
    steps = []
    episodes = np.arange(100, 2100, 100)
    for episode in episodes:
        r, st = s.q_learning(
            action_info=env.step, reset_func=env.reset, learning_rate=0.9,
            discount_rate=0.8, epsilon=1.0, decay=0.005, episodes=episode, max_steps=100
        )
        rewards.append(np.mean(r))
        steps.append(np.mean(st))
    plt.figure(1)
    plt.plot(episodes, rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Mean of rewards')
    plt.title('Q-learning rewards')
    plt.figure(2)
    plt.plot(episodes, steps)
    plt.xlabel('Episodes')
    plt.ylabel('Mean of steps')
    plt.title('Q-learning steps')
    plt.show()
    env.close()

def single_run():
    env = gym.make('Taxi-v3')
    env.reset()
    state_size = env.observation_space.n
    action_size = env.action_space.n
    s = solver(state_size, action_size)
    rewards, steps = s.q_learning(
        action_info=env.step, reset_func=env.reset, learning_rate=0.9,
        discount_rate=1, epsilon=1.0, decay=0.005, episodes=1000, max_steps=100
    )
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
    env.close()
    env2 = gym.make('Taxi-v3', render_mode='human')
    s.play(
        reset_function=env2.reset, action_info=env2.step, max_steps=100
    )


if __name__=="__main__":
    test_learning_rate()
    test_discount_rate()
    test_episode()
    single_run()
