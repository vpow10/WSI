import numpy as np
import gym
import random

class solver():
    def __init__(self):
        self.env = gym.make('Taxi-v3', render_mode='human')
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.q_table = np.zeros((self.state_size, self.action_size))

    def random_agent(self):
        steps = 100
        self.env.reset()
        for _ in range(steps):
            action = self.env.action_space.sample()
            self.env.step(action)
            self.env.render()
        self.env.close()

    def q_learning_agent(
            self, learning_rate: float, discount_rate: float, epsilon: float, decay: float, episodes: int, max_steps: int):
        for episode in range(episodes):
            state = self.env.reset()[0]
            done = False

            for _ in range(max_steps):
                # exploration - exploitation trade-off
                if random.uniform(0, 1) < epsilon:
                    # explore
                    action = self.env.action_space.sample()
                else:
                    # exploit
                    action = np.argmax(self.q_table[state, :])

            new_state, reward, done, truncated, _ = self.env.step(action)

            # update Q-table
            self.q_table[state, action] = self.q_table[state, action] + learning_rate * (reward + discount_rate * np.max(self.q_table[new_state, :]) - self.q_table[state, action])

            state = new_state
            if done or truncated:
                break
            epsilon *= np.exp(-decay * episode)
            print(f'Episode {episode} done')
        print(f"Training completed after {episode+1} episodes")
        self.env.close()

    def play(self, max_steps=100):
        self.env = gym.make('Taxi-v3', render_mode='human')
        state = self.env.reset()[0]
        done = False
        rewards = 0
        for _ in range(max_steps):
            action = np.argmax(self.q_table[state, :])
            print(action)
            new_state, reward, done, truncated, _ = self.env.step(action)
            rewards += reward
            state = new_state
            if done or truncated:
                break
        print(f"Total rewards: {rewards}")
        self.env.close()

if __name__ == '__main__':
    s = solver()
    s.q_learning_agent(0.9, 0.8, 1.0, 0.005, 100, 100)
    s.play()
