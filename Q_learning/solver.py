import numpy as np
import random


class solver():
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((self.state_size, self.action_size))

    def q_learning(
            self, action_info: callable, reset_func: callable,
            learning_rate: float, discount_rate: float, epsilon: float, decay: float, episodes: int, max_steps: int
    ):
        all_rewards = []
        all_steps = []
        for episode in range(episodes):
            done = False
            final_reward = 0
            state = reset_func()[0]
            for i in range(max_steps):
                # exploration - exploitation trade-off
                if random.uniform(0, 1) < epsilon:
                    # explore
                    action = random.choice(range(self.action_size))
                else:
                    # exploit
                    action = np.argmax(self.q_table[state, :])

                new_state, reward, done, _, _ = action_info(action)

                # update Q-table
                self.q_table[state, action] = self.q_table[state, action] + learning_rate * (
                            reward + discount_rate * np.max(self.q_table[new_state, :]) - self.q_table[state, action])

                state = new_state
                if done:
                    break
                final_reward += reward
            epsilon *= np.exp(-decay * episode)
            all_rewards.append(final_reward)
            all_steps.append(i)
            print(f'Episode {episode} done')
        print(f"Training completed after {episode+1} episodes")
        return all_rewards, all_steps

    def play(
            self, reset_function: callable, action_info: callable, max_steps=100):
        state = reset_function()[0]
        rewards = 0
        for _ in range(max_steps):
            action = np.argmax(self.q_table[state, :])
            state, reward, done, _, _ = action_info(action)
            rewards += reward
            if done:
                break
        print(f"Total rewards: {rewards}")
        return rewards
