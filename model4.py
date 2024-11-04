import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the game environment
class SimpleGame:
    def __init__(self):
        self.state_size = 4  # Example state size (could be x, y, goal_x, goal_y)
        self.action_size = 2  # Example action size (move left, move right)
        self.learning_rate = 0.001
        self.q_model = self.build_model()
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.epsilon_min = 0.01  # Minimum exploration probability
        self.max_steps_per_episode = 50  # Max steps per episode

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def reset(self):
        # Reset the environment to the initial state
        return np.random.rand(self.state_size)  # Example: random initial state

    def step(self, action):
        # Simulate taking an action in the environment
        next_state = np.random.rand(self.state_size)
        reward = 1 if np.random.rand() > 0.5 else -1  # Example reward
        done = np.random.rand() > 0.95  # Randomly end the episode
        return next_state, reward, done

    def train(self, episodes):
        for episode in range(episodes):
            state = self.reset()
            done = False
            steps = 0
            while not done and steps < self.max_steps_per_episode:
                # Choose action based on the current policy
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(range(self.action_size))
                else:
                    q_values = self.q_model.predict(state.reshape(1, -1), verbose=0)  # Avoid logging
                    action = np.argmax(q_values[0])

                next_state, reward, done = self.step(action)
                target = reward + 0.95 * np.max(self.q_model.predict(next_state.reshape(1, -1), verbose=0)[0])
                target_f = self.q_model.predict(state.reshape(1, -1), verbose=0)
                target_f[0][action] = target
                self.q_model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

                state = next_state
                steps += 1
            
            # Decay epsilon after each episode
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Print progress every 100 episodes
            if episode % 100 == 0:
                print(f"Episode {episode}/{episodes}, Epsilon: {self.epsilon:.3f}")

# Instantiate and train the agent
game = SimpleGame()
game.train(1000)

# Example test to visualize the Q-values for the initial state
state = game.reset()
q_values = game.q_model.predict(state.reshape(1, -1), verbose=0)
print("Q-values for the initial state:", q_values)
