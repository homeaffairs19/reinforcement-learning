import numpy as np
import random
import matplotlib.pyplot as plt

# Define the environment parameters
GRID_SIZE = 5
OBSTACLES = [(1, 1), (1, 3), (2, 1), (3, 1)]  # Example obstacles
GOAL = (4, 4)

class GridWorld:
    def __init__(self):
        self.state_space = GRID_SIZE * GRID_SIZE
        self.action_space = 4  # Up, Down, Left, Right
        self.q_table = np.zeros((self.state_space, self.action_space))
        self.epsilon = 0.1  # Exploration rate
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        
    def state_to_coordinates(self, state):
        return (state // GRID_SIZE, state % GRID_SIZE)

    def coordinates_to_state(self, coords):
        return coords[0] * GRID_SIZE + coords[1]

    def is_terminal(self, state):
        return state == self.coordinates_to_state(GOAL)

    def reset(self):
        self.agent_pos = (0, 0)
        return self.coordinates_to_state(self.agent_pos)

    def step(self, action):
        row, col = self.agent_pos
        if action == 0:  # Up
            new_pos = (max(row - 1, 0), col)
        elif action == 1:  # Down
            new_pos = (min(row + 1, GRID_SIZE - 1), col)
        elif action == 2:  # Left
            new_pos = (row, max(col - 1, 0))
        elif action == 3:  # Right
            new_pos = (row, min(col + 1, GRID_SIZE - 1))
        
        # Check for obstacles
        if new_pos in OBSTACLES:
            new_pos = self.agent_pos  # Stay in place if hitting obstacle
        
        self.agent_pos = new_pos
        new_state = self.coordinates_to_state(new_pos)
        
        reward = 1 if self.is_terminal(new_state) else -0.01
        return new_state, reward, self.is_terminal(new_state)

    def train(self, episodes):
        for _ in range(episodes):
            state = self.reset()
            done = False
            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(range(self.action_space))  # Explore
                else:
                    action = np.argmax(self.q_table[state])  # Exploit
                
                next_state, reward, done = self.step(action)
                
                # Update Q-value
                self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[state, action])
                
                state = next_state

# Training the agent
env = GridWorld()
env.train(1000)

# Testing the learned policy
def test_policy(env):
    state = env.reset()
    path = [state]
    done = False
    while not done:
        action = np.argmax(env.q_table[state])
        state, reward, done = env.step(action)
        path.append(state)
    return path

def visualize_grid(path):
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Mark obstacles
    for obs in OBSTACLES:
        grid[obs] = -1  # Obstacles as -1
    
    # Mark goal
    grid[GOAL] = 1  # Goal as 1
    
    # Mark agent's path
    for step in path:
        coords = env.state_to_coordinates(step)
        grid[coords] = 0.5  # Path as 0.5
    
    # Plotting the grid
    plt.imshow(grid, cmap='RdYlGn', interpolation='nearest')
    plt.title('GridWorld Environment')
    plt.colorbar(ticks=[-1, 0, 0.5, 1], label='Values')
    plt.xticks(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
    plt.yticks(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
    plt.grid(False)
    plt.show()

# Testing and visualizing the learned policy
test_path = test_policy(env)
print("Test Path (State Sequence):", test_path)
visualize_grid(test_path)
