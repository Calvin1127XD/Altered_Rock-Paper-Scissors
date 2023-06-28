import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

# Define the neural network for the Q-Learning agent
# Define the neural network for the Q-Learning agent
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load the models from the saved state dictionaries
player_a_model = QNetwork(1, 256, 128, 32, 2)
player_a_model.load_state_dict(torch.load('player_a_model.pth'))

player_b_model = QNetwork(1, 256, 128, 32, 2)
player_b_model.load_state_dict(torch.load('player_b_model.pth'))


# Load images
rock_img = Image.open("rock.png")
paper_img = Image.open("paper.png")
scissors_img = Image.open("scissors.png")

# Load the final average strategies
player_a_avg_prob = np.load('player_a_avg_prob.npy')
player_b_avg_prob = np.load('player_b_avg_prob.npy')

# Define the altered Rock Paper Scissors game
def get_result(player_a_action, player_b_action):
    if (player_a_action == 0 and player_b_action == 0):
        return "D"  # Draw
    elif (player_a_action == 0 and player_b_action == 1) or (player_a_action == 1 and player_b_action == 0):
        return "A"  # Player A wins
    elif (player_a_action == 1 and player_b_action == 1):
        return "B"  # Player B wins

# Function to map action integers to images
def action_image(player, action):
    if player == "A":
        if action == 0:
            return rock_img
        else:
            return paper_img
    elif player == "B":  # player == 'B'
        if action == 0:
            return rock_img
        else:
            return scissors_img

# Initialize the plot
fig, ax = plt.subplots()
ax.set_xlim(0, 2)
ax.set_ylim(-1, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Altered Rock-Paper-Scissors Animation")

# Initialize counters for player wins
player_a_wins = 0
player_b_wins = 0

# Function to update the plot for each game
def update(game):
    global player_a_wins, player_b_wins
    ax.clear()
    ax.set_xlim(0, 2)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Round {game + 1}")

    # Select actions for player A and B based on their final average strategies
    player_a_action = np.random.choice([0, 1], p=player_a_avg_prob)
    player_b_action = np.random.choice([0, 1], p=player_b_avg_prob)

    # Display the actions as images
    ax.imshow(action_image('A', player_a_action), extent=(0, 1, -0.5, 0.5))
    ax.imshow(action_image('B', player_b_action), extent=(1, 2, -0.5, 0.5))

    # Determine the winner and display the result
    result = get_result(player_a_action, player_b_action)
    if result == "D":
        ax.text(1, 0.7, "Draw", ha='center', va='center', fontsize=20)
    elif result == "A":
        ax.text(1, 0.7, "Player A wins", ha='center', va='center', fontsize=20)
        player_a_wins += 1
    elif result == "B":
        ax.text(1, 0.7, "Player B wins", ha='center', va='center', fontsize=20)
        player_b_wins += 1

    # Display the cumulative wins
    ax.text(0.5, 0.9, f"Player A wins: {player_a_wins}", ha='center', va='center', fontsize=10)
    ax.text(1.5, 0.9, f"Player B wins: {player_b_wins}", ha='center', va='center', fontsize=10)


# Create the animation
ani = FuncAnimation(fig, update, frames=range(1000), interval=750, repeat=False)

# Display the animation
plt.show()
