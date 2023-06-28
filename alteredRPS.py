import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

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


# Define the altered Rock Paper Scissors game
def altered_rps(player_a_action, player_b_action):
    if (player_a_action == 0 and player_b_action == 0):
        return 0, 0  # Draw
    elif (player_a_action == 0 and player_b_action == 1) or (player_a_action == 1 and player_b_action == 0):
        return 1, -1  # Player A wins
    elif (player_a_action == 1 and player_b_action == 1):
        return -1, 1  # Player B wins

# Q-Learning hyperparameters
alpha = 0.1
gamma = 0.95
epsilon = 0.5
min_epsilon=0.006
decay_rate=0.9999
num_epochs = 20000
num_games_per_epoch = 30

# Initialize neural networks and optimizers
player_a_network = QNetwork(1, 256, 128, 32, 2)
player_b_network = QNetwork(1, 256, 128, 32, 2)
player_a_optimizer = optim.Adam(player_a_network.parameters(), lr=alpha)
player_b_optimizer = optim.Adam(player_b_network.parameters(), lr=alpha)

player_a_strategies = []
player_b_strategies = []

# Training loop
for epoch in range(num_epochs):
    player_a_score = 0
    player_b_score = 0
    player_a_counts = [0, 0]
    player_b_counts = [0, 0]

    for game in range(num_games_per_epoch):
        epsilon = max(min_epsilon,epsilon*decay_rate)
        # Select actions for player A and B using epsilon-greedy exploration
        if random.random() < epsilon:
            player_a_action = random.randint(0, 1)
        else:
            player_a_action = torch.argmax(player_a_network(torch.tensor([0.0]))).item()

        if random.random() < epsilon:
            player_b_action = random.randint(0, 1)
        else:
            player_b_action = torch.argmax(player_b_network(torch.tensor([0.0]))).item()

        # Update counts
        player_a_counts[player_a_action] += 1
        player_b_counts[player_b_action] += 1

        # Play the game and get rewards
        reward_a, reward_b = altered_rps(player_a_action, player_b_action)
        player_a_score += reward_a
        player_b_score += reward_b

        # Update Q-Networks
        player_a_optimizer.zero_grad()
        player_b_optimizer.zero_grad()

        q_value_a = player_a_network(torch.tensor([0.0]))[player_a_action]
        q_value_b = player_b_network(torch.tensor([0.0]))[player_b_action]

        target_a = reward_a + gamma * torch.max(player_a_network(torch.tensor([0.0])))
        target_b = reward_b + gamma * torch.max(player_b_network(torch.tensor([0.0])))

        loss_a = torch.square(q_value_a - target_a).mean()
        loss_b = torch.square(q_value_b - target_b).mean()

        loss_a.backward()
        loss_b.backward()

        player_a_optimizer.step()
        player_b_optimizer.step()

    # Calculate probability vector for each player
    player_a_prob = np.array(player_a_counts) / num_games_per_epoch
    player_b_prob = np.array(player_b_counts) / num_games_per_epoch

    player_a_strategies.append(player_a_prob)
    player_b_strategies.append(player_b_prob)


    # Print the results for the current epoch
    print(f"Epoch {epoch + 1}/{num_epochs} - Player A score: {player_a_score} Player B score: {player_b_score}")
    print(f"Player A strategy: [rock: {player_a_prob[0]}, paper: {player_a_prob[1]}]")
    print(f"Player B strategy: [rock: {player_b_prob[0]}, scissors: {player_b_prob[1]}]")


# Calculate the final average strategies for the second half of the epochs
second_half_epochs = len(player_a_strategies) // 2
player_a_avg_prob = np.mean(np.array(player_a_strategies)[second_half_epochs:], axis=0)
player_b_avg_prob = np.mean(np.array(player_b_strategies)[second_half_epochs:], axis=0)


# Print final strategies
print(f"Final Player A strategy: [rock: {player_a_avg_prob[0]}, paper: {player_a_avg_prob[1]}]")
print(f"Final Player B strategy: [rock: {player_b_avg_prob[0]}, scissors: {player_b_avg_prob[1]}]")

# torch.save(player_a_network.state_dict(), 'player_a_model.pth')
# torch.save(player_b_network.state_dict(), 'player_b_model.pth')

#np.save('player_a_avg_prob.npy', player_a_avg_prob)
#np.save('player_b_avg_prob.npy', player_b_avg_prob)

