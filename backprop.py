import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feed-forward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Parameter(torch.tensor([1.2, 0.15]))   # First fully connected layer
        self.fc2 = nn.Parameter(torch.tensor([1.2, 0.15]))   # Second fully connected layer

    def forward(self, x):
        x2 = torch.empty(0)
        x1 = self.fc1[0] * x + self.fc1[1]

        for x0 in x:
            x2 = torch.cat((x2, (self.fc2[0] * x0 + self.fc2[1]).unsqueeze(0)), 0)
            break
        return x1, x2

# Initialize the network, loss function, and optimizer
model = SimpleNN()

# Loss function: Mean Squared Error (MSE)
criterion = nn.MSELoss()

# Optimizer: Adam
optimizer = optim.Adam(model.parameters(), lr=0.1)


# Dummy data: inputs (10 features) and target (single output)
inputs = torch.tensor([4,5,6,7,8,9,10,11,12,13,14,15,16,17], dtype=float)  # Batch of 64 with 10 features
targets = torch.tensor([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36], dtype=float)   # Batch of 64 with 1 target value

for _ in range(20):
# Forward pass: Compute predicted outputs by passing inputs through the network
    outputs1, outputs2 = model(inputs)

    # Compute the loss between predicted outputs and actual targets
    loss1 = criterion(outputs1, targets)
    loss2 = criterion(outputs2, targets[0])

    # Backward pass: Compute gradients of the loss with respect to model parameters
    optimizer.zero_grad()  # Clear previous gradients
    loss1.backward()        # Perform backpropagation (compute gradients)
    loss2.backward()        # Perform backpropagation (compute gradients)

    # Update the weights using the optimizer
    optimizer.step()

# Print the loss
print(f'Loss: {loss1.item()}')
