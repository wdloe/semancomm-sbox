import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

# 1️⃣ Import CIFAR-10 Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class SemanticEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoded_dim):
        super(SemanticEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, encoded_dim)
        )

    def forward(self, x):
        return self.model(x)

class Channel(nn.Module):
    def __init__(self, noise_std=0.1):
        super(Channel, self).__init__()
        self.noise_std = noise_std

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

class SemanticDecoder(nn.Module):
    def __init__(self, encoded_dim, hidden_dim, output_dim):
        super(SemanticDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


## 2️⃣ Adjust Encoder and Decoder for CIFAR Input
input_dim = 32 * 32 * 3  # CIFAR images are 32x32 with 3 color channels
hidden_dim = 256
encoded_dim = 128

# 3️⃣ Initialize Components
encoder = SemanticEncoder(input_dim, hidden_dim, encoded_dim)
channel = Channel(noise_std=0.1)
decoder = SemanticDecoder(encoded_dim, hidden_dim, input_dim)

# Move models to GPU (cuda:0)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
channel = channel.to(device)
decoder = decoder.to(device)

# Training parameters
epochs = 10
learning_rate = 0.001
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Training loop
losses = []
for epoch in range(epochs):
    epoch_loss = 0.0
    for images, _ in train_loader:
        # Prepare data
        x = images.view(-1, input_dim).to(device)

        # Forward pass
        encoded = encoder(x)
        transmitted = channel(encoded)
        recovered = decoder(transmitted)

        # Compute loss
        loss = mse_loss(recovered, x)
        epoch_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss}")
    losses.append(avg_loss)

# Evaluate the results after training
for images, _ in train_loader:
    x = images.view(-1, input_dim).to(device)
    encoded = encoder(x)
    transmitted = channel(encoded)
    recovered = decoder(transmitted)
    break

# Move back to CPU for visualization
x = x.cpu()
recovered = recovered.cpu()

# 5️⃣ Evaluation Metrics Update
def evaluate_metrics(original, recovered):
    # Mean Squared Error
    mse = mse_loss(original, recovered)
    # Signal-to-Noise Ratio (SNR)
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - recovered) ** 2)
    snr = 10 * torch.log10(signal_power / noise_power)
    # Print the metrics
    print("\n=== Evaluation Metrics ===")
    print(f"Mean Squared Error (MSE): {mse.item()}")
    print(f"Signal-to-Noise Ratio (SNR): {snr.item()} dB")

evaluate_metrics(x, recovered)

# 6️⃣ Optional: Visualize Original vs. Reconstructed Images
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(x[0].view(32, 32, 3).detach().numpy() * 0.5 + 0.5)
plt.subplot(1, 2, 2)
plt.title("Reconstructed")
plt.imshow(recovered[0].view(32, 32, 3).detach().numpy() * 0.5 + 0.5)

# Create results directory if it doesn't exist
import os
os.makedirs("results", exist_ok=True)

# Save the plot
plt.savefig("results/cifar10_original_vs_reconstructed.png")
print("Plot saved to results/cifar10_original_vs_reconstructed.png")
plt.show()
# Plot training loss curve
plt.figure()
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid()
plt.legend()

# Create results directory if it does not exist
os.makedirs("results", exist_ok=True)

# Save the plot to the results folder
plt.savefig("results/cifar10_training_loss.png")
print("Training Loss plot saved to results/cifar10_training_loss.png")
plt.show()

# Visualize the intermediate steps: Encoder, Channel, Decoder
import matplotlib.pyplot as plt

# Sample a batch for visualization
for images, _ in train_loader:
    x = images.view(-1, input_dim).to(device)
    encoded = encoder(x)
    transmitted = channel(encoded)
    recovered = decoder(transmitted)
    break

# Move back to CPU for visualization
x = x.cpu()
encoded = encoded.cpu()
transmitted = transmitted.cpu()
recovered = recovered.cpu()

# Visualization
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Original Image
axes[0].imshow(x[0].view(32, 32, 3).detach().numpy() * 0.5 + 0.5)
axes[0].set_title("Original Image")

# Encoded Representation (Flattened)
axes[1].plot(encoded[0].detach().numpy())
axes[1].set_title("Encoded Representation")

# Transmitted (Channel Output)
axes[2].plot(transmitted[0].detach().numpy())
axes[2].set_title("Channel Output")

# Reconstructed Image
axes[3].imshow(recovered[0].view(32, 32, 3).detach().numpy() * 0.5 + 0.5)
axes[3].set_title("Reconstructed Image")

# Save the visualization
os.makedirs("results", exist_ok=True)
plt.savefig("results/cifar10_intermediate_steps.png")
print("Intermediate visualization saved to results/cifar10_intermediate_steps.png")
plt.show()