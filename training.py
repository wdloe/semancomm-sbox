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
# Load CIFAR-10 Validation Set
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
epochs = 50
learning_rate = 0.001
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

# Early Stopping Parameters
patience = 5
best_val_loss = float('inf')
trigger_times = 0

# Make sure checkpoint directory exists
import os
os.makedirs("checkpoints", exist_ok=True)

# Training loop with validation and early stopping
losses = []
val_losses = []
for epoch in range(epochs):
    epoch_loss = 0.0
    encoder.train()
    decoder.train()
    
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
    losses.append(avg_loss)
    
    # 🔍 Validation Loop
    val_loss = 0.0
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for images, _ in val_loader:
            x = images.view(-1, input_dim).to(device)
            encoded = encoder(x)
            transmitted = channel(encoded)
            recovered = decoder(transmitted)
            val_loss += mse_loss(recovered, x).item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Early Stopping Logic
    print(f"Epoch [{epoch+1}/{epochs}] Training Loss: {avg_loss}, Validation Loss: {avg_val_loss}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(encoder.state_dict(), "checkpoints/best_encoder.pth")
        torch.save(decoder.state_dict(), "checkpoints/best_decoder.pth")
        print(f"Validation loss improved. Saving model.")
        trigger_times = 0
    else:
        trigger_times += 1
        print(f"No improvement for {trigger_times} epochs.")
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# Save the final models
torch.save(encoder.state_dict(), "checkpoints/last_encoder.pth")
torch.save(decoder.state_dict(), "checkpoints/last_decoder.pth")
print("Final models saved to 'checkpoints/'")

print("\nTraining completed. Models are saved and ready for testing.")

# Plot Training and Validation Loss
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)

# Ensure the results directory exists
os.makedirs("results", exist_ok=True)

# Save the plot as PNG
plt.savefig("results/training_validation_loss.png")
print("Loss plot saved as 'results/training_validation_loss.png'")
plt.show()