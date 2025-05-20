import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

# 1️⃣ Import STL-10 Dataset and ResNet
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Data transformations
transform = transforms.Compose([
    transforms.Resize((96, 96)),  # STL-10 is 96x96
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load STL-10 dataset
train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class SemanticEncoder(nn.Module):
    def __init__(self, encoded_dim):
        super(SemanticEncoder, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final classification layer
        self.fc = nn.Linear(resnet.fc.in_features, encoded_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Channel(nn.Module):
    def __init__(self, noise_std=0.1):
        super(Channel, self).__init__()
        self.noise_std = noise_std

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

class SemanticDecoder(nn.Module):
    def __init__(self, encoded_dim, output_channels=3):
        super(SemanticDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(encoded_dim, 4608),   # 512 * 3 * 3 = 4608
            nn.ReLU(),
            nn.Unflatten(1, (512, 3, 3)),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # -> 6x6
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # -> 12x12
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # -> 24x24
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # -> 48x48
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1),  # -> 96x96
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x


input_dim = 96 * 96 * 3  # STL-10 Image size
encoded_dim = 2048  # Increase to allow richer semantic representation

encoder = SemanticEncoder(encoded_dim)
channel = Channel(noise_std=0.1)

# --- Updated SemanticDecoder with skip connections (U-Net style) ---
class SemanticDecoder(nn.Module):
    def __init__(self, encoded_dim, output_channels=3):
        super(SemanticDecoder, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(encoded_dim, 4608),   # 512 * 3 * 3 = 4608
            nn.ReLU()
        )

        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # -> 6x6
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # -> 12x12
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # -> 24x24
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # -> 48x48
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1),  # -> 96x96
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 3, 3)
        x = self.decoder(x)
        return x

decoder = SemanticDecoder(encoded_dim, output_channels=3)

# Move models to GPU (cuda:1)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
channel = channel.to(device)
decoder = decoder.to(device)

# Training parameters
epochs = 50
learning_rate = 0.001
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate, weight_decay=1e-5)

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
        x = images.to(device)

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
    
    # Validation Loop
    val_loss = 0.0
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for images, _ in val_loader:
            x = images.to(device)
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
# Perceptual Loss Placeholder
# from torchvision.models import vgg16
# vgg = vgg16(pretrained=True).features[:8].to(device)
#
# def perceptual_loss(output, target):
#     return mse_loss(vgg(output), vgg(target))