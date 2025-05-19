import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from torch.nn.functional import mse_loss
import argparse

# Argument parser for noise configuration and image index
parser = argparse.ArgumentParser(description='Semantic Communication Testing')
parser.add_argument('--index', type=int, default=0, help='Index of the image to test (default: 0)')
parser.add_argument('--noise_std', type=float, default=0.1, help='Noise standard deviation for both semantic and traditional communication (default: 0.1)')
args = parser.parse_args()

# Ensure directories exist
os.makedirs("results", exist_ok=True)

# Device configuration
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 Validation Set
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Model definitions
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

# Initialize models and load checkpoints
input_dim = 32 * 32 * 3
hidden_dim = 256
encoded_dim = 128

encoder = SemanticEncoder(input_dim, hidden_dim, encoded_dim).to(device)
channel = Channel(noise_std=args.noise_std).to(device)
print(f"Communication Noise STD: {args.noise_std}")
decoder = SemanticDecoder(encoded_dim, hidden_dim, input_dim).to(device)

encoder.load_state_dict(torch.load("checkpoints/best_encoder.pth"))
decoder.load_state_dict(torch.load("checkpoints/best_decoder.pth"))

# Traditional Communication Simulation
def traditional_communication(x):
    # Quantization (8-bit per pixel simulation)
    x_quantized = (x * 255).clamp(0, 255).byte()
    # Transmission with modular noise level
    x_transmitted = x_quantized.float() + torch.randn_like(x_quantized.float()) * args.noise_std
    # Dequantization
    x_recovered = (x_transmitted / 255.0).clamp(0, 1)
    print(f"Traditional Communication Noise STD: {args.noise_std}")
    return x_recovered

# Inference and Visualization of Each Step
for idx, (images, _) in enumerate(val_loader):
    if idx != args.index:
        continue
    x = images.view(-1, input_dim).to(device)

    # Semantic Communication
    encoded = encoder(x)
    transmitted = channel(encoded)
    recovered = decoder(transmitted)

    mse_semantic = mse_loss(recovered, x).item()
    mse_traditional = mse_loss(traditional_communication(images).view(-1, input_dim), x.cpu()).item()

    plt.figure(figsize=(20, 5))

    # Original Image
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(images[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)

    # Encoder Output (Flattened)
    plt.subplot(1, 4, 2)
    plt.title(f"Encoder Output\n{encoded.numel() * 32} bits")
    plt.imshow(encoded.view(8, 16).detach().cpu().numpy(), cmap='gray')

    # Channel Output (Flattened)
    plt.subplot(1, 4, 3)
    plt.title(f"Channel Output\n{transmitted.numel() * 32} bits")
    plt.imshow(transmitted.view(8, 16).detach().cpu().numpy(), cmap='gray')

    # Decoder Output (Corrected Reshaping)
    plt.subplot(1, 4, 4)
    plt.title(f"Decoder Output\nMSE: {mse_semantic:.4f}")
    # Fix the reshaping logic to properly map back to 32x32x3
    recovered_image = recovered.view(-1, 3, 32, 32).permute(0, 2, 3, 1)
    plt.imshow(recovered_image[0].detach().cpu().numpy() * 0.5 + 0.5)

    # Save the figure
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/semantic_step_by_step.png")
    print("Step-by-step semantic communication saved to results/semantic_step_by_step.png")
    plt.show()

    plt.figure(figsize=(15, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(images[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)

    # Semantic Recovered
    plt.subplot(1, 3, 2)
    plt.title(f"Semantic Recovered\nMSE: {mse_semantic:.4f} | Bits: {encoded.numel() * 32}")
    recovered_image = recovered.view(-1, 3, 32, 32).permute(0, 2, 3, 1)
    plt.imshow(recovered_image[0].detach().cpu().numpy() * 0.5 + 0.5)

    # Traditional Recovered
    plt.subplot(1, 3, 3)
    plt.title(f"Traditional Recovered\nMSE: {mse_traditional:.4f} | Bits: {x.numel() * 32}")
    plt.imshow(traditional_communication(images)[0].permute(1, 2, 0).cpu().numpy())

    # Save the figure
    plt.savefig(f"results/comparison_with_mse.png")
    print("Comparison with MSE saved to results/comparison_with_mse.png")
    plt.show()
    break