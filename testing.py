import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from torch.nn.functional import mse_loss
import argparse
import numpy as np

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
    transforms.Resize((96, 96)),  # STL-10 is 96x96
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load STL-10 dataset
val_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Model definitions
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


# Initialize models and load checkpoints
input_dim = 96 * 96 * 3  # STL-10 Image size
encoded_dim = 2048

encoder = SemanticEncoder(encoded_dim)
channel = Channel(noise_std=args.noise_std)
decoder = SemanticDecoder(encoded_dim, output_channels=3)

encoder = encoder.to(device)
channel = channel.to(device)
decoder = decoder.to(device)

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
    x = images.to(device)

    # Semantic Communication
    encoded = encoder(x)
    transmitted = channel(encoded)
    recovered = decoder(transmitted)

    mse_semantic = mse_loss(recovered, x).item()
    mse_traditional = mse_loss(traditional_communication(images).to(device), x).item()

    plt.figure(figsize=(20, 5))

    # Original Image
    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(images[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)

    # Encoder Output (Improved Visualization)
    plt.subplot(1, 4, 2)
    plt.title(f"Encoder Output\n{encoded.numel() * 32} bits")
    encoded_np = encoded.detach().cpu().numpy()
    encoded_flat = encoded_np.reshape(-1)
    enc_len = encoded_flat.shape[0]
    enc_side = int(enc_len ** 0.5)
    if enc_side * enc_side == enc_len:
        enc_vis = encoded_flat.reshape(enc_side, enc_side)
    else:
        enc_width = 64
        enc_height = (enc_len + enc_width - 1) // enc_width
        padded_enc = np.zeros(enc_height * enc_width, dtype=encoded_flat.dtype)
        padded_enc[:enc_len] = encoded_flat
        enc_vis = padded_enc.reshape(enc_height, enc_width)
    plt.imshow(enc_vis, cmap='gray', aspect='auto')

    # Channel Output (Improved Visualization)
    plt.subplot(1, 4, 3)
    plt.title(f"Channel Output\n{transmitted.numel() * 32} bits")
    transmitted_np = transmitted.detach().cpu().numpy()
    transmitted_flat = transmitted_np.reshape(-1)
    trans_len = transmitted_flat.shape[0]
    trans_side = int(trans_len ** 0.5)
    if trans_side * trans_side == trans_len:
        trans_vis = transmitted_flat.reshape(trans_side, trans_side)
    else:
        trans_width = 64
        trans_height = (trans_len + trans_width - 1) // trans_width
        padded_trans = np.zeros(trans_height * trans_width, dtype=transmitted_flat.dtype)
        padded_trans[:trans_len] = transmitted_flat
        trans_vis = padded_trans.reshape(trans_height, trans_width)
    plt.imshow(trans_vis, cmap='gray', aspect='auto')

    # Decoder Output (Corrected Reshaping)
    plt.subplot(1, 4, 4)
    plt.title(f"Decoder Output\nMSE: {mse_semantic:.4f}")
    recovered_image = recovered[0].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5
    plt.imshow(recovered_image)

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
    plt.imshow(recovered_image)

    # Traditional Recovered
    plt.subplot(1, 3, 3)
    plt.title(f"Traditional Recovered\nMSE: {mse_traditional:.4f} | Bits: {input_dim * 32}")
    plt.imshow(traditional_communication(images)[0].permute(1, 2, 0).cpu().numpy())

    # Save the figure
    plt.savefig(f"results/comparison_with_mse.png")
    print("Comparison with MSE saved to results/comparison_with_mse.png")
    plt.show()
    break