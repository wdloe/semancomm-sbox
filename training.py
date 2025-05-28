import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import torch.nn.functional as F
from torch.nn import Module
import math

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

# Discriminator for Adversarial Loss
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass for the discriminator.
        Determines if the input is real or generated.
        """
        return self.model(x).view(-1)

class DiffusionUNet(Module):
    def __init__(self, cond_dim, channels=3):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, 256)

        self.net = nn.Sequential(
            nn.Conv2d(channels + 1 + 256, 64, 3, padding=1),  # input + timestep + conditioning
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, padding=1)
        )

    def forward(self, x, t, cond):
        b, c, h, w = x.shape
        t_embed = t[:, None, None, None].float() / 1000  # normalize
        t_embed = t_embed.expand(-1, 1, h, w)
        cond_embed = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1).expand(-1, 256, h, w)
        x_cat = torch.cat([x, t_embed, cond_embed], dim=1)
        return self.net(x_cat)

input_dim = 96 * 96 * 3  # STL-10 Image size
encoded_dim = 2048  # Increase to allow richer semantic representation

encoder = SemanticEncoder(encoded_dim)
channel = Channel(noise_std=0.1)
decoder = SemanticDecoder(encoded_dim, output_channels=3)

# Move models to GPU (cuda:1)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

encoder = encoder.to(device)
channel = channel.to(device)
decoder = decoder.to(device)

# Load VGG16 for perceptual loss (feature-based loss)
vgg = vgg16(pretrained=True).features[:8].to(device)
vgg.eval()  # Set to evaluation mode

# Define Perceptual Loss
def perceptual_loss(output, target):
    """
    Compute perceptual loss based on VGG feature maps.
    This loss focuses on semantic differences instead of pixel differences.
    """
    output_features = vgg(output)
    target_features = vgg(target)
    return F.mse_loss(output_features, target_features)

# Initialize Discriminator and optimizer
discriminator = Discriminator().to(device)
learning_rate = 2e-4
epochs = 200
patience = 15
disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training parameters
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate, weight_decay=1e-5)
encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

timesteps = 1000
betas = linear_beta_schedule(timesteps).to(device)
alphas = 1. - betas
alpha_hat = torch.cumprod(alphas, dim=0)

diffusion_decoder = DiffusionUNet(encoded_dim).to(device)
diff_optimizer = torch.optim.Adam(diffusion_decoder.parameters(), lr=learning_rate)
diff_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(diff_optimizer, mode='min', factor=0.5, patience=5)

# Early Stopping Parameters
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
    discriminator.train()
    
    for images, _ in train_loader:
        # Prepare data
        x = images.to(device)

        # Forward pass
        encoded = encoder(x)
        transmitted = channel(encoded)

        # # Original decoder training (temporarily disabled)
        # recovered = decoder(transmitted)
        # loss = perceptual_loss(recovered, x)

        # # Backpropagation and optimization for encoder and decoder
        # optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        # optimizer.step()

        # Diffusion training step
        noise = torch.randn_like(x)
        t = torch.randint(0, timesteps, (x.size(0),), device=device)
        alpha_t = alpha_hat[t].view(-1, 1, 1, 1)
        noisy_img = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

        pred_noise = diffusion_decoder(noisy_img, t, encoded)

        # Generate x_gen for perceptual supervision
        with torch.no_grad():
            x_gen = noisy_img.clone()
            for i in reversed(range(10)):  # quick reverse loop for supervision
                t_tmp = torch.full((x.size(0),), i, device=device, dtype=torch.long)
                alpha_t_tmp = alpha_hat[t_tmp].view(-1, 1, 1, 1)
                beta_t_tmp = betas[t_tmp].view(-1, 1, 1, 1)
                pred_noise_tmp = diffusion_decoder(x_gen, t_tmp, encoded)
                x_gen = (1 / torch.sqrt(alphas[t_tmp]).view(-1, 1, 1, 1)) * (
                    x_gen - beta_t_tmp / torch.sqrt(1 - alpha_hat[t_tmp]).view(-1, 1, 1, 1) * pred_noise_tmp
                )
                if i > 0:
                    x_gen += torch.sqrt(beta_t_tmp) * torch.randn_like(x_gen)

        perceptual = perceptual_loss(x_gen, x)
        diff_loss = F.mse_loss(pred_noise, noise) + 0.1 * perceptual  # weighted perceptual supervision

        diff_optimizer.zero_grad()
        diff_loss.backward()
        diff_optimizer.step()

        # # Train the Discriminator (temporarily disabled)
        # disc_real = discriminator(x)
        # disc_fake = discriminator(recovered.detach())
        # real_labels = torch.ones_like(disc_real, device=device)
        # fake_labels = torch.zeros_like(disc_fake, device=device)

        # real_loss = F.binary_cross_entropy(disc_real, real_labels)
        # fake_loss = F.binary_cross_entropy(disc_fake, fake_labels)
        # disc_loss = (real_loss + fake_loss) / 2
        # disc_optimizer.zero_grad()
        # disc_loss.backward()
        # disc_optimizer.step()

        # Print losses

        epoch_loss += diff_loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    
    # Validation Loop
    val_loss = 0.0
    encoder.eval()
    decoder.eval()
    discriminator.eval()
    diffusion_decoder.eval()
    with torch.no_grad():
        for images, _ in val_loader:
            x = images.to(device)
            encoded = encoder(x)
            transmitted = channel(encoded)
            # recovered = decoder(transmitted)
            # val_loss += perceptual_loss(recovered, x).item()
            # Diffusion validation step
            noise = torch.randn_like(x)
            t = torch.randint(0, timesteps, (x.size(0),), device=device)
            alpha_t = alpha_hat[t].view(-1, 1, 1, 1)
            noisy_img = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

            pred_noise = diffusion_decoder(noisy_img, t, encoded)

            # Reconstruct x_gen for perceptual validation loss
            x_gen = noisy_img.clone()
            for i in reversed(range(10)):
                t_tmp = torch.full((x.size(0),), i, device=device, dtype=torch.long)
                alpha_t_tmp = alpha_hat[t_tmp].view(-1, 1, 1, 1)
                beta_t_tmp = betas[t_tmp].view(-1, 1, 1, 1)
                pred_noise_tmp = diffusion_decoder(x_gen, t_tmp, encoded)
                x_gen = (1 / torch.sqrt(alphas[t_tmp]).view(-1, 1, 1, 1)) * (
                    x_gen - beta_t_tmp / torch.sqrt(1 - alpha_hat[t_tmp]).view(-1, 1, 1, 1) * pred_noise_tmp
                )
                if i > 0:
                    x_gen += torch.sqrt(beta_t_tmp) * torch.randn_like(x_gen)

            perceptual_val = perceptual_loss(x_gen, x)
            val_loss += F.mse_loss(pred_noise, noise).item() + 0.1 * perceptual_val.item()
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    encoder_scheduler.step(avg_val_loss)
    diff_scheduler.step(avg_val_loss)

    # Early Stopping Logic
    print(f"Epoch [{epoch+1}/{epochs}] Training Loss: {avg_loss:.5f}, Validation Loss: {avg_val_loss:.5f}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(encoder.state_dict(), "checkpoints/best_encoder.pth")
        torch.save(decoder.state_dict(), "checkpoints/best_decoder.pth")
        torch.save(discriminator.state_dict(), "checkpoints/best_discriminator.pth")
        torch.save(diffusion_decoder.state_dict(), "checkpoints/best_diffusion_decoder.pth")
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
torch.save(discriminator.state_dict(), "checkpoints/last_discriminator.pth")
torch.save(diffusion_decoder.state_dict(), "checkpoints/last_diffusion_decoder.pth")
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