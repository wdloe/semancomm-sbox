import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import math
from torchvision.utils import save_image
# --- Added for annotated image saving
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont
import torchvision.utils as vutils

# -------------------------------------------------------------
# Dataset loader (STL‑10 → 96×96 crops)
from torchvision.datasets import STL10
from torchvision import transforms


def get_dataloader(batch_size: int = 64, split: str = 'train') -> DataLoader:
    tfm = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
    ])
    ds = STL10(root='./data', split=split, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'),
                      num_workers=4, pin_memory=True)

# -------------------------------------------------------------
# Metric: PSNR
def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse.item()))

# -------------------------------------------------------------
# VQ‑VAE components (encoder, quantiser, decoder)
class Encoder(torch.nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 128, z_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, hidden, 4, 2, 1),  # 96→48
            torch.nn.GroupNorm(32, hidden),
            torch.nn.SiLU(),

            torch.nn.Conv2d(hidden, hidden, 3, 1, 1),
            torch.nn.GroupNorm(32, hidden),
            torch.nn.SiLU(),

            torch.nn.Conv2d(hidden, hidden * 2, 4, 2, 1),  # 48→24
            torch.nn.GroupNorm(32, hidden * 2),
            torch.nn.SiLU(),

            torch.nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1),
            torch.nn.GroupNorm(32, hidden * 2),
            torch.nn.SiLU(),

            torch.nn.Conv2d(hidden * 2, hidden * 4, 4, 2, 1),  # 24→12
            torch.nn.GroupNorm(32, hidden * 4),
            torch.nn.SiLU(),

            torch.nn.Conv2d(hidden * 4, z_dim, 1),  # 12×12 bottleneck
        )

    def forward(self, x):
        return self.net(x)  # (B, z_dim, 12, 12)


class VectorQuantizerEMA(torch.nn.Module):
    """EMA codebook as in VQ‑VAE‑2."""

    def __init__(self, num_embed: int = 1024, dim: int = 256, beta: float = 0.25,
                 decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_embed, self.dim, self.beta, self.decay, self.eps = (
            num_embed, dim, beta, decay, eps)

        embed = torch.randn(dim, num_embed)
        self.register_buffer('embedding', embed)
        self.register_buffer('cluster_size', torch.zeros(num_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z.view(-1, self.dim)

        # squared L2 distance to each codebook vector
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding
            + self.embedding.pow(2).sum(0, keepdim=True)
        )
        indices = torch.argmin(dist, dim=1)
        enc_onehot = torch.nn.functional.one_hot(indices, self.num_embed).type(flat.dtype)
        z_q = enc_onehot @ self.embedding.t()
        z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        if self.training:
            with torch.no_grad():
                enc_onehot_sum = enc_onehot.sum(0)
                self.cluster_size.mul_(self.decay).add_(enc_onehot_sum, alpha=1 - self.decay)

                embed_sum = flat.t() @ enc_onehot
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                n = self.cluster_size.sum()
                cluster_size = ((self.cluster_size + self.eps) / (n + self.num_embed * self.eps)) * n
                self.embedding.copy_(self.embed_avg / cluster_size.unsqueeze(0))

        # losses
        q_loss = self.beta * torch.mean((z_q.detach() - z_e) ** 2)
        commit_loss = torch.mean((z_q - z_e.detach()) ** 2)
        loss = q_loss + commit_loss

        # straight‑through
        z_q = z_e + (z_q - z_e).detach()
        return z_q, loss, indices.view(B, H, W)


class Decoder(torch.nn.Module):
    def __init__(self, out_ch: int = 3, hidden: int = 128, z_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(z_dim, hidden * 4, 3, 1, 1),
            torch.nn.SiLU(),

            torch.nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, 2, 1),  # 12→24
            torch.nn.SiLU(),
            torch.nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1),
            torch.nn.SiLU(),

            torch.nn.ConvTranspose2d(hidden * 2, hidden, 4, 2, 1),  # 24→48
            torch.nn.SiLU(),
            torch.nn.Conv2d(hidden, hidden, 3, 1, 1),
            torch.nn.SiLU(),

            torch.nn.ConvTranspose2d(hidden, out_ch, 4, 2, 1),  # 48→96
            torch.nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class VQVAE(torch.nn.Module):
    def __init__(self, z_dim: int = 256, num_embed: int = 1024):
        super().__init__()
        self.encoder = Encoder(z_dim=z_dim)
        self.quantizer = VectorQuantizerEMA(num_embed=num_embed, dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, q_loss, indices = self.quantizer(z_e)
        x_rec = self.decoder(z_q)
        rec_loss = torch.mean((x - x_rec) ** 2)
        return x_rec, rec_loss, q_loss, indices

# -------------------------------------------------------------
# QPSK modulation / demodulation helpers (2 bits per symbol)

def bits_to_symbols(bits: torch.Tensor) -> torch.Tensor:
    bits = bits.view(-1, 2)
    mapping = torch.tensor([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], device=bits.device)
    idx = bits[:, 0] * 2 + bits[:, 1]
    return mapping[idx]


def symbols_to_bits(sym: torch.Tensor) -> torch.Tensor:
    real = (sym.real < 0).long()
    imag = (sym.imag < 0).long()
    return torch.stack([real, imag], dim=1).flatten()

# -------------------------------------------------------------
# Simple AWGN channel model

def awgn(sym: torch.Tensor, snr_db: float) -> torch.Tensor:
    snr_linear = 10 ** (snr_db / 10)
    power = sym.abs().pow(2).mean()
    noise_var = power / snr_linear
    noise = torch.randn_like(sym) * torch.sqrt(noise_var / 2)
    noise += 1j * torch.randn_like(sym) * torch.sqrt(noise_var / 2)
    return sym + noise

# -------------------------------------------------------------
# Placeholder diffusion UNet (to be implemented later)
class DiffusionUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Implement UNet as per blueprint

    def forward(self, x, t, cond):
        # TODO: implement denoising step
        return x

# -------------------------------------------------------------
# Minimal training loop for Stage‑A (VQ‑VAE)

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    vqvae = VQVAE().to(device)
    opt = optim.Adam(vqvae.parameters(), lr=2e-4)
    train_loader = get_dataloader()

    # Validation loader
    val_loader = get_dataloader(batch_size=64, split='test')

    # Prepare fixed samples for visualization
    fixed_imgs, _ = next(iter(val_loader))
    fixed_imgs = fixed_imgs.to(device)[:8]
    initial_rec = None

    num_epochs = 10  # adjust as needed
    for epoch in range(1, num_epochs + 1):
        # Training
        vqvae.train()
        train_loss = 0.0
        for img, _ in train_loader:
            img = img.to(device)
            _, rec_loss, q_loss, _ = vqvae(img)
            loss = rec_loss + q_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        vqvae.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for img, _ in val_loader:
                img = img.to(device)
                x_rec, rec_loss, q_loss, _ = vqvae(img)
                loss = rec_loss + q_loss
                val_loss += loss.item() * img.size(0)
                # compute PSNR per batch
                val_psnr += compute_psnr(img, x_rec) * img.size(0)
            val_loss /= len(val_loader.dataset)
            val_psnr /= len(val_loader.dataset)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB")

        # Capture initial reconstruction
        if epoch == 1:
            with torch.no_grad():
                initial_rec = vqvae(fixed_imgs)[0].detach().cpu()




    # After training, save original, initial, and final reconstructions
    vqvae.eval()
    with torch.no_grad():
        final_rec = vqvae(fixed_imgs)[0].detach().cpu()
    # Stack: originals, initial reconstructions, final reconstructions
    comparison = torch.cat([fixed_imgs.cpu(), initial_rec, final_rec], dim=0)

    # --- Annotated saving: factual image/latent sizes ---
    # Compute original and latent sizes for annotations
    B, C, H, W = fixed_imgs.shape
    # Original image bytes (8-bit RGB)
    orig_bytes = H * W * C  # each channel 1 byte
    orig_kb = orig_bytes / 1024
    # Latent codes from quantizer
    with torch.no_grad():
        z_e = vqvae.encoder(fixed_imgs.to(device))
        _, _, indices = vqvae.quantizer(z_e)
    h_latent, w_latent = indices.shape[1], indices.shape[2]
    bits_per_index = math.ceil(math.log2(vqvae.quantizer.num_embed))
    latent_bits = h_latent * w_latent * bits_per_index
    latent_kb = (latent_bits / 8) / 1024  # bytes to KB

    # Create a grid of all images (3 rows of 8)
    grid = vutils.make_grid(comparison, nrow=8, padding=2)
    pil_img = to_pil_image(grid)

    # Draw size annotations
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    # Row y-positions: first row starts at y=0, each image row is height H=96, plus padding H_pad=2
    row_height = fixed_imgs.size(2) + 2
    annotations = [
        f"Original: {H}×{W}, {orig_kb:.2f} KB",
        f"Latent ({h_latent}×{w_latent} indices): {latent_kb:.2f} KB",
        f"Latent ({h_latent}×{w_latent} indices): {latent_kb:.2f} KB",
    ]
    for i, text in enumerate(annotations):
        y = i * row_height + 5  # 5px down from top of each row
        draw.text((5, y), text, font=font)

    # Save annotated image
    pil_img.save("recon_initial_and_final_annotated.png")


if __name__ == '__main__':
    main()

