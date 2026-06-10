from pathlib import Path
import random
import shutil
import math
import csv

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration
# ============================================================

RAW_DATA_DIR = Path("datasets") / "AIDER"
WORK_DATA_DIR = Path("data") / "aider_split"

IMAGE_SIZE = 240
BATCH_SIZE = 64
NUM_EPOCHS = 100

# The paper uses initial LR = 0.1 with Adam and cosine annealing.
# In PyTorch, 0.1 may be aggressive. Start with 1e-3 for stability.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1

LATENT_DIM = 128

NUM_TX_ANTENNAS = 4
NUM_RX_ANTENNAS = 4
NUM_CHANNEL_USES = LATENT_DIM // NUM_TX_ANTENNAS

SNR_DB = 10.0
SNR_SWEEP_DB = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
SNR_SWEEP_TRIALS = 10

K_FACTOR = 5.0

TRAIN_RATIO = 0.5
VAL_RATIO = 0.2
TEST_RATIO = 0.3

SEED = 42
NUM_WORKERS = 2

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. Dataset preparation
# ============================================================

def is_image_file(path):
    return path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]


def prepare_dataset(
    raw_data_dir,
    work_data_dir,
    train_ratio,
    val_ratio,
    test_ratio,
    seed,
):
    random.seed(seed)

    if work_data_dir.exists():
        print("Dataset split already exists at: {}".format(work_data_dir))
        return

    if not raw_data_dir.exists():
        raise FileNotFoundError(
            "Raw dataset folder not found: {}".format(raw_data_dir)
        )

    class_dirs = [p for p in raw_data_dir.iterdir() if p.is_dir()]

    if len(class_dirs) == 0:
        raise RuntimeError(
            "No class folders found in {}. Check whether the dataset is nested.".format(
                raw_data_dir
            )
        )

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    for class_dir in class_dirs:
        image_paths = [
            p for p in class_dir.rglob("*")
            if p.is_file() and is_image_file(p)
        ]

        if len(image_paths) == 0:
            print("Warning: no images found in {}".format(class_dir))
            continue

        random.shuffle(image_paths)

        n_total = len(image_paths)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        split_files = {
            "train": image_paths[:n_train],
            "val": image_paths[n_train:n_train + n_val],
            "test": image_paths[n_train + n_val:],
        }

        for split_name, files in split_files.items():
            output_class_dir = work_data_dir / split_name / class_dir.name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            for src_path in files:
                dst_path = output_class_dir / src_path.name
                shutil.copy2(src_path, dst_path)

        print(
            "{}: {} train, {} val, {} test".format(
                class_dir.name,
                len(split_files["train"]),
                len(split_files["val"]),
                len(split_files["test"]),
            )
        )


def make_balanced_sampler(dataset):
    targets = torch.tensor(dataset.targets, dtype=torch.long)

    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


# ============================================================
# 4. Activation
# ============================================================

class CappedLeakyReLU(nn.Module):
    """
    Approximation of the capped leaky ReLU idea used in EmergencyNet.

    During training, it keeps a small negative slope.
    The upper cap is kept high, so it does not usually affect normalized
    floating-point training but keeps the design close to the paper.
    """

    def __init__(self, negative_slope=0.1, cap=255.0):
        super().__init__()
        self.negative_slope = negative_slope
        self.cap = cap

    def forward(self, x):
        x = torch.where(x >= 0, x, self.negative_slope * x)
        x = torch.clamp(x, max=self.cap)
        return x


# ============================================================
# 5. EmergencyNet-style ACFF block
# ============================================================

class ACFFBlock(nn.Module):
    """
    Atrous Convolutional Feature Fusion block.

    This block uses three depthwise atrous convolution paths:

        dilation = 1, effective receptive field 3 x 3
        dilation = 2, effective receptive field 5 x 5
        dilation = 3, effective receptive field 7 x 7

    The paths are fused by addition.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        use_pool=False,
        dropout=0.2,
    ):
        super().__init__()

        reduced_channels = max(out_channels // 2, 8)

        self.reduce = nn.Sequential(
            nn.Conv2d(
                in_channels,
                reduced_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(reduced_channels),
            CappedLeakyReLU(),
        )

        self.branch_d1 = nn.Sequential(
            nn.Conv2d(
                reduced_channels,
                reduced_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
                groups=reduced_channels,
                bias=False,
            ),
            nn.BatchNorm2d(reduced_channels),
            CappedLeakyReLU(),
        )

        self.branch_d2 = nn.Sequential(
            nn.Conv2d(
                reduced_channels,
                reduced_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=reduced_channels,
                bias=False,
            ),
            nn.BatchNorm2d(reduced_channels),
            CappedLeakyReLU(),
        )

        self.branch_d3 = nn.Sequential(
            nn.Conv2d(
                reduced_channels,
                reduced_channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                groups=reduced_channels,
                bias=False,
            ),
            nn.BatchNorm2d(reduced_channels),
            CappedLeakyReLU(),
        )

        self.expand = nn.Sequential(
            nn.Conv2d(
                reduced_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            CappedLeakyReLU(),
        )

        if use_pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.reduce(x)

        y1 = self.branch_d1(x)
        y2 = self.branch_d2(x)
        y3 = self.branch_d3(x)

        y = y1 + y2 + y3
        y = self.expand(y)
        y = self.pool(y)
        y = self.dropout(y)

        return y


# ============================================================
# 6. EmergencyNet-style semantic coder
# ============================================================

class EmergencyNetFeatureCoder(nn.Module):
    """
    EmergencyNet-style source coder.

    It maps an input aerial image to a compact task-oriented latent vector.
    """

    def __init__(self, latent_dim):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            CappedLeakyReLU(),
        )

        self.acff1 = ACFFBlock(16, 64, use_pool=True, dropout=0.2)
        self.acff2 = ACFFBlock(64, 96, use_pool=False, dropout=0.2)
        self.acff3 = ACFFBlock(96, 128, use_pool=True, dropout=0.2)
        self.acff4 = ACFFBlock(128, 128, use_pool=False, dropout=0.2)
        self.acff5 = ACFFBlock(128, 128, use_pool=False, dropout=0.2)
        self.acff6 = ACFFBlock(128, 256, use_pool=False, dropout=0.2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(256, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            CappedLeakyReLU(),
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.acff1(x)
        x = self.acff2(x)
        x = self.acff3(x)
        x = self.acff4(x)
        x = self.acff5(x)
        x = self.acff6(x)

        x = self.global_pool(x)
        x = x.flatten(start_dim=1)

        z = self.fc(x)

        return z


# ============================================================
# 7. Channel coder
# ============================================================

class ChannelCoder(nn.Module):
    """
    Maps semantic features to MIMO transmit symbols.

    Output shape:
        s: [batch, Nt, L]
    """

    def __init__(
        self,
        latent_dim,
        num_tx_antennas,
        num_channel_uses,
    ):
        super().__init__()

        self.num_tx_antennas = num_tx_antennas
        self.num_channel_uses = num_channel_uses
        self.output_dim = num_tx_antennas * num_channel_uses

        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),

            nn.Linear(latent_dim, self.output_dim),
        )

    def forward(self, z):
        s = self.net(z)

        power = torch.mean(s ** 2, dim=1, keepdim=True)
        s = s / torch.sqrt(power + 1e-8)

        s = s.view(
            z.size(0),
            self.num_tx_antennas,
            self.num_channel_uses,
        )

        return s


# ============================================================
# 8. Rician MIMO channel with AWGN
# ============================================================

class RicianMIMOChannel(nn.Module):
    """
    Real-valued flat Rician MIMO channel with AWGN.

    Channel model:
        Y = H S + N

    Rician fading:
        H = sqrt(K / (K + 1)) H_los
            + sqrt(1 / (K + 1)) H_nlos

    Input:
        s: [batch, Nt, L]

    Output:
        y_noisy: [batch, Nr, L]
        h: [batch, Nr, Nt]
    """

    def __init__(self, num_tx, num_rx, snr_db, k_factor):
        super().__init__()
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.snr_db = float(snr_db)
        self.k_factor = float(k_factor)

    def forward(self, s):
        batch_size = s.size(0)
        device = s.device

        k = torch.tensor(float(self.k_factor), device=device)
        nt = torch.tensor(float(self.num_tx), device=device)

        h_los = torch.ones(
            batch_size,
            self.num_rx,
            self.num_tx,
            device=device,
        ) / torch.sqrt(nt)

        h_nlos = torch.randn(
            batch_size,
            self.num_rx,
            self.num_tx,
            device=device,
        ) / torch.sqrt(nt)

        h = (
            torch.sqrt(k / (k + 1.0)) * h_los
            + torch.sqrt(1.0 / (k + 1.0)) * h_nlos
        )

        y = torch.bmm(h, s)

        signal_power = torch.mean(y ** 2)
        snr_linear = 10.0 ** (self.snr_db / 10.0)
        noise_power = signal_power / snr_linear

        noise = torch.sqrt(noise_power + 1e-8) * torch.randn_like(y)

        y_noisy = y + noise

        return y_noisy, h


# ============================================================
# 9. Receive beamformer
# ============================================================

class ReceiveBeamformer(nn.Module):
    """
    Learnable receive beamformer.

    It combines Nr received antenna streams into one stream.

    Input:
        y: [batch, Nr, L]

    Output:
        r: [batch, L]
    """

    def __init__(self, num_rx_antennas):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(num_rx_antennas))

    def forward(self, y):
        w = self.weights / (torch.norm(self.weights) + 1e-8)
        r = torch.einsum("r,brl->bl", w, y)
        return r


# ============================================================
# 10. Task decoder
# ============================================================

class TaskDecoder(nn.Module):
    """
    Decodes the received beamformed signal and predicts the class.
    """

    def __init__(self, num_channel_uses, latent_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(num_channel_uses, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),

            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),

            nn.Dropout(0.3),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, r):
        logits = self.net(r)
        return logits


# ============================================================
# 11. Full task-oriented MIMO system
# ============================================================

class TaskOrientedMIMOSystem(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_tx_antennas,
        num_rx_antennas,
        num_channel_uses,
        num_classes,
        snr_db,
        k_factor,
    ):
        super().__init__()

        self.coder = EmergencyNetFeatureCoder(latent_dim)

        self.channel_coder = ChannelCoder(
            latent_dim=latent_dim,
            num_tx_antennas=num_tx_antennas,
            num_channel_uses=num_channel_uses,
        )

        self.channel = RicianMIMOChannel(
            num_tx=num_tx_antennas,
            num_rx=num_rx_antennas,
            snr_db=snr_db,
            k_factor=k_factor,
        )

        self.receive_beamformer = ReceiveBeamformer(
            num_rx_antennas=num_rx_antennas
        )

        self.decoder = TaskDecoder(
            num_channel_uses=num_channel_uses,
            latent_dim=latent_dim,
            num_classes=num_classes,
        )

    def forward(self, x):
        z = self.coder(x)
        s = self.channel_coder(z)
        y, _ = self.channel(s)
        r = self.receive_beamformer(y)
        logits = self.decoder(r)

        return logits


# ============================================================
# 12. Metrics
# ============================================================

def compute_macro_f1_from_counts(confusion_matrix):
    num_classes = confusion_matrix.size(0)
    f1_values = []

    for c in range(num_classes):
        tp = confusion_matrix[c, c].float()
        fp = confusion_matrix[:, c].sum().float() - tp
        fn = confusion_matrix[c, :].sum().float() - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        f1_values.append(f1)

    macro_f1 = torch.stack(f1_values).mean().item()
    return macro_f1


def update_confusion_matrix(confusion_matrix, labels, predictions, num_classes):
    for true_label, pred_label in zip(labels.view(-1), predictions.view(-1)):
        confusion_matrix[true_label.long(), pred_label.long()] += 1

    return confusion_matrix


# ============================================================
# 13. Training and evaluation
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    num_classes,
    optimizer=None,
):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    confusion_matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long,
    )

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if is_training:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        predictions = torch.argmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        confusion_matrix = update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            labels=labels.detach().cpu(),
            predictions=predictions.detach().cpu(),
            num_classes=num_classes,
        )

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    macro_f1 = compute_macro_f1_from_counts(confusion_matrix)

    return average_loss, accuracy, macro_f1, confusion_matrix


# ============================================================
# 14. SNR sweep
# ============================================================

def evaluate_snr_sweep(
    model,
    test_loader,
    criterion,
    num_classes,
    snr_values_db,
    output_dir,
    num_trials,
):
    model.eval()

    output_dir.mkdir(exist_ok=True)

    snr_results = []

    for snr_db in snr_values_db:
        model.channel.snr_db = float(snr_db)

        trial_losses = []
        trial_accs = []
        trial_f1s = []

        for trial in range(num_trials):
            with torch.no_grad():
                test_loss, test_acc, test_f1, _ = run_epoch(
                    model=model,
                    loader=test_loader,
                    criterion=criterion,
                    num_classes=num_classes,
                    optimizer=None,
                )

            trial_losses.append(test_loss)
            trial_accs.append(test_acc)
            trial_f1s.append(test_f1)

        mean_loss = sum(trial_losses) / len(trial_losses)
        mean_acc = sum(trial_accs) / len(trial_accs)
        mean_f1 = sum(trial_f1s) / len(trial_f1s)

        acc_var = sum((a - mean_acc) ** 2 for a in trial_accs) / len(trial_accs)
        f1_var = sum((f - mean_f1) ** 2 for f in trial_f1s) / len(trial_f1s)

        acc_std = math.sqrt(acc_var)
        f1_std = math.sqrt(f1_var)

        snr_results.append(
            {
                "snr_db": snr_db,
                "test_loss": mean_loss,
                "test_accuracy": mean_acc,
                "test_accuracy_std": acc_std,
                "test_macro_f1": mean_f1,
                "test_macro_f1_std": f1_std,
            }
        )

        print(
            "SNR {:>6.1f} dB | Loss {:.4f} | Acc {:.4f} +/- {:.4f} | F1 {:.4f} +/- {:.4f}".format(
                snr_db,
                mean_loss,
                mean_acc,
                acc_std,
                mean_f1,
                f1_std,
            )
        )

    csv_path = output_dir / "snr_sweep_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snr_db",
                "test_loss",
                "test_accuracy",
                "test_accuracy_std",
                "test_macro_f1",
                "test_macro_f1_std",
            ],
        )
        writer.writeheader()

        for row in snr_results:
            writer.writerow(row)

    snrs = [row["snr_db"] for row in snr_results]
    accs = [row["test_accuracy"] * 100.0 for row in snr_results]
    acc_stds = [row["test_accuracy_std"] * 100.0 for row in snr_results]

    plt.figure()
    plt.errorbar(snrs, accs, yerr=acc_stds, marker="o", capsize=4)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Task-Oriented UAV Classification Accuracy vs SNR")
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()

    plot_path = output_dir / "snr_sweep_accuracy.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    f1s = [row["test_macro_f1"] * 100.0 for row in snr_results]
    f1_stds = [row["test_macro_f1_std"] * 100.0 for row in snr_results]

    plt.figure()
    plt.errorbar(snrs, f1s, yerr=f1_stds, marker="o", capsize=4)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Macro F1 Score (%)")
    plt.title("Task-Oriented UAV Classification Macro F1 vs SNR")
    plt.grid(True)
    plt.ylim(0, 100)
    plt.tight_layout()

    f1_plot_path = output_dir / "snr_sweep_macro_f1.png"
    plt.savefig(f1_plot_path, dpi=300)
    plt.close()

    print("SNR sweep CSV saved to: {}".format(csv_path))
    print("Accuracy plot saved to: {}".format(plot_path))
    print("Macro F1 plot saved to: {}".format(f1_plot_path))

    return snr_results


# ============================================================
# 15. Plot training curves
# ============================================================

def save_training_curves(history, output_dir):
    output_dir.mkdir(exist_ok=True)

    epochs = [row["epoch"] for row in history]

    train_acc = [row["train_acc"] * 100.0 for row in history]
    val_acc = [row["val_acc"] * 100.0 for row in history]

    plt.figure()
    plt.plot(epochs, train_acc, marker="o", label="Train")
    plt.plot(epochs, val_acc, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 100)
    plt.tight_layout()

    acc_path = output_dir / "training_accuracy.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()

    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure()
    plt.plot(epochs, train_loss, marker="o", label="Train")
    plt.plot(epochs, val_loss, marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    loss_path = output_dir / "training_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    csv_path = output_dir / "training_history.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "train_f1",
                "val_loss",
                "val_acc",
                "val_f1",
            ],
        )
        writer.writeheader()

        for row in history:
            writer.writerow(row)

    print("Training history saved to: {}".format(csv_path))
    print("Training accuracy plot saved to: {}".format(acc_path))
    print("Training loss plot saved to: {}".format(loss_path))


# ============================================================
# 16. Main
# ============================================================

def main():
    set_seed(SEED)

    print("Using device: {}".format(DEVICE))
    print("Raw dataset: {}".format(RAW_DATA_DIR))
    print("Working dataset: {}".format(WORK_DATA_DIR))

    prepare_dataset(
        raw_data_dir=RAW_DATA_DIR,
        work_data_dir=WORK_DATA_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        root=WORK_DATA_DIR / "train",
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=WORK_DATA_DIR / "val",
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=WORK_DATA_DIR / "test",
        transform=eval_transform,
    )

    num_classes = len(train_dataset.classes)

    train_sampler = make_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print("Classes: {}".format(train_dataset.classes))
    print("Number of classes: {}".format(num_classes))
    print("Train images: {}".format(len(train_dataset)))
    print("Validation images: {}".format(len(val_dataset)))
    print("Test images: {}".format(len(test_dataset)))
    print("Image size: {} x {}".format(IMAGE_SIZE, IMAGE_SIZE))
    print("Batch size: {}".format(BATCH_SIZE))
    print("Epochs: {}".format(NUM_EPOCHS))
    print("Learning rate: {}".format(LEARNING_RATE))
    print("Weight decay: {}".format(WEIGHT_DECAY))
    print("Label smoothing: {}".format(LABEL_SMOOTHING))
    print("Nt: {}".format(NUM_TX_ANTENNAS))
    print("Nr: {}".format(NUM_RX_ANTENNAS))
    print("Channel uses: {}".format(NUM_CHANNEL_USES))
    print("Training SNR: {} dB".format(SNR_DB))
    print("Rician K-factor: {}".format(K_FACTOR))

    model = TaskOrientedMIMOSystem(
        latent_dim=LATENT_DIM,
        num_tx_antennas=NUM_TX_ANTENNAS,
        num_rx_antennas=NUM_RX_ANTENNAS,
        num_channel_uses=NUM_CHANNEL_USES,
        num_classes=num_classes,
        snr_db=SNR_DB,
        k_factor=K_FACTOR,
    ).to(DEVICE)

    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    except TypeError:
        print("Your PyTorch version does not support label_smoothing.")
        print("Using standard CrossEntropyLoss instead.")
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=0.0,
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    best_model_path = checkpoint_dir / "best_task_oriented_emergencynet_mimo.pt"

    best_val_f1 = 0.0
    history = []

    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, train_f1, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            num_classes=num_classes,
            optimizer=optimizer,
        )

        with torch.no_grad():
            val_loss, val_acc, val_f1, _ = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                num_classes=num_classes,
                optimizer=None,
            )

        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

        history.append(
            {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
        )

        print(
            "Epoch {:03d}/{} | LR {:.6f} | Train Loss {:.4f} | Train Acc {:.4f} | Train F1 {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Val F1 {:.4f}".format(
                epoch + 1,
                NUM_EPOCHS,
                current_lr,
                train_loss,
                train_acc,
                train_f1,
                val_loss,
                val_acc,
                val_f1,
            )
        )

    save_training_curves(
        history=history,
        output_dir=results_dir,
    )

    model.load_state_dict(
        torch.load(best_model_path, map_location=DEVICE)
    )

    model.channel.snr_db = float(SNR_DB)

    with torch.no_grad():
        test_loss, test_acc, test_f1, test_confusion = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            num_classes=num_classes,
            optimizer=None,
        )

    print("Final result at training SNR")
    print("Test Loss: {:.4f}".format(test_loss))
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Test Macro F1: {:.4f}".format(test_f1))
    print("Best Validation Macro F1: {:.4f}".format(best_val_f1))

    confusion_path = results_dir / "test_confusion_matrix.csv"

    with open(confusion_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + train_dataset.classes)

        for class_name, row in zip(train_dataset.classes, test_confusion.tolist()):
            writer.writerow([class_name] + row)

    print("Test confusion matrix saved to: {}".format(confusion_path))

    print("Running SNR sweep")

    evaluate_snr_sweep(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        num_classes=num_classes,
        snr_values_db=SNR_SWEEP_DB,
        output_dir=results_dir,
        num_trials=SNR_SWEEP_TRIALS,
    )


if __name__ == "__main__":
    main()