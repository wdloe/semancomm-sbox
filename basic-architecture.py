

import torch
import torch.nn as nn

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


# Initialize components
input_dim = 128
hidden_dim = 64
encoded_dim = 32

encoder = SemanticEncoder(input_dim, hidden_dim, encoded_dim)
channel = Channel(noise_std=0.1)
decoder = SemanticDecoder(encoded_dim, hidden_dim, input_dim)

# Sample input
x = torch.randn((1, input_dim))

# Forward pass
encoded = encoder(x)
print("Encoded Representation:", encoded)

transmitted = channel(encoded)
print("After Channel Transmission:", transmitted)

recovered = decoder(transmitted)
print("Recovered Information:", recovered)