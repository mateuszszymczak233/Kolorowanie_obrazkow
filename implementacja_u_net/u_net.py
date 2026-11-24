import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, kernel_size=3, padding=1):
        super().__init__()
        # Define a convolutional block with two convolutional layers, ReLU activations, and dropout
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Forward pass through the convolutional block
        return self.layers(x)


class AttentionGate(nn.Module):
    def __init__(self, channels, kernel_size=1, padding=0):
        super().__init__()
        # Define an attention gate that computes attention scores for skip connections
        hidden_channels = max(1, channels // 2)
        self.skip_projection = nn.Conv2d(channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.gate_projection = nn.Conv2d(channels, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.score = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, skip, gate):
        # Compute attention scores based on the skip connection and gating signal
        attention = self.score(
            self.skip_projection(skip) + self.gate_projection(gate)
        )
        return skip * attention


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, dropout, kernel_size=3, padding=1):
        super().__init__()
        # Define a decoder block consisting of an upsampling layer, an attention gate, and a convolutional block
        self.upsample = nn.ConvTranspose2d(
            in_channels, skip_channels, kernel_size=kernel_size, stride=2, padding=padding
        )
        self.attention = AttentionGate(skip_channels, kernel_size=kernel_size, padding=padding)
        self.convolution = ConvBlock(skip_channels * 2, skip_channels, dropout, kernel_size=kernel_size, padding=padding)

    def forward(self, x, skip):
        # Upsample the input feature map
        x = self.upsample(x)

        # Ensure the spatial dimensions match for concatenation
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        # Apply attention to the skip connection and concatenate with the upsampled feature map
        skip = self.attention(skip, x)

        return self.convolution(torch.cat((skip, x), dim=1))


class u_net(nn.Module):
    """Attention U-Net mapping normalized grayscale L to normalized Lab ab."""

    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        num_layers=4,
        features_start=64,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        kernel_size=3,
        padding=1,
    ):
        super().__init__()

        # Validate input parameters
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive.")
        if num_layers < 1 or features_start < 1:
            raise ValueError("num_layers and features_start must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if learning_rate <= 0.0 or weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Build the U-Net architecture
        self.encoders = nn.ModuleList()
        skip_channels = []
        current_channels = in_channels

        # Define the encoder blocks
        for layer in range(num_layers):
            channels = features_start * (2**layer)
            self.encoders.append(ConvBlock(current_channels, channels, dropout))
            skip_channels.append(channels)
            current_channels = channels

        # Define the pooling layer and bottleneck
        self.pool = nn.MaxPool2d(kernel_size=2)
        bottleneck_channels = current_channels * 2
        self.bottleneck = ConvBlock(current_channels, bottleneck_channels, dropout)

        # Define the decoder blocks
        self.decoders = nn.ModuleList()
        for channels in reversed(skip_channels):
            self.decoders.append(
                DecoderBlock(bottleneck_channels, channels, dropout, kernel_size=kernel_size, padding=padding)
            )
            bottleneck_channels = channels

        # Define the final output layer
        self.output = nn.Conv2d(features_start, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # Validate the input tensor shape
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input shaped [batch, {self.in_channels}, height, width]."
            )

        # Perform the forward pass through the U-Net architecture
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        # Pass through the bottleneck and then through the decoder blocks with skip connections
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Use tanh (later the values are going to be scaled by 110 needed to convert to RGB)
        return torch.tanh(self.output(x))

    def configure_optimizer(self):
        """Create an Adam optimizer using the configurable training parameters."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
