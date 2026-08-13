import torch
import torch.nn as nn

class SimpleTransUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(SimpleTransUNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder without upsampling (keeps shape 128x128x128)
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=1)  # logits for each class
        )

    def forward(self, x):
        x = self.encoder(x)     # (B, 64, 128, 128, 128)
        x = self.decoder(x)     # (B, out_channels, 128, 128, 128)
        return x
