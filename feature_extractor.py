import torch
import torch.nn as nn
import torchvision.models as models

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, feature_dim=512):
        super(MultiScaleFeatureExtractor, self).__init__()

        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            modules = list(base_model.children())[:-2]
            self.encoder = nn.Sequential(*modules)
            self.out_channels = 512
        else:
            raise NotImplementedError("Only resnet18 is currently supported.")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(self.out_channels, feature_dim)

    def forward_single_scale(self, video):
        """
        video: [T, C, H, W] -> output: [T, feature_dim]
        """
        features = []
        for frame in video:
            x = self.encoder(frame.unsqueeze(0))     # [1, C, H, W]
            x = self.pool(x)                         # [1, C, 1, 1]
            x = x.view(1, -1)                        # [1, C]
            x = self.projection(x)                   # [1, feature_dim]
            features.append(x)
        return torch.cat(features, dim=0)            # [T, feature_dim]

    def forward(self, videos_by_scale):
        """
        videos_by_scale: list of 5 tensors [T, C, H, W] at different scales
        returns: list of 5 feature sequences [T, D] (same length T assumed)
        """
        return [self.forward_single_scale(video) for video in videos_by_scale]
