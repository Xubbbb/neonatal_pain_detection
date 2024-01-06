import torchvision.models as models
import torch.nn as nn
from torchvision.models import vgg16

# We only use VGG16's CNN part as feature extractor
# The output is 7x7x512, there are 512 channels
class CustomVGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(CustomVGG16FeatureExtractor, self).__init__()
        _vgg16 = vgg16(pretrained=True)
        self.features = _vgg16.features
    
    def forward(self, x):
        # Output will be (batch_size, 512, 7, 7)
        x = self.features(x)
        return x