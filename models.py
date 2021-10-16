from functools import partial

import torch
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from timm.models.resnet import resnext50_32x4d


ENCODERS = {
'resnext50_32x4d': {
'features': 2048,
'init_op': partial(resnext50_32x4d, pretrained=True)
}
}


class SignsClassifier(nn.Module):
    """
    A model for classifying signs.
    """

    def __init__(self, encoder_name: str, n_classes: int, dropout_rate: float = 0.5):
        """Initializing the class.

        :param encoder_name: name of the network encoder
        :param n_classes: number of output classes
        :param dropout_rate: dropout rate
        """
        super().__init__()
        self.encoder = ENCODERS[encoder_name]['init_op']()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(ENCODERS[encoder_name]['features'], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Getting the model prediction.

        :param x: input batch tensor
        :return: prediction
        """
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
