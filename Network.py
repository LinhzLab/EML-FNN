import torch.nn as nn
import torch
import random

manualSeed = random.randint(1, 10000)  
torch.manual_seed(manualSeed)

class FNN(nn.Module):
    def __init__(self, input_dim=4, dropout_p=0.02,width=256,init_weights=False):
        super(FNN, self).__init__()
        self.regression = nn.Sequential(
            nn.Linear(input_dim,width),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Linear(width, 1),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.regression(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1e-4)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 1e-4)