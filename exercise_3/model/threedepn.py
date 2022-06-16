#from nis import cat
#from tkinter import _Padding
import torch
import torch.nn as nn


class ThreeDEPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 80

        # TODO: 4 Encoder layers
        self.e1 = nn.Conv3d(2, self.num_features, 4, stride = 2, padding = 1)
        #self.b1 = nn.BatchNorm3d(self.num_features)
        
        self.e2 = nn.Conv3d(self.num_features, self.num_features*2, 4, stride = 2, padding=1)
        self.ebn2 = nn.BatchNorm3d(self.num_features*2)

        self.e3 = nn.Conv3d(self.num_features*2, self.num_features*4, 4, stride=2, padding=1)
        self.ebn3 = nn.BatchNorm3d(self.num_features*4)

        self.e4 = nn.Conv3d(self.num_features*4, self.num_features*8, 4, stride=1, padding=0)
        self.ebn4 = nn.BatchNorm3d(self.num_features*8)
        
        # TODO: 2 Bottleneck layers
        self.bottleneck = nn.Sequential(nn.Linear(self.num_features * 8,self.num_features * 8),
                                        nn.ReLU(),
                                        nn.Linear(self.num_features*8, self.num_features*8),
                                        nn.ReLU())
        # TODO: 4 Decoder layers
        self.d4 = nn.ConvTranspose3d(self.num_features*8*2, self.num_features*4, 4, stride=1, padding=0)
        self.dbn4 = nn.BatchNorm3d(self.num_features*4)

        self.d3 = nn.ConvTranspose3d(self.num_features*4*2, self.num_features*2, 4, stride=2, padding=1)
        self.dbn3 = nn.BatchNorm3d(self.num_features*2)

        self.d2 = nn.ConvTranspose3d(self.num_features*2*2, self.num_features, 4, stride=2, padding=1)
        self.dbn2 = nn.BatchNorm3d(self.num_features)

        self.d1 = nn.ConvTranspose3d(self.num_features*2, 1, 4, stride=2, padding=1)
    def forward(self, x):
        b = x.shape[0]
        # Encode
        # TODO: Pass x though encoder while keeping the intermediate outputs for the skip connections
        x_e1 = nn.functional.leaky_relu(self.e1(x), 0.2)
        x_e2 = nn.functional.leaky_relu(self.ebn2(self.e2(x_e1)), 0.2)
        x_e3 = nn.functional.leaky_relu(self.ebn3(self.e3(x_e2)), 0.2)
        x_e4 = nn.functional.leaky_relu(self.ebn4(self.e4(x_e3)), 0.2)

        # Reshape and apply bottleneck layers
        x = x_e4.view(b, -1)
        x = self.bottleneck(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)
        # Decode
        # TODO: Pass x through the decoder, applying the skip connections in the process
        x_d4 = nn.functional.relu(self.dbn4(self.d4(torch.cat([x, x_e4], axis=1))))
        x_d3 = nn.functional.relu(self.dbn3(self.d3(torch.cat([x_d4, x_e3], axis=1))))
        x_d2 = nn.functional.relu(self.dbn2(self.d2(torch.cat([x_d3, x_e2], axis=1))))
        x_d1 = self.d1(torch.cat([x_d2, x_e1], axis=1))

        x = torch.squeeze(x_d1, dim=1)
        # TODO: Log scaling
        x = torch.log(torch.abs(x) + 1)
        return x
