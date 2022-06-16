import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # TODO: Define model
        self.linear1 = nn.Sequential(nn.utils.weight_norm(nn.Linear(259, 512)),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob)) 
        self.linear2 = nn.Sequential(nn.utils.weight_norm(nn.Linear(512, 512)),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob))
        self.linear3 = nn.Sequential(nn.utils.weight_norm(nn.Linear(512, 512)),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob))
        self.linear4 = nn.Sequential(nn.utils.weight_norm(nn.Linear(512, 253)),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob))
        self.linear5 = nn.Sequential(nn.utils.weight_norm(nn.Linear(512, 512)),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob))
        self.linear6 = nn.Sequential(nn.utils.weight_norm(nn.Linear(512, 512)),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob))
        self.linear7 = nn.Sequential(nn.utils.weight_norm(nn.Linear(512, 512)),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob))
        self.linear8 = nn.Sequential(nn.utils.weight_norm(nn.Linear(512, 512)),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob))
        self.linear_final = nn.Linear(512,1)

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x = self.linear1(x_in)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(torch.cat([x,x_in], axis=1))
        x = self.linear6(x)
        x = self.linear7(x)
        x = self.linear8(x)
        x = self.linear_final(x)
        
        return x
