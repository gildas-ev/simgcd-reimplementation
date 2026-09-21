import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, out_dim, mlp_dims):
        super().__init__()
        
        assert len(mlp_dims) >= 2
        features_dim = mlp_dims[0]
        
        # mlp
        layers = []
        for i in range(len(mlp_dims)-1):
            layer = nn.Linear(mlp_dims[i], mlp_dims[i+1])
            nn.init.trunc_normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias, 0)
            
            layers.append(layer)

            if i != len(mlp_dims) - 2: # GELU activation unless for the last layer
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

        # prototypes
        self.prototypes = nn.Parameter(torch.empty(out_dim, features_dim))
        nn.init.kaiming_uniform_(self.prototypes, a=math.sqrt(5))

    def forward(self, x):
        x_proj = self.mlp(x)

        x_norm = F.normalize(x, dim=-1)
        prototypes_norm = F.normalize(self.prototypes, dim=-1)
        logits = x_norm @ prototypes_norm.T

        return x_proj, logits
