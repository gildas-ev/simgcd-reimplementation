import torch

class Model(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)
