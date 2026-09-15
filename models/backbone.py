import torch

def build_backbone(encoder_name):
    """Loads and returns the encoder (dino vitb 16)

    All parameters have requires_grad set to False"""
    backbone = torch.hub.load(*encoder_name, trust_repo=True)
    for _, param in backbone.named_parameters():
        param.requires_grad = False
    
    return backbone

def unfreeze_vit_from(backbone, from_num):
    """Sets the requires_grad of the parameters of the blocks starting from the number from_num to True (inplace)"""
    for name, param in backbone.named_parameters():
        if name[:6] == 'blocks':
            num = int(name.split('.')[1])
            if num >= from_num:
                param.requires_grad = True
