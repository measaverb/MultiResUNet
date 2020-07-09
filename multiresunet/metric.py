import torch


def dice_coeff(pred, target, threshold=0.5, epsilon=1e-6, use_sigmoid = True):
    pred = pred.contiguous()
    if use_sigmoid:
        pred = torch.nn.Sigmoid()(pred)
    target = target.contiguous()
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + epsilon) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + epsilon)
    return dice.mean()
