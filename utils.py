import torch

def dice_loss(predict: torch.Tensor, ground_truth: torch.Tensor, 
              eps = 1e-8):
    # There are only two classes in the images (i.e, one background and one kind of foreground)
    # so this implementation might be fine
    intersect = 2*torch.sum((predict*ground_truth) > 0) + eps
    union = torch.sum(predict > 0) + torch.sum(ground_truth > 0) + eps
    return 1 - intersect / union

def dice_scores(predict: torch.Tensor, ground_truth: torch.Tensor, eps = 1e-8):
    intersect = 2*torch.sum((predict*ground_truth) > 0) + eps
    union = torch.sum(predict > 0) + torch.sum(ground_truth > 0) + eps
    return intersect / union 