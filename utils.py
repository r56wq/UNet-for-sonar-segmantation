import torch
import torch.nn as nn
import torch.nn.functional as F

class soft_dice(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        Args:
            smooth (float): A smoothing factor to avoid division by zero. Default: 1e-5.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, digits, y_true):
         """
        Args:
            digits (torch.Tensor): Raw logits of shape (N, C, H, W), it is before softmax
            y_true (torch.Tensor): Ground truth one-hot encoded tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: The computed Soft Dice Loss (scalar).
        """
         y_pred = torch.softmax(digits, dim=1)
         intersection = torch.sum((y_pred*y_true), dim=(2, 3)) + self.smooth
         union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) + self.smooth
         dice = 2*intersection/union
         return 1 - dice.mean()
    

class hard_dice(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        Args:
            smooth (float): A smoothing factor to avoid division by zero. Default: 1e-5.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, digits, y_true):
         """
        Args:
            digits (torch.Tensor): Raw logits of shape (N, C, H, W), it is before softmax
            y_true (torch.Tensor): Ground truth one-hot encoded tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: The computed Soft Dice Loss (scalar).
        """
         y_pred = digits.argmax(dim=1)
         # convert y_pred to one hot 
         y_pred_one_hot = F.one_hot(y_pred, num_classes=y_true.shape[1]).permute(0, 3, 1, 2)
         intersection = torch.sum((y_pred_one_hot*y_true), dim=(2, 3)) + self.smooth
         union = y_pred_one_hot.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) + self.smooth
         dice = 2*intersection/union
         return 1 - dice.mean()

if __name__ == "__main__":
    # Example usage
    batch_size, num_classes, height, width = 2, 3, 4, 4
    digits = torch.randn(batch_size, num_classes, height, width)  # Raw logits
    y_true = torch.randint(0, num_classes, (batch_size, height, width))  # Ground truth class indices
    y_true = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2).float()  # Convert to one-hot

    # Initialize losses
    soft_dice_loss = soft_dice()
    hard_dice_loss = hard_dice()

    # Compute losses
    soft_loss = soft_dice_loss(digits, y_true)
    hard_loss = hard_dice_loss(digits, y_true)

    print(f"Soft Dice Loss: {soft_loss.item()}")
    print(f"Hard Dice Loss: {hard_loss.item()}")