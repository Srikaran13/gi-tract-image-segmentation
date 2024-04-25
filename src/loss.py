from torch.nn import BCEWithLogitsLoss

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCEWithLogitsLoss()
    
    def forward(self, y_pred, y_true):
        dice_loss = 1 - (2 * (y_pred * y_true).sum() + 1) / ((y_pred + y_true).sum() + 1)
        bce_loss = self.bce(y_pred, y_true)
        return 0.5 * dice_loss + 0.5 * bce_loss

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
