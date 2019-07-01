import torch


def dice_loss(pred, target):
    """Cacluate dice loss 

    Parameters
    ----------
        pred:
            predictions from the model
        target:
            ground truth label
    """
                                                                         
    smooth = 1.                                                          
                                                                         
    p_flat = pred.contiguous().view(-1)                                   
    t_flat = target.contiguous().view(-1)                                 
    intersection = (p_flat * t_flat).sum()                                 
                                                                         
    a_sum = torch.sum(p_flat * p_flat)                                     
    b_sum = torch.sum(t_flat * t_flat)                                     
                                                                         
    return 1 - ((2. * intersection + smooth) / (a_sum + b_sum + smooth) )

