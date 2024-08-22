import torch
from . import soft_dtw
from . import path_soft_dtw 

def dilate_loss(outputs, targets, alpha, gamma, device):
    """Compute the dilate loss between outputs and targets.

    Args:
        outputs (torch.Tensor): Predicted outputs, shape (batch_size, N_output, 1).
        targets (torch.Tensor): Target values, shape (batch_size, N_output, 1).
        alpha (float): Weight parameter for shape loss.
        gamma (float): Gamma parameter for Soft-DTW loss.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: Total loss.
        torch.Tensor: Shape loss component.
        torch.Tensor: Temporal loss component.
    """
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]
    loss_shape = 0
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    D = torch.zeros((batch_size, N_output, N_output)).to(device)
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1))
        D[k:k + 1, :, :] = Dk     
    loss_shape = softdtw_batch(D, gamma)
    
    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)           
    Omega = soft_dtw.pairwise_distances(torch.arange(1, N_output + 1).view(N_output, 1)).to(device)
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output) 
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal


def dilate_mse_loss(outputs, targets, alpha, beta, gamma, device):
    """Compute the dilate loss between outputs and targets.

    Args:
        outputs (torch.Tensor): Predicted outputs, shape (batch_size, N_output, 1).
        targets (torch.Tensor): Target values, shape (batch_size, N_output, 1).
        alpha (float): Weight parameter for shape loss.
        beta (float): Weight parameter for temporal loss.
        gamma (float): Gamma parameter for Soft-DTW loss.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: Total loss.
        torch.Tensor: Shape loss component.
        torch.Tensor: Temporal loss component.
    """
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]
    loss_shape = 0
    softdtw_batch = soft_dtw.SoftDTWBatch.apply
    D = torch.zeros((batch_size, N_output, N_output)).to(device)
    for k in range(batch_size):
        Dk = soft_dtw.pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1))
        D[k:k + 1, :, :] = Dk     
    loss_shape = softdtw_batch(D, gamma)
    
    path_dtw = path_soft_dtw.PathDTWBatch.apply
    path = path_dtw(D, gamma)           
    Omega = soft_dtw.pairwise_distances(torch.arange(1, N_output + 1).view(N_output, 1)).to(device)
    loss_temporal = torch.sum(path * Omega) / (N_output * N_output) 
    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal