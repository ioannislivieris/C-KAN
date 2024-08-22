import numpy as np
import torch
from numba import jit
from torch.autograd import Function

def pairwise_distances(x, y=None):
    '''
    Calculate pairwise distances between vectors in x and y.

    Args:
        x (torch.Tensor): Input tensor of shape (N, d).
        y (torch.Tensor, optional): Optional tensor of shape (M, d). Defaults to None.

    Returns:
        torch.Tensor: Pairwise distance matrix of shape (N, M).
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x.float(), y_t.float())
    return torch.clamp(dist, 0.0, float('inf'))

@jit(nopython=True)
def compute_softdtw(D, gamma):
    '''
    Compute the soft dynamic time warping (DTW) distance.

    Args:
        D (numpy.ndarray): Input distance matrix of shape (N, M).
        gamma (float): Regularization parameter.

    Returns:
        numpy.ndarray: Soft DTW matrix.
    '''
    N = D.shape[0]
    M = D.shape[1]
    R = np.zeros((N + 2, M + 2)) + 1e8
    R[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            r0 = -R[i - 1, j - 1] / gamma
            r1 = -R[i - 1, j] / gamma
            r2 = -R[i, j - 1] / gamma
            rmax = max(max(r0, r1), r2)
            rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
            softmin = - gamma * (np.log(rsum) + rmax)
            R[i, j] = D[i - 1, j - 1] + softmin
    return R

@jit(nopython=True)
def compute_softdtw_backward(D_, R, gamma):
    '''
    Compute the gradient of soft dynamic time warping (DTW) distance.

    Args:
        D_ (numpy.ndarray): Input distance matrix of shape (N, M).
        R (numpy.ndarray): Soft DTW matrix.
        gamma (float): Regularization parameter.

    Returns:
        numpy.ndarray: Gradient of soft DTW.
    '''
    N = D_.shape[0]
    M = D_.shape[1]
    D = np.zeros((N + 2, M + 2))
    E = np.zeros((N + 2, M + 2))
    D[1:N + 1, 1:M + 1] = D_
    E[-1, -1] = 1
    R[:, -1] = -1e8
    R[-1, :] = -1e8
    R[-1, -1] = R[-2, -2]
    for j in range(M, 0, -1):
        for i in range(N, 0, -1):
            a0 = (R[i + 1, j] - R[i, j] - D[i + 1, j]) / gamma
            b0 = (R[i, j + 1] - R[i, j] - D[i, j + 1]) / gamma
            c0 = (R[i + 1, j + 1] - R[i, j] - D[i + 1, j + 1]) / gamma
            a = np.exp(a0)
            b = np.exp(b0)
            c = np.exp(c0)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c
    return E[1:N + 1, 1:M + 1]

class SoftDTWBatch(Function):
    '''
    Computes the soft dynamic time warping (DTW) distance for a batch of inputs.

    Attributes:
        gamma (float): Regularization parameter.
    '''
    @staticmethod
    def forward(ctx, D, gamma=1.0):
        '''
        Forward pass of SoftDTWBatch.

        Args:
            D (torch.Tensor): Input tensor of shape (batch_size, N, N).
            gamma (float, optional): Regularization parameter. Defaults to 1.0.

        Returns:
            torch.Tensor: Total loss averaged over the batch.
        '''
        dev = D.device
        batch_size, N, _ = D.shape
        gamma = torch.FloatTensor([gamma]).to(dev)
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()

        total_loss = 0
        R = torch.zeros((batch_size, N + 2, N + 2)).to(dev)   
        for k in range(batch_size):   
            Rk = torch.FloatTensor(compute_softdtw(D_[k, :, :], g_)).to(dev)
            R[k:k+1, :, :] = Rk
            total_loss = total_loss + Rk[-2, -2]
        ctx.save_for_backward(D, R, gamma)
        return total_loss / batch_size
  
    @staticmethod
    def backward(ctx, grad_output):
        '''
        Backward pass of SoftDTWBatch.

        Args:
            grad_output (torch.Tensor): Gradient of the loss.

        Returns:
            tuple: Gradient of input tensor and None.
        '''
        dev = grad_output.device
        D, R, gamma = ctx.saved_tensors
        batch_size, N, _ = D.shape
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()

        E = torch.zeros((batch_size, N, N)).to(dev) 
        for k in range(batch_size):         
            Ek = torch.FloatTensor(compute_softdtw_backward(D_[k, :, :], R_[k, :, :], g_)).to(dev)
            E[k:k+1, :, :] = Ek

        return grad_output * E, None
