import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_adjacency_matrix(X, threshold=0.0):
    """
    Compute the adjacency matrix based on variable correlations.
    
    Parameters:
        X: np.ndarray of shape (T, D) – time-series data.
        threshold: float – minimum absolute correlation to include an edge.

    Returns:
        A: np.ndarray of shape (D, D) – adjacency matrix.
    """
    corr_matrix = np.corrcoef(X.T)  # (D, D)
    corr_matrix = np.nan_to_num(corr_matrix)

    # Optional: threshold to sparsify adjacency
    if threshold > 0.0:
        corr_matrix[np.abs(corr_matrix) < threshold] = 0.0
    
    return corr_matrix

class TriggerGeneratorGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adjacency_matrix):
        """
        Parameters:
            input_dim: int – number of input features (D).
            hidden_dim: int – GCN hidden layer size.
            output_dim: int – output dimension (usually same as input_dim).
            adjacency_matrix: np.ndarray of shape (D, D) – variable-level graph.
        """
        super(TriggerGeneratorGCN, self).__init__()
        
        A = adjacency_matrix
        if not isinstance(A, torch.Tensor):
            A = torch.tensor(A, dtype=torch.float32)
        
        # Normalize adjacency matrix (GCN trick: A_hat = D^-0.5 * A * D^-0.5)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(A.sum(1) + 1e-6))
        self.A_hat = D_inv_sqrt @ A @ D_inv_sqrt  # normalized adjacency

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass of GCN.
        
        Parameters:
            x: Tensor of shape (D, input_dim) – identity input (or node features).
        Returns:
            out: Tensor of shape (D, output_dim) – trigger values.
        """
        h = self.A_hat @ x
        h = F.relu(self.fc1(h))
        h = self.A_hat @ h
        out = self.fc2(h)
        return out
