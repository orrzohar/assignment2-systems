import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx

def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Annotated version of scaled dot product attention with NVTX ranges for profiling.
    This function replaces the original implementation in cs336_basics.model for profiling.
    
    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharing leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
    
    Returns:
        Tensor with output of scaled dot product attention.
    """
    with nvtx.range("scaled_dot_product_attention"):
        with nvtx.range("computing_attention_scores"):
            d_k = Q.size(-1)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

        with nvtx.range("computing_softmax"):
            attn_weights = F.softmax(scores, dim=-1)

        with nvtx.range("final_matmul"):
            output = torch.matmul(attn_weights, V)

        return output

class PyTorchAttention(nn.Module):
    """
    PyTorch implementation of attention mechanism.
    
    This is a standard attention implementation that computes:
    Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    
    Args:
        d_model: Dimension of the model
        compiled: Whether to use torch.compile
    """
    def __init__(self, d_model: int, compiled: bool = False):
        super().__init__()
        self.d_model = d_model
        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.compiled = compiled
        
        if compiled:
            self.forward = torch.compile(self._forward)
        else:
            self.forward = self._forward
        
    def _forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Compute attention.
        
        Args:
            Q: Query tensor of shape (batch_size, seq_len, d_model)
            K: Key tensor of shape (batch_size, seq_len, d_model)
            V: Value tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Compute output
        output = torch.matmul(attn_weights, V)
        
        return output