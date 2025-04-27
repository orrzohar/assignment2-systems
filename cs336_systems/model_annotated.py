import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
import math

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
        # Compute attention scores
        with nvtx.range("computing_attention_scores"):
            d_k = Q.size(-1)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute softmax
        with nvtx.range("computing_softmax"):
            attn_weights = F.softmax(scores, dim=-1)

        # Final matrix multiplication
        with nvtx.range("final_matmul"):
            output = torch.matmul(attn_weights, V)

        return output