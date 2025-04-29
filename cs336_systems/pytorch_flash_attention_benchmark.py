import torch
import triton
import triton.language as tl
import math
from einops import rearrange
from typing import Tuple, Optional


#############################################
# Part (a): PyTorch FlashAttention-2 Forward Pass
#############################################

class FlashAttentionPytorch(torch.autograd.Function):
    """
    Pure PyTorch implementation of the FlashAttention-2 forward pass.
    This will be slower than the regular PyTorch implementation but helps debug the Triton kernel.
    """
    
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, scale=None):
        """
        Forward pass for FlashAttention-2.
        
        Args:
            ctx: Context to save tensors for the backward pass
            q: Query tensor of shape [batch_size, n_heads, seq_len_q, d_head]
            k: Key tensor of shape [batch_size, n_heads, seq_len_k, d_head]
            v: Value tensor of shape [batch_size, n_heads, seq_len_k, d_head]
            is_causal: Whether to apply causal masking
            scale: Scaling factor (1/sqrt(d_head) if None)
            
        Returns:
            o: Output tensor of shape [batch_size, n_heads, seq_len_q, d_head]
        """
        # Save tensors needed for backward
        ctx.save_for_backward(q, k, v)
        ctx.is_causal = is_causal
        
        # Get dimensions
        batch_size, n_heads, seq_len_q, d_head = q.shape
        _, _, seq_len_k, _ = k.shape
        
        # Set the scaling factor if not provided
        if scale is None:
            scale = 1.0 / math.sqrt(d_head)
        ctx.scale = scale
        
        # Define tile sizes (as mentioned in the assignment, use powers of 2 that divide sequence length)
        # For example:
        q_tile_size = 16  # Bq in the algorithm
        k_tile_size = 16  # Bk in the algorithm
        
        # Initialize output tensors
        o = torch.zeros_like(q)
        l = torch.zeros((batch_size, n_heads, seq_len_q), device=q.device)
        
        # TODO: Implement FlashAttention-2 tiled algorithm in PyTorch
        # Following Algorithm 1 from the assignment:
        
        # 1. Compute the number of tiles
        num_q_tiles = (seq_len_q + q_tile_size - 1) // q_tile_size
        num_k_tiles = (seq_len_k + k_tile_size - 1) // k_tile_size
        
        # 2. Iterate through query tiles
        for i in range(num_q_tiles):
            # Calculate query tile bounds
            q_tile_start = i * q_tile_size
            q_tile_end = min(q_tile_start + q_tile_size, seq_len_q)
            q_tile = q[:, :, q_tile_start:q_tile_end, :]
            
            # Initialize accumulators for this query tile
            o_tile = torch.zeros_like(q_tile)
            m_prev = torch.full((batch_size, n_heads, q_tile_end - q_tile_start), 
                                float('-inf'), device=q.device)
            l_prev = torch.zeros((batch_size, n_heads, q_tile_end - q_tile_start), 
                                device=q.device)
            
            # 3. Iterate through key tiles
            for j in range(num_k_tiles):
                # Calculate key tile bounds
                k_tile_start = j * k_tile_size
                k_tile_end = min(k_tile_start + k_tile_size, seq_len_k)
                k_tile = k[:, :, k_tile_start:k_tile_end, :]
                v_tile = v[:, :, k_tile_start:k_tile_end, :]
                
                # 4. Compute attention scores for this tile
                # TODO: S_tile = q_tile @ k_tile.transpose(-1, -2) * scale
                
                # 5. Apply causal masking if needed
                # TODO: if is_causal: ...
                
                # 6. Update accumulators (following the algorithm)
                # TODO:
                # - Compute m_curr (max of m_prev and rowmax of S_tile)
                # - Compute P_tilde (exp(S_tile - m_curr))
                # - Update l_curr based on Algorithm 1
                # - Update o_tile
                
                # 7. Save updated values for next iteration
                # TODO: m_prev = m_curr, l_prev = l_curr
            
            # 8. Normalize output for this query tile
            # TODO: o_tile = o_tile / l_curr.unsqueeze(-1)
            
            # 9. Compute logsumexp for this tile
            # TODO: l_tile = m_prev + torch.log(l_prev)
            
            # 10. Store results in output tensors
            o[:, :, q_tile_start:q_tile_end, :] = o_tile
            l[:, :, q_tile_start:q_tile_end] = m_prev + torch.log(l_prev)
        
        return o
    
    @staticmethod
    def backward(ctx, grad_output):
        # Not implementing backward for now
        raise NotImplementedError("Backward pass not implemented")


#############################################
# Part (b): Triton FlashAttention-2 Forward Pass
#############################################

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False,
):
    """
    Triton kernel for FlashAttention-2 forward pass.
    
    Args:
        Q_ptr: Pointer to Query tensor
        K_ptr: Pointer to Key tensor
        V_ptr: Pointer to Value tensor
        O_ptr: Pointer to Output tensor
        L_ptr: Pointer to Logsumexp tensor
        stride_*: Strides for each tensor dimension
        N_QUERIES: Total number of queries
        N_KEYS: Total number of keys
        scale: Scaling factor (1/sqrt(d_head))
        D: Head dimension size (compile-time constant)
        Q_TILE_SIZE: Query tile size (compile-time constant)
        K_TILE_SIZE: Key tile size (compile-time constant)
        is_causal: Whether to apply causal masking (compile-time constant)
    """
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    # TODO: Create block pointers for K, V, O, and L tensors
    # K_block_ptr = tl.make_block_ptr(...)
    # V_block_ptr = tl.make_block_ptr(...)
    # O_block_ptr = tl.make_block_ptr(...)
    # L_block_ptr = tl.make_block_ptr(...)
    
    # Initialize accumulators as described in Algorithm 1
    # O_i = 0, l_i = 0, m_i = -inf
    O_tile = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)
    l_prev = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    m_prev = tl.full([Q_TILE_SIZE], float('-inf'), dtype=tl.float32)
    
    # Create offset arrays for causal masking if needed
    if is_causal:
        # TODO: Create query and key indices for causal masking
        # query_indices = ...
        # key_indices = ...
        pass
    
    # Iterate over key tiles
    num_k_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for k_tile_idx in range(num_k_tiles):
        # Calculate offsets for the current key tile
        k_tile_offset = k_tile_idx * K_TILE_SIZE
        
        # TODO: Advance K and V block pointers to the current tile
        # K_tile_ptr = ...
        # V_tile_ptr = ...
        
        # Load key and value tiles
        # TODO: K_tile = tl.load(K_tile_ptr, boundary_check=(0, 1))
        # TODO: V_tile = tl.load(V_tile_ptr, boundary_check=(0, 1))
        
        # Load query tile (already offset with the query_tile_index)
        Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1))
        
        # TODO: Compute attention scores for this tile
        # S_tile = tl.dot(Q_tile, K_tile, trans_b=True) * scale
        
        # Apply causal masking if needed
        if is_causal:
            # TODO: Apply causal mask by adding -1e6 to masked positions
            # mask = ...
            # S_tile = tl.where(mask, S_tile, S_tile - 1e6)
            pass
        
        # TODO: Update accumulators (following the algorithm)
        # 1. Update m_curr (max of m_prev and rowmax of S_tile)
        # 2. Compute P_tilde (exp(S_tile - m_curr))
        # 3. Update l_curr based on Algorithm 1
        # 4. Update O_tile
        
        # TODO: Save updated values for the next iteration
        # m_prev = m_curr
        # l_prev = l_curr
    
    # TODO: Normalize the output and compute logsumexp
    # O_tile = O_tile / l_prev.unsqueeze(-1)
    # L_tile = m_prev + tl.log(l_prev)
    
    # TODO: Store results in output tensors
    # tl.store(O_block_ptr, O_tile)
    # tl.store(L_block_ptr, L_tile)


class FlashAttentionTriton(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2 forward pass.
    """
    
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, scale=None):
        """
        Forward pass for FlashAttention-2 using Triton.
        
        Args:
            ctx: Context to save tensors for the backward pass
            q: Query tensor of shape [batch_size, n_heads, seq_len_q, d_head]
            k: Key tensor of shape [batch_size, n_heads, seq_len_k, d_head]
            v: Value tensor of shape [batch_size, n_heads, seq_len_k, d_head]
            is_causal: Whether to apply causal masking
            scale: Scaling factor (1/sqrt(d_head) if None)
            
        Returns:
            o: Output tensor of shape [batch_size, n_heads, seq_len_q, d_head]
        """
        # Save tensors needed for backward
        ctx.save_for_backward(q, k, v)
        ctx.is_causal = is_causal
        
        # Get dimensions and check inputs
        batch_size, n_heads, seq_len_q, d_head = q.shape
        _, _, seq_len_k, _ = k.shape
        
        # Set the scaling factor if not provided
        if scale is None:
            scale = 1.0 / math.sqrt(d_head)
        ctx.scale = scale
        
        # Reshape inputs for Triton kernel
        # - Merge batch and head dimensions
        # - Ensure contiguous tensors
        q_reshaped = rearrange(q, 'b h s d -> (b h) s d').contiguous()
        k_reshaped = rearrange(k, 'b h s d -> (b h) s d').contiguous()
        v_reshaped = rearrange(v, 'b h s d -> (b h) s d').contiguous()
        
        # Define tile sizes
        # TODO: Tune these parameters for performance
        q_tile_size = 16  # Bq in the algorithm
        k_tile_size = 16  # Bk in the algorithm
        
        # Ensure tile sizes are powers of 2 and divide sequence lengths
        # (In practice, you'd verify this or adjust to the next power of 2)
        
        # Create output tensors
        o_reshaped = torch.empty_like(q_reshaped)
        l_reshaped = torch.empty((batch_size * n_heads, seq_len_q), device=q.device)
        
        # Calculate number of tiles needed
        # Each thread block processes one query tile for one batch+head
        num_q_tiles = (seq_len_q + q_tile_size - 1) // q_tile_size
        grid = (num_q_tiles, batch_size * n_heads)
        
        # Launch Triton kernel
        flash_fwd_kernel[grid](
            q_reshaped, k_reshaped, v_reshaped,  # Input tensors
            o_reshaped, l_reshaped,              # Output tensors
            q_reshaped.stride(0), q_reshaped.stride(1), q_reshaped.stride(2),  # Q strides
            k_reshaped.stride(0), k_reshaped.stride(1), k_reshaped.stride(2),  # K strides
            v_reshaped.stride(0), v_reshaped.stride(1), v_reshaped.stride(2),  # V strides
            o_reshaped.stride(0), o_reshaped.stride(1), o_reshaped.stride(2),  # O strides
            l_reshaped.stride(0), l_reshaped.stride(1),                       # L strides
            seq_len_q, seq_len_k,
            scale,
            d_head,                             # D (compile-time constant)
            q_tile_size,                        # Q_TILE_SIZE (compile-time constant)
            k_tile_size,                        # K_TILE_SIZE (compile-time constant)
            is_causal,                          # is_causal (compile-time constant)
        )
        
        # Reshape output back to original shape
        o = rearrange(o_reshaped, '(b h) s d -> b h s d', b=batch_size, h=n_heads)
        l = rearrange(l_reshaped, '(b h) s -> b h s', b=batch_size, h=n_heads)
        
        # Store additional values needed for backward pass
        ctx.l = l
        
        return o
    
    @staticmethod
    def backward(ctx, grad_output):
        # Not implementing backward for now
        raise NotImplementedError("Backward pass not implemented")


#############################################
# Adapter functions for testing
#############################################

def get_flashattention_autograd_function_pytorch():
    """
    Returns the PyTorch FlashAttention autograd function.
    """
    return FlashAttentionPytorch.apply

def get_flash_autograd_function_triton():
    """
    Returns the Triton FlashAttention autograd function.
    """
    return FlashAttentionTriton.apply