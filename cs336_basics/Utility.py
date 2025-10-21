import os
import torch
import numpy as np


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    """
    Given a long sequence of token IDs, sample batches of subsequences for training.
    Args:
        x: numpy array of token IDs
        batch_size: number of sequences per batch
        context_length: sequence length per sample
        device: 'cpu', 'cuda', or 'mps'
    Returns:
        input_batch, target_batch (torch.Tensor): shape (batch_size, context_length)
    """
    # Ensure valid indices
    n = len(x) - context_length
    starts = np.random.randint(0, n, size=batch_size)
    input_batch = np.stack([x[i:i+context_length] for i in starts])
    target_batch = np.stack([x[i+1:i+1+context_length] for i in starts])

    # Move to torch tensor and device
    input_batch = torch.tensor(input_batch, dtype=torch.long, device=device)
    target_batch = torch.tensor(target_batch, dtype=torch.long, device=device)
    return input_batch, target_batch

def save_checkpoint(model, optimizer, iteration: int, out):
    """
    Save model, optimizer state, and iteration number to a file.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    """
    Load model, optimizer, and iteration from checkpoint.
    Returns:
        iteration (int)
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]