"""Image utilities."""
import numpy as np


def resize_image(image, target_size):
    """
    Simple image resize using numpy interpolation.
    
    Args:
        image: [H, W] or [H, W, C] array
        target_size: (H_new, W_new) tuple
    
    Returns:
        resized: [H_new, W_new] or [H_new, W_new, C]
    """
    h_old, w_old = image.shape[0], image.shape[1]
    h_new, w_new = target_size
    
    if image.ndim == 3:
        # [H, W, C] image
        c = image.shape[2]
        resized = np.zeros((h_new, w_new, c), dtype=image.dtype)
        for ch in range(c):
            h_indices = (np.arange(h_new) * h_old / h_new).astype(int)
            w_indices = (np.arange(w_new) * w_old / w_new).astype(int)
            h_indices = np.clip(h_indices, 0, h_old - 1)
            w_indices = np.clip(w_indices, 0, w_old - 1)
            resized[:, :, ch] = image[np.ix_(h_indices, w_indices, [ch])].squeeze(-1)
        return resized
    else:
        # [H, W] grayscale
        h_indices = (np.arange(h_new) * h_old / h_new).astype(int)
        w_indices = (np.arange(w_new) * w_old / w_new).astype(int)
        h_indices = np.clip(h_indices, 0, h_old - 1)
        w_indices = np.clip(w_indices, 0, w_old - 1)
        return image[np.ix_(h_indices, w_indices)]
