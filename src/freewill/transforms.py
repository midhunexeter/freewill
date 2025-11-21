import numpy as np
from typing import Callable, List, Union, Dict

def compose(*funcs: Callable) -> Callable:
    """
    Composes multiple functions into a single function.
    Functions are applied in order: funcs[0](x) -> funcs[1](result) -> ...
    
    Args:
        *funcs: Variable number of functions to compose.
        
    Returns:
        Callable: A new function that applies the given functions in sequence.
    """
    def composed(x):
        for f in funcs:
            x = f(x)
        return x
    return composed

def apply_to_key(key: str, func: Callable) -> Callable:
    """
    Returns a function that applies `func` to a specific key in a dictionary.
    Useful for transforming only the 'eeg' part of a sample dict.
    
    Args:
        key (str): The dictionary key to apply the function to.
        func (Callable): The function to apply.
        
    Returns:
        Callable: A function that takes a dict and returns a dict with the transformed value.
    """
    def wrapper(data: Dict):
        if key in data:
            data[key] = func(data[key])
        return data
    return wrapper

def normalize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Performs Z-score normalization (standardization) along the time dimension (last dimension).
    x = (x - mean) / (std + eps)
    
    Args:
        array (np.ndarray): Input array of shape (..., Time).
        eps (float): Small value to avoid division by zero.
        
    Returns:
        np.ndarray: Normalized array.
    """
    mean = array.mean(axis=-1, keepdims=True)
    std = array.std(axis=-1, keepdims=True)
    return (array - mean) / (std + eps)

def select_channels(array: np.ndarray, channel_indices: List[int]) -> np.ndarray:
    """
    Selects specific channels from the EEG array.
    
    Args:
        array (np.ndarray): Input array of shape (..., Channels, Time).
        channel_indices (List[int]): List of channel indices to keep.
        
    Returns:
        np.ndarray: Array with only selected channels.
    """
    # Assuming Channels is the second to last dimension (..., C, T)
    return array[..., channel_indices, :]

def scale(array: np.ndarray, factor: float) -> np.ndarray:
    """
    Scales the array by a scalar factor.
    
    Args:
        array (np.ndarray): Input array.
        factor (float): Scaling factor.
        
    Returns:
        np.ndarray: Scaled array.
    """
    return array * factor
