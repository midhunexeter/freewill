# API Reference

This document provides a reference for the classes and functions available in the `freewill` package.

## Data Loading

### `freewill.dataset.FreewillDataset`

A PyTorch `Dataset` class for loading and processing the Freewill EEG Reaching Grasping dataset.

```python
class FreewillDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, subject_ids: list = None, transform: callable = None, fixed_length: int = 3000):
        ...
```

**Arguments:**

*   `root_dir` (str): Path to the root directory of the dataset (should contain `derivatives/matfiles`).
*   `subject_ids` (list, optional): List of subject IDs to include (e.g., `['01', '02']`). If `None`, automatically discovers all available subjects.
*   `transform` (callable, optional): A function/transform that takes in a sample and returns a transformed version.
*   `fixed_length` (int, optional): The target length (number of time points) for each trial. Trials are padded with zeros or truncated to match this length. Default is `3000`.

**Returns:**

*   `__getitem__` returns a dictionary:
    *   `'eeg'`: `torch.Tensor` of shape `(Channels, Time)`.
    *   `'label'`: `int` representing the target ID.
    *   `'info'`: `dict` containing metadata (subject, session, trial index, etc.).

---

## Functional Transforms

The `freewill.transforms` module provides pure functions for data transformation. These can be composed and passed to the `FreewillDataset`.

### `freewill.transforms.compose`

```python
def compose(*funcs: Callable) -> Callable
```

Composes multiple functions into a single function. `compose(f, g)(x)` is equivalent to `g(f(x))`.

### `freewill.transforms.apply_to_key`

```python
def apply_to_key(key: str, func: Callable) -> Callable
```

Wraps a function to apply it only to a specific key in a dictionary. Useful for applying tensor transforms to the `'eeg'` key of a sample.

**Example:**
```python
transform = compose(
    apply_to_key('eeg', normalize),
    apply_to_key('eeg', lambda x: scale(x, 1e6)) # Scale to microvolts
)
```

### `freewill.transforms.normalize`

```python
def normalize(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor
```

Performs Z-score normalization (standardization) along the time dimension (last dimension) for each channel.

### `freewill.transforms.select_channels`

```python
def select_channels(tensor: torch.Tensor, channel_indices: List[int]) -> torch.Tensor
```

Selects a subset of channels from the EEG tensor.

### `freewill.transforms.scale`

```python
def scale(tensor: torch.Tensor, factor: float) -> torch.Tensor
```

Multiplies the tensor by a scalar factor.
