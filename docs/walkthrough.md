# Using the Freewill PyTorch Dataset

I have implemented a custom PyTorch Dataset class `FreewillDataset` to easily load and use the EEG data.

## Features
- **Automatic Discovery**: Finds all subjects and sessions in the data directory.
- **Lazy Loading**: Loads .mat files only when needed to save memory.
- **Fixed Length**: Automatically pads or truncates trials to a fixed length (default 3000 time points) to ensure valid batching.
- **PyTorch Compatible**: Works seamlessly with `torch.utils.data.DataLoader`.

## Usage Example

```python
from freewill.dataset import FreewillDataset
from torch.utils.data import DataLoader
import os

# Define path to data
root_dir = '../.data/Freewill_EEG_Reaching_Grasping 2'

# Initialize Dataset
dataset = FreewillDataset(root_dir, subject_ids=['01'], fixed_length=3000)

# Access a single sample
sample = dataset[0]
print(f"EEG Shape: {sample['eeg'].shape}") # (Channels, Time)
print(f"Label: {sample['label']}")

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch in loader:
    eeg_data = batch['eeg']
    labels = batch['label']
    print(f"Batch Shape: {eeg_data.shape}") # (32, 39, 3000)
    break
```

### JAX Usage

```python
import jax.numpy as jnp

# Access a sample (returns NumPy array)
sample = dataset[0]
eeg_np = sample['eeg']

# Convert to JAX array
eeg_jax = jnp.array(eeg_np)
print(f"JAX Array Shape: {eeg_jax.shape}")
```

## Implementation Details
The dataset is located in `src/freewill/dataset.py`. It uses `scipy.io.loadmat` to read the data and `pandas` to parse the trial table.
The `__getitem__` method returns standard **NumPy arrays**, making it compatible with any framework.
