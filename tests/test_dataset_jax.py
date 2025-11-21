import jax.numpy as jnp
from freewill.dataset import FreewillDataset
import os
import numpy as np

def test_dataset_jax():
    # Assuming we are running from project root
    root_dir = os.path.join(os.getcwd(), '.data', 'Freewill_EEG_Reaching_Grasping 2')
    
    print(f"Looking for data in: {root_dir}")
    
    if not os.path.exists(root_dir):
        print("Data directory not found. Please check path.")
        return

    try:
        # Initialize dataset
        dataset = FreewillDataset(root_dir, subject_ids=['01'], fixed_length=3000)
        print(f"Dataset initialized. Total samples: {len(dataset)}")
        
        if len(dataset) == 0:
            print("No samples found!")
            return

        # Get a sample (returns numpy array)
        sample = dataset[0]
        eeg_np = sample['eeg']
        label = sample['label']
        
        print(f"Sample 0 (NumPy):")
        print(f"  EEG Shape: {eeg_np.shape}")
        print(f"  Type: {type(eeg_np)}")
        
        # Convert to JAX array
        eeg_jax = jnp.array(eeg_np)
        
        print(f"Sample 0 (JAX):")
        print(f"  EEG Shape: {eeg_jax.shape}")
        print(f"  Type: {type(eeg_jax)}")
        
        # Simple JAX operation to verify
        mean_val = jnp.mean(eeg_jax)
        print(f"  Mean value (computed by JAX): {mean_val}")
        
        print("\nJAX Verification Successful!")
        
    except Exception as e:
        print(f"\nVerification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_jax()
