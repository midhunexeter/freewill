import torch
from torch.utils.data import DataLoader
from freewill.dataset import FreewillDataset
import os

def test_dataset():
    # Assuming we are running from project root
    root_dir = os.path.join(os.getcwd(), '.data', 'Freewill_EEG_Reaching_Grasping 2')
    
    print(f"Looking for data in: {root_dir}")
    
    if not os.path.exists(root_dir):
        print("Data directory not found. Please check path.")
        return

    try:
        # Initialize dataset with just one subject for speed
        dataset = FreewillDataset(root_dir, subject_ids=['01'])
        print(f"Dataset initialized. Total samples: {len(dataset)}")
        
        if len(dataset) == 0:
            print("No samples found!")
            return

        # Test __getitem__
        sample = dataset[0]
        eeg = sample['eeg']
        label = sample['label']
        info = sample['info']
        
        print(f"Sample 0:")
        print(f"  EEG Shape: {eeg.shape}")
        print(f"  Label: {label}")
        print(f"  Info: {info}")
        
        # Test DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        
        print("\nBatch (size 4):")
        print(f"  EEG Batch Shape: {batch['eeg'].shape}")
        print(f"  Labels: {batch['label']}")
        
        print("\nVerification Successful!")
        
    except Exception as e:
        print(f"\nVerification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
