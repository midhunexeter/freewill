import scipy.io as sio
import os

# Path to a sample .mat file
mat_path = ".data/Freewill_EEG_Reaching_Grasping 2/derivatives/matfiles/sub-01/ses-01/sub-01_ses-01_task-reachingandgrasping_eeg.mat"

if os.path.exists(mat_path):
    try:
        mat = sio.loadmat(mat_path, simplify_cells=True)
        if 'data' in mat and 'metadata' in mat['data']:
            metadata = mat['data']['metadata']
            print("Metadata found:")
            # Print nicely if it's a dictionary or list
            if isinstance(metadata, dict):
                for k, v in metadata.items():
                    print(f"{k}: {v}")
            else:
                print(metadata)
        else:
            print("Metadata field not found in 'data' struct.")
    except Exception as e:
        print(f"Error loading mat file: {e}")
else:
    print(f"File not found: {mat_path}")
