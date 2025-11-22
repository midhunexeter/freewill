import os
import warnings
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import pandas as pd

class FreewillDataset(Dataset):
    """
    PyTorch Dataset for the Freewill EEG Reaching Grasping dataset.
    """
    def __init__(self, root_dir, subject_ids=None, transform=None, fixed_length=3000):
        """
        Args:
            root_dir (str): Path to the root directory of the dataset (containing derivatives/matfiles).
            subject_ids (list, optional): List of subject IDs to include (e.g., ['01', '02']). 
                                          If None, all available subjects are used.
            transform (callable, optional): Optional transform to be applied on a sample.
            fixed_length (int, optional): Fixed length to pad/truncate trials to. Default 3000.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.fixed_length = fixed_length
        self.samples = []

        if subject_ids is None:
            # Auto-discover subjects
            matfiles_dir = os.path.join(root_dir, 'derivatives', 'matfiles')
            if os.path.exists(matfiles_dir):
                 subject_ids = [d.replace('sub-', '') for d in os.listdir(matfiles_dir) if d.startswith('sub-')]
                 subject_ids.sort()
            else:
                raise FileNotFoundError(f"Could not find matfiles directory at {matfiles_dir}")

        self._prepare_index(subject_ids)

    def _prepare_index(self, subject_ids):
        """
        Scans files and builds an index of all available trials.
        """
        for sub_id in subject_ids:
            # Construct path: derivatives/matfiles/sub-XX/ses-01/sub-XX_ses-01_task-reachingandgrasping_eeg.mat
            # Note: Assuming session 01 for simplicity, can be expanded.
            # The structure seems to be derivatives/matfiles/sub-XX/ses-XX/...
            
            sub_dir = os.path.join(self.root_dir, 'derivatives', 'matfiles', f'sub-{sub_id}')
            if not os.path.exists(sub_dir):
                warnings.warn(f"Subject directory not found: {sub_dir}", UserWarning)
                continue
                
            for ses_name in os.listdir(sub_dir):
                if not ses_name.startswith('ses-'): continue
                
                mat_path = os.path.join(sub_dir, ses_name, f'sub-{sub_id}_{ses_name}_task-reachingandgrasping_eeg.mat')
                
                if os.path.exists(mat_path):
                    # We load the mat file metadata to know how many trials there are
                    # For efficiency, we might want to cache this or just load the table
                    try:
                        mat = loadmat(mat_path, simplify_cells=True)
                        if 'data' not in mat: continue
                        data = mat['data']
                        
                        if 'dataTable' not in data: continue
                        data_table = data['dataTable']
                        
                        # Convert to DataFrame to easily parse
                        # Row 0 is header
                        df = pd.DataFrame(data_table[1:], columns=data_table[0])
                        
                        # Filter valid trials if necessary (e.g. TrialDiscardIndex == 0)
                        # The notebook showed 'TrialDiscardIndex' column
                        if 'TrialDiscardIndex' in df.columns:
                            valid_trials = df[df['TrialDiscardIndex'] == 0]
                        else:
                            valid_trials = df

                        for idx, row in valid_trials.iterrows():
                            self.samples.append({
                                'mat_path': mat_path,
                                'trial_index': idx, # Index in the original dataframe/table
                                'subject': sub_id,
                                'session': ses_name,
                                'target_id': row.get('TgtID', -1), # Example label
                                # Store other metadata as needed
                                'start_index': row.get('TrialStartIndex', 0),
                                'end_index': row.get('TrialEndIndex', 0)
                            })
                            
                    except Exception as e:
                        print(f"Error loading {mat_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_info = self.samples[idx]
        mat_path = sample_info['mat_path']
        
        # Lazy loading: Load the file only when needed
        # Optimization: Cache open files if memory allows, or rely on OS caching
        mat = loadmat(mat_path, simplify_cells=True)
        data = mat['data']
        
        # Extract EEG data
        # rawEEG is (channels x time) or (time x channels)? 
        # Based on explore.py: raw_eeg[0].T[:, 0] -> Transpose suggests rawEEG is (channels x time) originally?
        # Let's check explore.py again: df = pd.DataFrame(raw_eeg[0].T, columns= data['channelName'])
        # This implies raw_eeg[0] is (channels, time). Transpose makes it (time, channels).
        
        raw_eeg = data['rawEEG'] # This seems to contain all runs concatenated or list of runs?
        # The structure of rawEEG needs to be confirmed. 
        # Usually in these datasets, rawEEG might be a continuous recording.
        # But dataTable has start/end indices.
        
        # Assuming rawEEG is a continuous array for the session
        # We need to slice it using start/end index
        
        # Note: explore.py used raw_eeg[0]. This suggests rawEEG might be a 1-element array containing the actual matrix?
        # Or maybe it's a cell array in MATLAB.
        
        eeg_signal = raw_eeg # Placeholder, might need [0]
        if isinstance(eeg_signal, list) or (isinstance(eeg_signal, np.ndarray) and eeg_signal.dtype == object):
             # It's likely a cell array, take first element if it's the main data
             # But wait, if there are multiple runs, rawEEG might correspond to runs?
             # The README says: "Continuous EEG... for all runs"
             pass

        # Let's assume for now based on explore.py:
        # eeg_data_ch1 = raw_eeg[0].T[:, 0]
        # So raw_eeg[0] is the data matrix.
        
        full_signal = raw_eeg
        if isinstance(full_signal, np.ndarray) and full_signal.ndim == 1:
             full_signal = full_signal[0] # Unpack if it's nested
             
        # full_signal shape should be (Channels, Time)
        
        start = int(sample_info['start_index']) - 1 # MATLAB 1-based indexing
        end = int(sample_info['end_index'])
        
        # Slice the trial
        # Check dimensions. If (Channels, Time), slice second dim.
        trial_data = full_signal[:, start:end]
        
        # Handle variable length
        if self.fixed_length:
            current_len = trial_data.shape[1]
            if current_len > self.fixed_length:
                # Truncate
                trial_data = trial_data[:, :self.fixed_length]
            elif current_len < self.fixed_length:
                # Pad with zeros
                pad_len = self.fixed_length - current_len
                # Pad the second dimension (time)
                trial_data = np.pad(trial_data, ((0, 0), (0, pad_len)), mode='constant')
        
        # Convert to float32 numpy array (PyTorch DataLoader will convert to tensor automatically)
        eeg_array = trial_data.astype(np.float32)
        
        # Label
        label = int(sample_info['target_id'])
        
        sample = {'eeg': eeg_array, 'label': label, 'info': sample_info}

        if self.transform:
            sample = self.transform(sample)

        return sample
