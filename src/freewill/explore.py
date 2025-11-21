import numpy as np
import scipy as sc
from scipy.io import loadmat
import pandas as pd
from matplotlib import pyplot as plt

path = "derivatives/matfiles/sub-01/ses-01/sub-01_ses-01_task-reachingandgrasping_eeg.mat"

data = loadmat(path, simplify_cells=True)['data']

fs = data['fs']
channel_names = data['channelName']
raw_eeg = data['rawEEG']
data_table = data['dataTable']


df_data_table = pd.DataFrame(data_table[1:], columns=data_table[0]).dropna()

acc_start = df_data_table['AccStartIndex'].dropna().astype(int).values - 1
eeg_data_ch1 = raw_eeg[0].T[:, 0]

df = pd.DataFrame(raw_eeg[0].T, columns= data['channelName'])

df ['AccStart'] = 0
df.loc[acc_start, 'AccStart'] = 1000000


df.plot()
plt.show()