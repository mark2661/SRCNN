import pandas as pd
import os
from definitions import ROOT_DIR
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_pickle(os.path.join(ROOT_DIR, 'outputs', '128_network', 'model1', 'model1_385epochs.pickle'))
#print(df[['Training Loss', 'Validation Loss']])
x_data = np.arange(1, len(df)+1)
fig, axs = plt.subplots(2, 2, figsize= (18, 10))
fig.suptitle('128-64-1 Network Training History', fontsize='20', weight='bold')
axs[0, 0].plot(x_data, df['Training Loss'], 'tab:blue')
axs[0, 0].set_title('Training Loss', weight='bold')
axs[0, 0].set_ylabel('MSE')
axs[0, 1].plot(x_data, df['Validation Loss'], 'tab:red')
axs[0, 1].set_title('Validation Loss', weight='bold')
axs[0, 1].set_ylabel('MSE')
axs[1, 0].plot(x_data, df['Training_PSNR'], 'tab:blue')
axs[1, 0].set_title('Training PSNR', weight='bold')
axs[1, 0].set_ylabel('PSNR')
axs[1, 1].plot(x_data, df['Validation PSNR'], 'tab:red')
axs[1, 1].set_title('Validation PSNR', weight='bold')
axs[1, 1].set_ylabel('PSNR')

for ax in axs.flat:
    ax.set(xlabel='Epoch')
axs[0, 0].grid()
axs[0, 1].grid()
axs[1, 0].grid()
axs[1, 1].grid()
plt.show()