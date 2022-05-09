import glob
import os
import Test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from definitions import ROOT_DIR

SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')
best_models_training_histories = dict()

for network in glob.glob(os.path.join(ROOT_DIR, 'outputs', '*')):
    network_filter_number = int(network.split('\\')[-1].split('_')[0])
    best_model_score = 0
    best_model = None
    for model_state_dict in glob.glob(os.path.join(network, 'model*', 'model*.pth')):
        test_score = Test.main(SET14_PATH, model_state_dict, network_filter_number)[0]
        if test_score > best_model_score:
            best_model_score = test_score
            best_model = model_state_dict[:-3] + 'pickle'
    print(best_model)
    #best_models_training_histories[network_filter_number] = best_model

# for model in best_models_training_histories.keys():
#     df = pd.read_pickle(best_models_training_histories[model])
#     x_data = np.arange(1, len(df) + 1)
#     fig, axs = plt.subplots(2, 2, figsize=(19.2, 10.8))
#     fig.suptitle('{}-{}-1 Network Best Model Training History'.format(str(model), str(model//2)), fontsize='20',
#                  weight='bold')
#     axs[0, 0].plot(x_data, df['Training Loss'], 'tab:blue')
#     axs[0, 0].set_title('Training Average Loss', weight='bold')
#     axs[0, 0].set_ylabel('MSE')
#     axs[0, 1].plot(x_data, df['Validation Loss'], 'tab:red')
#     axs[0, 1].set_title('Validation Average Loss', weight='bold')
#     axs[0, 1].set_ylabel('MSE')
#     axs[1, 0].plot(x_data, df['Training_PSNR'], 'tab:blue')
#     axs[1, 0].set_title('Training Average PSNR', weight='bold')
#     axs[1, 0].set_ylabel('PSNR')
#     axs[1, 1].plot(x_data, df['Validation PSNR'], 'tab:red')
#     axs[1, 1].set_title('Validation Average PSNR', weight='bold')
#     axs[1, 1].set_ylabel('PSNR')
#
#     for ax in axs.flat:
#         ax.set(xlabel='Epoch')
#     axs[0, 0].grid()
#     axs[0, 1].grid()
#     axs[1, 0].grid()
#     axs[1, 1].grid()
#     plt.savefig(os.path.join(ROOT_DIR, 'Data', 'Plots', '{}_network_training_history.svg'.format(model)), format='svg',
#                 dpi=1200)
