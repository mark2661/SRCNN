import os
import matplotlib.pyplot as plt
import pickle
from definitions import ROOT_DIR
import numpy as np


with open(os.path.join(ROOT_DIR, 'Data', 'test_results.pickle'), 'rb') as f:
    test_results = pickle.load(f)

with open(os.path.join(ROOT_DIR, 'test_results_set14_fixed.pickle'), 'rb') as f:
    set14_test_results = pickle.load(f)

bsds_results = {key: test_results[key][0] for key in sorted(test_results.keys())}
set5_results = {key: test_results[key][1] for key in sorted(test_results.keys())}
#set14_results = {key: test_results[key][2] for key in sorted(test_results.keys())}
urban_results = {key: test_results[key][3] for key in sorted(test_results.keys())}

set14_results = set14_test_results

test_sets = [set5_results, set14_results, bsds_results, urban_results]
plot_titles = {id(set5_results): 'Set5', id(set14_results): 'Set14', id(bsds_results): 'BSDS100',
               id(urban_results): 'Urban100'}
#print(set14_results)
#print(bsds_results)

# # PSNR and MSE bar charts
for test_set_results in test_sets:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False, figsize=(19.2, 10.8))

    # PSNR and MSE bar charts
    psnr_scores = [test_set_results[key][0] for key in sorted(test_set_results.keys())]
    ssim_scores = [test_set_results[key][1] for key in sorted(test_set_results.keys())]
    mse_scores = [test_set_results[key][2] for key in sorted(test_set_results.keys())]

    fig.suptitle('{} Median Test Scores (x3 upscaling)'.format(plot_titles[id(test_set_results)]), weight='bold', fontsize=20)
    # create bar charts
    bar_width = 0.3
    ax1.set_title('MSE and PSNR (dB)', weight='bold', fontsize=15)

    # x-axis spacing
    r1 = np.arange(len(psnr_scores))
    r2 = [x + bar_width for x in r1]

    # Create orange bars
    mse_bar = ax1.bar(r1, mse_scores, width=bar_width, color='orange', align='center', edgecolor='black',
                      label='Median MSE')
    # Create blue bars
    psnr_bar = ax1.bar(r2, psnr_scores, width=bar_width, color='blue', align='center', edgecolor='black',
                       label='Median PSNR')

    # #Create orange bars
    # mse_bar = ax1.bar(r2, mse_scores, width=bar_width, color='orange', align='center', edgecolor='black',
    #                   label='Median MSE')

    # general layout
    ax1.set_xticks([r + bar_width/2 for r in range(len(psnr_scores))], ['16-8-1 filter network',
                                                                        '32-16-1 filter network',
                                                                        '64-32-1 filter network',
                                                                        '128-64-1 filter network',
                                                                        '256-128-1 filter network'])
    # plt.ylim(0, max(max(psnr_scores), max(mse_scores)))
    ax1.set_yticks(np.arange(0, max(max(psnr_scores), max(mse_scores))+20, 5))
    ax1.bar_label(psnr_bar, padding=3, fmt='%.4f')
    ax1.bar_label(mse_bar, padding=3, fmt='%.4f')
    ax1.legend(loc=2)

    # ssim plot
    ax2.set_title('SSIM', weight='bold', fontsize=15)
    ssim_bar = ax2.bar(r1, ssim_scores, width=bar_width, color='red', align='center', edgecolor='black',
                       label='Median SSIM')
    ax2.set_xticks([r for r in range(len(psnr_scores))],
                   ['16-8-1 filter network', '32-16-1 filter network',
                    '64-32-1 filter network', '128-64-1 filter network',
                    '256-128-1 filter network'])
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.bar_label(ssim_bar, padding=3, fmt='%.6f')
    ax2.legend(loc=2)

    # save graph
    plt.savefig(os.path.join(ROOT_DIR, 'Data', 'plots', '{}_test_results_with_mse.svg'.format(plot_titles[id(test_set_results)]))
                ,format='svg', dpi=1200)
    plt.show()
