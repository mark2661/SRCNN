import os
import matplotlib.pyplot as plt
import pickle
from definitions import ROOT_DIR
import numpy as np


with open(os.path.join(ROOT_DIR, 'Data', 'real_set14_test_result.pkl'), 'rb') as f:
    psnr_results = pickle.load(f)
with open(os.path.join(ROOT_DIR, 'Data', 'test_results.pickle'), 'rb') as f:
    test_results = pickle.load(f)


bsds_results = {key: test_results[key][0] for key in sorted(test_results.keys())}
set5_results = {key: test_results[key][1] for key in sorted(test_results.keys())}
set14_results = {key: test_results[key][2] for key in sorted(test_results.keys())}
urban_results = {key: test_results[key][3] for key in sorted(test_results.keys())}

test_sets = [set5_results, set14_results, bsds_results, urban_results]
plot_titles = {id(set5_results): 'Set14', id(set14_results): 'Set5', id(bsds_results): 'BSDS100',
               id(urban_results): 'Urban100'}
print(psnr_results)






# PSNR and MSE bar charts

fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False, figsize=(19.2, 10.8))

# # PSNR and MSE bar charts
psnr_scores = [psnr_results[key] for key in sorted(psnr_results.keys())]
ssim_scores = [set5_results[key][1] for key in sorted(set5_results.keys())]
#mse_scores = [test_set_results[key][2] for key in sorted(test_set_results.keys())]

fig.suptitle('{} Median Test Scores'.format('Set14'), weight='bold', fontsize=20)
# create bar charts
bar_width = 0.3
ax1.set_title('PSNR (dB)', weight='bold', fontsize=15)

# x-axis spacing
r1 = np.arange(len(psnr_scores))
r2 = [x + bar_width for x in r1]

# Create blue bars
psnr_bar = ax1.bar(r1, psnr_scores, width=bar_width, color='blue', align='center', edgecolor='black',
                       label='Median PSNR')

# Create cyan bars
# mse_bar = ax1.bar(r2, mse_scores, width=bar_width, color='orange', align='center', edgecolor='black',
#                   label='Average MSE')

# general layout
ax1.set_xticks([r for r in range(len(psnr_scores))], ['16-8-1 filter network',
                                                                    '32-16-1 filter network',
                                                                    '64-32-1 filter network',
                                                                    '128-64-1 filter network',
                                                                    '256-128-1 filter network'])
# plt.ylim(0, max(max(psnr_scores), max(mse_scores)))
ax1.set_yticks(np.arange(0, max(psnr_scores)+15, 5))
ax1.bar_label(psnr_bar, padding=3, fmt='%.4f')
#ax1.bar_label(mse_bar, padding=3)
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
ax2.bar_label(ssim_bar, padding=3)
ax2.legend(loc=2)

# save graph
plt.savefig(os.path.join(ROOT_DIR, 'Data', 'plots', '{}_test_results.svg'.format('set14'))
            ,format='svg', dpi=1200)
plt.show()
