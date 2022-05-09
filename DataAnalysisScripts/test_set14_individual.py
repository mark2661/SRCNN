import os
import glob
from definitions import ROOT_DIR
import Test
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns


def average_scores(scores):
    for image in scores.keys():
        scores[image] = np.mean(scores[image])
    return scores


if __name__ == '__main__':
    SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set5')
    results = dict()

    for network in glob.glob(os.path.join(ROOT_DIR, 'outputs', '*')):
        network_filter_number = int(network.split('\\')[-1].split('_')[0])
        model_scores = dict()
        for model_state_dict in glob.glob(os.path.join(network, 'model*', 'model*.pth')):
            for image_path in glob.glob(os.path.join(SET14_PATH, '*.bmp')):
                image_name = image_path.split('\\')[-1].split('.')[0]
                test_score = Test.test_srcnn(image_path, model_state_dict, network_filter_number)[0]
                if image_name in model_scores:
                    model_scores[image_name].append(test_score)
                else:
                    model_scores[image_name] = [test_score]
        results[str(network_filter_number)] = average_scores(model_scores)

    bicubic_interpolation_scores = dict()
    for image_path in glob.glob(os.path.join(SET14_PATH, '*.bmp')):
        image_name = image_path.split('\\')[-1].split('.')[0]
        test_score = Test.test_interpolation(image_path)[0]
        bicubic_interpolation_scores[image_name] = test_score

    results[str(0)] = bicubic_interpolation_scores
    # print(results)

    for network in results.keys():
        if network == '0':
            continue
        else:
            for image in results[network].keys():
                results[network][image] -= results['0'][image]
    # print(results)
    # results['SRCNN'] = {'baboon': 0.13, 'barbara': 0.27, 'bridge': 0.25, 'coastguard': 0.18, 'comic': 0.49,
    #                     'face': 0.47, 'flowers': 0.72, 'foreman': 1.31, 'lenna': 0.75, 'man': 0.42, 'monarch': 1.68,
    #                     'pepper': 1.03, 'ppt3': 1.04, 'zebra': 0.92}
    df = pd.DataFrame.from_dict(results, orient='columns')
    df = df[sorted(df.columns.tolist())]
    df.to_pickle('./set5_individual_results.pkl')
    df = pd.read_pickle('./set5_individual_results.pkl')
    print(df)
    # #df_t = df[[16, 32, 64, 128, 256]].transpose()
    # sns.set_theme()
    # f, ax = plt.subplots(figsize=(9, 6))
    # f.suptitle('Set14 PSNR Gains (dB)', fontsize=20)
    # #cmap = sns.palplot(sns.color_palette("coolwarm", 12))
    # sns.heatmap(data=df[['16', '32', '64', '128', '256', 'SRCNN']], annot=True, fmt=".2f", linewidths=.5, ax=ax,
    #             cmap='coolwarm')
    # #f.tight_layout()
    # plt.show()

df.to_excel(os.path.join(ROOT_DIR, 'Data', 'Set5_individual_image_scores.xlsx'),
            sheet_name='Set5_individual_image_scores', index=True)
