import os
import glob
from definitions import ROOT_DIR
import Test
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def average_scores(scores):
    for image in scores.keys():
        avg_psnr = np.mean([scores[image][i][0] for i, _ in enumerate(scores[image])])
        avg_ssim = np.mean([scores[image][i][1] for i, _ in enumerate(scores[image])])
        avg_mse = np.mean([scores[image][i][2] for i, _ in enumerate(scores[image])])
        scores[image] = (avg_psnr, avg_ssim, avg_mse)
    return scores


if __name__ == '__main__':
    SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')
    SET5_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set5')
    set14_results = dict()
    set5_results = dict()

    # for network in glob.glob(os.path.join(ROOT_DIR, 'outputs', '*')):
    #     network_filter_number = int(network.split('\\')[-1].split('_')[0])
    #     set14_model_scores = dict()
    #     set5_model_scores = dict()
    #     for model_state_dict in glob.glob(os.path.join(network, 'model*', 'model*.pth')):
    #         # set 14
    #         for image_path in glob.glob(os.path.join(SET14_PATH, '*.bmp')):
    #             image_name = image_path.split('\\')[-1].split('.')[0]
    #             test_score = (Test.test_srcnn(image_path, model_state_dict, network_filter_number))
    #             if image_name in set14_model_scores:
    #                 set14_model_scores[image_name].append(test_score)
    #             else:
    #                 set14_model_scores[image_name] = [test_score]
    #         # set 5
    #         for image_path in glob.glob(os.path.join(SET5_PATH, '*.bmp')):
    #             image_name = image_path.split('\\')[-1].split('.')[0]
    #             test_score = (Test.test_srcnn(image_path, model_state_dict, network_filter_number))
    #             if image_name in set5_model_scores:
    #                 set5_model_scores[image_name].append(test_score)
    #             else:
    #                 set5_model_scores[image_name] = [test_score]
    #     # print(set14_model_scores)
    #     # break
    #     set14_results[str(network_filter_number)] = average_scores(set14_model_scores)
    #     set5_results[str(network_filter_number)] = average_scores(set5_model_scores)
    #
    # best_models = dict()
    #
    # # find best model for each network
    # for network in glob.glob(os.path.join(ROOT_DIR, 'outputs', '*')):
    #     network_filter_number = int(network.split('\\')[-1].split('_')[0])
    #     best_model_score = 0
    #     best_model = None
    #     for model_state_dict in glob.glob(os.path.join(network, 'model*', 'model*.pth')):
    #         test_score = Test.main(SET14_PATH, model_state_dict, network_filter_number)[0]
    #         if test_score > best_model_score:
    #             best_model_score = test_score
    #             best_model = model_state_dict
    #     best_models[str(network_filter_number)] = best_model
    #
    # with open(os.path.join(ROOT_DIR, 'Data', 'best_models.pickle'), 'wb') as f:
    #     pickle.dump(best_models, f)
    #
    # for model_state_dict in best_models:
    #     network_filter_number = int(model_state_dict)
    #     set14_model_scores = dict()
    #     set5_model_scores = dict()
    #     # set 14
    #     for image_path in glob.glob(os.path.join(SET14_PATH, '*.bmp')):
    #         image_name = image_path.split('\\')[-1].split('.')[0]
    #         test_score = (Test.test_srcnn(image_path, best_models[model_state_dict], network_filter_number))
    #         set14_model_scores[image_name] = test_score
    #     # set 5
    #     for image_path in glob.glob(os.path.join(SET5_PATH, '*.bmp')):
    #         image_name = image_path.split('\\')[-1].split('.')[0]
    #         test_score = (Test.test_srcnn(image_path, best_models[model_state_dict], network_filter_number))
    #         set5_model_scores[image_name] = test_score
    # # print(set14_model_scores)
    # # break
    #     set14_results[str(network_filter_number)] = set14_model_scores
    #     set5_results[str(network_filter_number)] = set5_model_scores
    #
    # set14_bicubic_interpolation_scores = dict()
    # for image_path in glob.glob(os.path.join(SET14_PATH, '*.bmp')):
    #     image_name = image_path.split('\\')[-1].split('.')[0]
    #     test_score = (Test.test_interpolation(image_path))
    #     set14_bicubic_interpolation_scores[image_name] = test_score
    #
    # set5_bicubic_interpolation_scores = dict()
    # for image_path in glob.glob(os.path.join(SET5_PATH, '*.bmp')):
    #     image_name = image_path.split('\\')[-1].split('.')[0]
    #     test_score = (Test.test_interpolation(image_path))
    #     set5_bicubic_interpolation_scores[image_name] = test_score
    #
    # set14_results[str(0)] = set14_bicubic_interpolation_scores
    # set5_results[str(0)] = set5_bicubic_interpolation_scores
    # print(set14_results)
    # print(set5_results)
    # #
    # df_14_psnr = pd.DataFrame.from_dict({network: {image: set14_results[network][image][0] for image in set14_results[network]} for network in set14_results}, orient='columns')
    # #print(df_14_psnr)
    # df_14_ssim = pd.DataFrame.from_dict({network: {image: set14_results[network][image][1] for image in set14_results[network]} for network in set14_results}, orient='columns')
    # #print(df_14_ssim)
    # df_14_mse = pd.DataFrame.from_dict({network: {image: set14_results[network][image][2] for image in set14_results[network]} for network in set14_results}, orient='columns')
    # df_14_psnr = df_14_psnr[sorted(df_14_psnr.columns.tolist())]
    # df_14_ssim = df_14_ssim[sorted(df_14_ssim.columns.tolist())]
    # df_14_mse = df_14_mse[sorted(df_14_mse.columns.tolist())]
    # df_14_psnr.to_pickle(os.path.join(ROOT_DIR, 'Data', 'set14_heatmap_psnr_data.pkl'))
    # df_14_ssim.to_pickle(os.path.join(ROOT_DIR, 'Data', 'set14_heatmap_ssim_data.pkl'))
    # df_14_mse.to_pickle(os.path.join(ROOT_DIR, 'Data', 'set14_heatmap_mse_data.pkl'))
    # df_5_psnr = pd.DataFrame.from_dict({network: {image: set5_results[network][image][0] for image in set5_results[network]} for network in set5_results}, orient='columns')
    # df_5_ssim = pd.DataFrame.from_dict({network: {image: set5_results[network][image][1] for image in set5_results[network]} for network in set5_results}, orient='columns')
    # df_5_mse = pd.DataFrame.from_dict({network: {image: set5_results[network][image][2] for image in set5_results[network]} for network in set5_results}, orient='columns')
    # df_5_psnr = df_5_psnr[sorted(df_5_psnr.columns.tolist())]
    # df_5_ssim = df_5_ssim[sorted(df_5_ssim.columns.tolist())]
    # df_5_mse = df_5_mse[sorted(df_5_mse.columns.tolist())]
    # df_5_psnr.to_pickle(os.path.join(ROOT_DIR, 'Data', 'set5_heatmap_psnr_data.pkl'))
    # df_5_ssim.to_pickle(os.path.join(ROOT_DIR, 'Data', 'set5_heatmap_ssim_data.pkl'))
    # df_5_mse.to_pickle(os.path.join(ROOT_DIR, 'Data', 'set5_heatmap_mse_data.pkl'))

    df_14_psnr = pd.read_pickle(os.path.join(ROOT_DIR, 'Data', 'set14_heatmap_psnr_data.pkl'))
    df_14_ssim = pd.read_pickle(os.path.join(ROOT_DIR, 'Data', 'set14_heatmap_ssim_data.pkl'))
    df_14_mse = pd.read_pickle(os.path.join(ROOT_DIR, 'Data', 'set14_heatmap_mse_data.pkl'))
    df_5_psnr = pd.read_pickle(os.path.join(ROOT_DIR, 'Data', 'set5_heatmap_psnr_data.pkl'))
    df_5_ssim = pd.read_pickle(os.path.join(ROOT_DIR, 'Data', 'set5_heatmap_ssim_data.pkl'))
    df_5_mse = pd.read_pickle(os.path.join(ROOT_DIR, 'Data', 'set5_heatmap_mse_data.pkl'))

    df_14_psnr = df_14_psnr.rename(columns={'0': 'Bi-linear',
                                            '16': '16-8-1', '32': '32-16-1', '64': '64-32-1',
                                            '128': '128-64-1', '256': '256-128-1'})

    df_14_ssim = df_14_ssim.rename(columns={'0': 'Bi-linear',
                                            '16': '16-8-1', '32': '32-16-1', '64': '64-32-1',
                                            '128': '128-64-1', '256': '256-128-1'})

    df_14_mse = df_14_mse.rename(columns={'0': 'Bi-linear',
                                            '16': '16-8-1', '32': '32-16-1', '64': '64-32-1',
                                            '128': '128-64-1', '256': '256-128-1'})

    df_5_psnr = df_5_psnr.rename(columns={'0': 'Bi-linear',
                                            '16': '16-8-1', '32': '32-16-1', '64': '64-32-1',
                                            '128': '128-64-1', '256': '256-128-1'})

    df_5_ssim = df_5_ssim.rename(columns={'0': 'Bi-linear',
                                            '16': '16-8-1', '32': '32-16-1', '64': '64-32-1',
                                            '128': '128-64-1', '256': '256-128-1'})

    df_5_mse = df_5_mse.rename(columns={'0': 'Bi-linear',
                                          '16': '16-8-1', '32': '32-16-1', '64': '64-32-1',
                                          '128': '128-64-1', '256': '256-128-1'})

    # sns.set_theme()
    # f, ax = plt.subplots(2, 2, figsize=(19.2, 10.8))
    # f.suptitle('PSNR Heatmap', fontsize=20)
    #
    # sns.heatmap(data=df_14_psnr[['Bi-linear interpolation', '16-8-1', '32-16-1', '64-32-1', '128-64-1', '256-128-1']],
    #             annot=True, fmt=".3f", linewidths=.5, ax=ax[0][0], cmap='Reds')
    # ax[0][0].set_title('Set14 PSNR Results')
    # sns.heatmap(data=df_5_psnr[['Bi-linear interpolation', '16-8-1', '32-16-1', '64-32-1', '128-64-1', '256-128-1']],
    #             annot=True, fmt=".3f", linewidths=.5, ax=ax[1][0], cmap='Reds')
    # ax[1][0].set_title('Set5 PSNR Results')
    # #ax[1][0].set_yticklabels(ax[1][0].get_yticklabels(), rotation=90)
    # ax[1][0].tick_params(axis='x', rotation=90)
    #
    # sns.heatmap(data=df_14_ssim[['Bi-linear interpolation', '16-8-1', '32-16-1', '64-32-1', '128-64-1', '256-128-1']],
    #             annot=True, fmt=".3f", linewidths=.5, ax=ax[0][1], cmap='Reds')
    # ax[0][1].set_title('Set14 SSIM Results')
    # sns.heatmap(data=df_5_ssim[['Bi-linear interpolation', '16-8-1', '32-16-1', '64-32-1', '128-64-1', '256-128-1']],
    #             annot=True, fmt=".3f", linewidths=.5, ax=ax[1][1], cmap='Reds')
    # ax[1][1].set_title('Set5 SSIM Results')
    # # f.tight_layout()
    # plt.show()

    # sns.set_theme()
    # f, ax = plt.subplots(1, 2, figsize=(19.2, 10.8))
    # f.suptitle('Image Metric Heatmap Set5', fontsize=20)
    #
    # # sns.heatmap(data=df_14_psnr[['Bi-linear', '16-8-1', '32-16-1', '64-32-1', '128-64-1', '256-128-1']],
    # #             annot=True, fmt=".3f", linewidths=.5, ax=ax[0], cmap='Reds')
    # # ax[0].set_title('PSNR Results')
    # # ax[0].tick_params(axis='x', rotation=30)
    #
    # sns.heatmap(data=df_5_psnr[['Bi-linear', '16-8-1', '32-16-1', '64-32-1', '128-64-1', '256-128-1']],
    #             annot=True, fmt=".3f", linewidths=.5, ax=ax[0], cmap='Reds')
    # ax[0].set_title('PSNR Results')
    # #ax[1][0].set_yticklabels(ax[1][0].get_yticklabels(), rotation=90)
    # ax[0].tick_params(axis='x', rotation=30)
    #
    # # sns.heatmap(data=df_14_ssim[['Bi-linear', '16-8-1', '32-16-1', '64-32-1', '128-64-1', '256-128-1']],
    # #             annot=True, fmt=".3f", linewidths=.5, ax=ax[1], cmap='Reds')
    # # ax[1].set_title('SSIM Results')
    # # ax[1].tick_params(axis='x', rotation=30)
    #
    # sns.heatmap(data=df_5_ssim[['Bi-linear', '16-8-1', '32-16-1', '64-32-1', '128-64-1', '256-128-1']],
    #             annot=True, fmt=".3f", linewidths=.5, ax=ax[1], cmap='Reds')
    # ax[1].set_title('SSIM Results')
    # ax[1].tick_params(axis='x', rotation=30)
    # # # f.tight_layout()
    # plt.savefig(os.path.join(ROOT_DIR, 'Data', 'plots', 'Set5_hm.svg'), format='svg', dpi=1200)
    # plt.show()

    """
    ************ MSE Plots ********************************************************************************************
    """

    sns.set_theme()
    f, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
    f.suptitle('Image Metric Heatmap Set14', fontsize=20)

    sns.heatmap(data=df_14_mse[['Bi-linear', '16-8-1', '32-16-1', '64-32-1', '128-64-1', '256-128-1']],
                annot=True, fmt=".3f", linewidths=.5, ax=ax, cmap='Reds')
    ax.set_title('MSE Results')
    ax.tick_params(axis='x', rotation=30)

    # sns.heatmap(data=df_5_mse[['Bi-linear', '16-8-1', '32-16-1', '64-32-1', '128-64-1', '256-128-1']],
    #             annot=True, fmt=".3f", linewidths=.5, ax=ax, cmap='Reds')
    # ax.set_title('MSE Results')
    # #ax[1][0].set_yticklabels(ax[1][0].get_yticklabels(), rotation=90)
    # ax.tick_params(axis='x', rotation=30)


    plt.savefig(os.path.join(ROOT_DIR, 'Data', 'plots', 'Set14_hm_mse.svg'), format='svg', dpi=1200)
    plt.show()

