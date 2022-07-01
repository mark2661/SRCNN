# calculate a 5-number summary
from numpy import percentile
import pandas as pd
import os
import openpyxl
import glob
import Test
from definitions import ROOT_DIR
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np


def calculate_five_number_summary(data):
    q1, median, q3 = percentile(data, [25, 50, 75])
    minimum, maximum = min(data), max(data)
    return minimum, q1, median, q3, maximum


if __name__ == '__main__':

    SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')
    model_averages_dict = dict()
    psnr_s = dict()
    mse_s = dict()

    for network in glob.glob(os.path.join(ROOT_DIR, 'outputs', '*')):
        network_filter_number = int(network.split('\\')[-1].split('_')[0])
        models = []
        model_averages = []
        print(network_filter_number)

        # for each model in the network folder append the .pth file containing the model state dict to models
        for model_state_dict in glob.glob(os.path.join(network, 'model*', 'model*.pth')):
            models.append(model_state_dict)

        # for each model in the network folder calculate the average PSNR for the test set
        for model_state_dict in models:
            model_averages.append(Test.main(SET14_PATH, model_state_dict, network_filter_number)[0])

        # for each model in the network folder calculate the average MSE for the test set
        # for model_state_dict in models:
        #     model_averages.append(Test.main(SET14_PATH, model_state_dict, network_filter_number)[2])

        # add the PSNR medians for each model to model averages_dict
        model_averages_dict[network_filter_number] = model_averages
        psnr_s[network_filter_number] = calculate_five_number_summary(model_averages)[2]

        # # add the MSE medians for each model to model averages_dict
        # model_averages_dict[network_filter_number] = model_averages
        # mse_s[network_filter_number] = calculate_five_number_summary(model_averages)[2]
        #print(mse_s)
        print(model_averages)
        print([network_filter_number] + list(calculate_five_number_summary(model_averages)))
        # calculate the five number summary and append results to five_number_summary.xlsx
        if os.path.exists(os.path.join(os.getcwd(), 'five_number_summary.xlsx')):
            wb = openpyxl.load_workbook(filename='five_number_summary.xlsx')
        else:
            wb = openpyxl.Workbook()
            page = wb.active
            page.title = 'Five Number Summary'
            page.append(['Network', 'Minimum', 'First quartile', 'Median', 'Second quartile', 'Maximum'])

        page = wb.active
        page.append([network_filter_number] + list(calculate_five_number_summary(model_averages)))
        wb.save(filename=os.path.join(ROOT_DIR, 'Data', 'five_number_summary_mse.xlsx'))


    # create a pandas data frame with average test PSNR for each model for each network for box plots
    df = pd.DataFrame.from_dict(model_averages_dict, orient='columns')
    df = df[sorted(df.columns.tolist())]
    print(df)
    #df.to_pickle(os.path.join(ROOT_DIR, 'Data', 'real_set14_test_result.pkl'))
    with open(os.path.join(ROOT_DIR, 'Data', 'real_set14_test_result.pkl'), 'wb') as f:
        pickle.dump(mse_s, f)




    # # code for box plot of data frame data
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    g = sns.boxplot(data=df, width=0.7, showfliers=False)
    sns.stripplot(data=df, color='black')
    plt.title("Five Number Summary Box Plots of Set14 PSNR Scores", fontsize=20, weight='bold')
    xvalues = ['16-8-1 filter network', '32-16-1 filter network', '64-32-1 filter network', '128-64-1 filter network',
               '256-128-1 filter network']
    plt.xticks(np.arange(5), xvalues, fontsize=16)
    plt.yticks(fontsize=16)
    g.set(ylabel="PSNR (dB)")

    # remove all borders except bottom
    sns.despine(top=True,
                right=True,
                left=True,
                bottom=False)

    # Set colors of box plots
    palette = ['plum', 'g', 'orange', 'b', 'r']
    color_dict = dict(zip(xvalues, palette))
    for i in range(0, 5):
        mybox = g.artists[i]
        mybox.set_facecolor(color_dict[xvalues[i]])
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, 'Data', 'five_num_sum_boxplots.svg'), format='svg', dpi=1200)
    plt.show()

    """
    ******************* MSE plots **********************************************
    """

    # # # code for box plot of data frame data
    # sns.set(style='whitegrid')
    # fig, ax = plt.subplots(figsize=(19.2, 10.8))
    # g = sns.boxplot(data=df, width=0.7, showfliers=False)
    # sns.stripplot(data=df, color='black')
    # plt.title("Five Number Summary Box Plots of Set14 MSE Scores", fontsize=20, weight='bold')
    # xvalues = ['16-8-1 filter network', '32-16-1 filter network', '64-32-1 filter network', '128-64-1 filter network',
    #            '256-128-1 filter network']
    # plt.xticks(np.arange(5), xvalues, fontsize=16)
    # plt.yticks(fontsize=16)
    # g.set(ylabel="MSE")
    #
    # # remove all borders except bottom
    # sns.despine(top=True,
    #             right=True,
    #             left=True,
    #             bottom=False)
    #
    # # Set colors of box plots
    # palette = ['plum', 'g', 'orange', 'b', 'r']
    # color_dict = dict(zip(xvalues, palette))
    # for i in range(0, 5):
    #     mybox = g.artists[i]
    #     mybox.set_facecolor(color_dict[xvalues[i]])
    # plt.tight_layout()
    # #plt.savefig(os.path.join(ROOT_DIR, 'Data', 'five_num_sum_boxplots_mse.svg'), format='svg', dpi=1200)
    # #plt.show()
