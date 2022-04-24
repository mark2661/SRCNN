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
import numpy as np


def calculate_five_number_summary(data):
    q1, median, q3 = percentile(data, [25, 50, 75])
    minimum, maximum = min(data), max(data)
    return minimum, q1, median, q3, maximum


if __name__ == '__main__':

    SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')
    model_averages_dict = dict()

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

        # add the PSNR averages for each model to model averages_dict
        model_averages_dict[network_filter_number] = model_averages

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
        wb.save(filename=os.path.join(ROOT_DIR, 'Data', 'five_number_summary.xlsx'))

    # create a pandas data frame with average test PSNR for each model for each network for box plots
    df = pd.DataFrame.from_dict(model_averages_dict, orient='columns')
    df = df[sorted(df.columns.tolist())]

    # code for box plot of data frame data
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    g = sns.boxplot(data=df, width=0.7)
    sns.stripplot(data=df, color='black')
    plt.title("Five Number Summary Box Plots From Average PSNR Score on Set14 Test Set", fontsize=20, weight='bold')
    xvalues = ['16-8-1 filter network', '32-16-1 filter network', '64-32-1 filter network', '128-64-1 filter network',
               '256-128-1 filter network']
    plt.xticks(np.arange(5), xvalues)
    # plt.yticks(np.arange(df.min().min(), df.max().max()))
    g.set(ylabel="Average PSNR")

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
    #plt.show()
