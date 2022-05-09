import pandas as pd
import numpy as np
import openpyxl
import os
import glob
import Test
from definitions import ROOT_DIR
from scipy.stats import mannwhitneyu


if __name__ == '__main__':

    SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')
    network_test_scores = dict()

    for network in glob.glob(os.path.join(ROOT_DIR, 'outputs', '*')):
        network_filter_number = int(network.split('\\')[-1].split('_')[0])
        model_test_scores = []

        for model_state_dict in glob.glob(os.path.join(network, 'model*', 'model*.pth')):
            # model_test_scores.append(Test.main(SET14_PATH, model_state_dict, network_filter_number, median=True)[0])
            model_test_scores.append(Test.main(SET14_PATH, model_state_dict, network_filter_number, median=True)[2])

        network_test_scores[network_filter_number] = model_test_scores

    df = pd.DataFrame.from_dict(network_test_scores, orient='columns')
    df = df[sorted(df.columns.tolist())]
    print(df)
    #print(type(list(df)[0]))

    # # create excel workbook to log results
    # wb = openpyxl.Workbook()
    # page = wb.active
    # page.title = 'Mann-Whitney-U results'
    # page.append(['Network 1', 'Network 2', 'P value', 'U value'])
    #
    # # perform pairwise Mann-Whitney-U test
    # df_column_names = list(df)
    # for i in range(len(df_column_names)-1):
    #     for j in range(i+1, len(df_column_names)):
    #         U, p_score = mannwhitneyu(df[df_column_names[i]], df[df_column_names[j]])
    #         page.append(['{}-{}-1'.format(df_column_names[i], df_column_names[i]//2),
    #                      '{}-{}-1'.format(df_column_names[j], df_column_names[j]//2), p_score, U])
    # wb.save(filename=os.path.join(ROOT_DIR, 'Data', 'mann-whitney-u-results_mse.xlsx'))

