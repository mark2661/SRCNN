# calculate a 5-number summary
from numpy import percentile
import pandas as pd
import argparse
import os
import glob
import re
from definitions import ROOT_DIR
from Test import main as test


def calculate_five_number_summary(data):
    q1, median, q3 = percentile(data, [25, 50, 75])
    minimum, maximum = min(data), max(data)
    return minimum, q1, median, q3, maximum


if __name__ == '__main__':
    models = []
    TEST_SET_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')
    # for network_dir in os.listdir(os.path.join(ROOT_DIR, 'outputs', '64_network')):
    #     for model_dir in os.listdir(os.path.join(ROOT_DIR, 'outputs', network_dir, '64_network')):
    #         for file in glob.glob(os.path.join(ROOT_DIR, 'outputs', network_dir, model_dir, '*.pth')):
    #             models.append(file)
    #     break
    # avg = [test(TEST_SET_PATH, model) for model in models]
    # print(avg)

    for model_dir in os.listdir(os.path.join(ROOT_DIR, 'outputs', '64_network')):
        for file in glob.glob(os.path.join(ROOT_DIR, 'outputs', '64_network', model_dir, '*.pth')):
            models.append(file)

    avg = [test(TEST_SET_PATH, model, 64) for model in models]
    print(calculate_five_number_summary(avg))


    # five_num_sums = [calculate_five_number_summary(data) for data in models]
    # five_num_sum_df = pd.DataFrame(five_num_sums, columns=['Minimum', '1st Quartile', 'Median', '3rd Quartile',
    #                                                        'Maximum'])
    # averages = pd.DataFrame([five_num_sum_df[column].mean(axis=0) for column in five_num_sum_df],
    #                         columns=['Minimum', '1st Quartile', 'Median', '3rd Quartile', 'Maximum'], index=['average'])
    # print(five_num_sum_df)
    # print(averages)
    # five_num_sum_df = pd.concat([five_num_sum_df, averages], axis=0)
