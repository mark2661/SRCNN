# calculate a 5-number summary
from numpy import percentile
import pandas as pd
import argparse
import os
import glob
import re
from definitions import ROOT_DIR


def calculate_five_number_summary(data):
    q1, median, q3 = percentile(data, [25, 50, 75])
    minimum, maximum = min(data), max(data)
    return minimum, q1, median, q3, maximum



if __name__ == '__main__':
    models = []
    for network_dir in os.listdir(os.path.join(ROOT_DIR, 'outputs')):
        for model_dir in os.listdir(os.path.join(ROOT_DIR, 'outputs', network_dir)):
            for file in glob.glob(os.path.join(ROOT_DIR, 'outputs', network_dir, model_dir, '*.pth')):
                models.append(file)
        break
    print(models)

    # five_num_sums = [calculate_five_number_summary(data) for data in models]
    # five_num_sum_df = pd.DataFrame(five_num_sums, columns=['Minimum', '1st Quartile', 'Median', '3rd Quartile',
    #                                                        'Maximum'])
    # averages = pd.DataFrame([five_num_sum_df[column].mean(axis=0) for column in five_num_sum_df],
    #                         columns=['Minimum', '1st Quartile', 'Median', '3rd Quartile', 'Maximum'], index=['average'])
    # print(five_num_sum_df)
    # print(averages)
    #five_num_sum_df = pd.concat([five_num_sum_df, averages], axis=0)

