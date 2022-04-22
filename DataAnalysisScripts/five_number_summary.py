# calculate a 5-number summary
from numpy import percentile
import pandas as pd
import argparse
import os
import openpyxl
import glob
import Test
import re
from definitions import ROOT_DIR
from Test import main as test


def calculate_five_number_summary(data):
    q1, median, q3 = percentile(data, [25, 50, 75])
    minimum, maximum = min(data), max(data)
    return minimum, q1, median, q3, maximum


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--network-path', type=str, required=True)
    # parser.add_argument('--network-filter-number', type=int, required=True)
    #
    # args = parser.parse_args()
    # NETWORK_DIR_PATH = args.network_path
    # NETWORK_FILTER_NUMBER = args.network_filter_number
    SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')

    for network in glob.glob(os.path.join(ROOT_DIR, 'outputs', '*')):
        network_filter_number = int(network.split('\\')[-1].split('_')[0])
        models = []
        model_averages = []
        print(network_filter_number)

        for model_state_dict in glob.glob(os.path.join(network, 'model*', 'model*.pth')):
            models.append(model_state_dict)

        for model_state_dict in models:
            model_averages.append(Test.main(SET14_PATH, model_state_dict, network_filter_number))

        if os.path.exists(os.path.join(os.getcwd(), 'five_number_summery.xlsx')):
            wb = openpyxl.load_workbook(filename='five_number_summery.xlsx')
        else:
            wb = openpyxl.Workbook()
            page = wb.active
            page.title = 'Five Number Summary'
            page.append(['Network', 'Minimum', 'First quartile', 'Median', 'Second quartile', 'Maximum'])

        page = wb.active
        page.append([network_filter_number] + list(calculate_five_number_summary(model_averages)))
        wb.save(filename='five_number_summery.xlsx')






