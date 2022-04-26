import os
import glob
from definitions import ROOT_DIR
import Test
import numpy as np
import pandas as pd
import openpyxl


def average_scores(scores):
    for image in scores.keys():
        scores[image] = np.mean(scores[image])
    return scores


if __name__ == '__main__':
    SET14_PATH = os.path.join(ROOT_DIR, 'testSets', 'Set14')
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
        results[network_filter_number] = average_scores(model_scores)

    df = pd.DataFrame.from_dict(results, orient='columns')
    df = df[sorted(df.columns.tolist())]
    df.to_excel(os.path.join(ROOT_DIR, 'Data', 'Set14_individual_image_scores.xlsx'),
                sheet_name='Set14_individual_image_scores', index=True)
