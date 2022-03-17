import pickle
import numpy as np
import pandas as pd
import os
from definitions import ROOT_DIR
import matplotlib.pyplot as plt
import argparse


def plot_loss_and_moving_average(df):
    df.plot.line(y=['Loss', 'Moving Average'])
    plt.show()


def main(path):
    with open(path, 'rb') as f:
        y_data = pickle.load(f)
    x_data = np.linspace(1, 1000, 1000)
    df = pd.DataFrame(data={'Loss': y_data}, index=x_data)
    df['Moving Average'] = df['Loss'].rolling(window=100, min_periods=10).mean()
    plot_loss_and_moving_average(df)
    best_epoch, min_avg_loss = df['Moving Average'].idxmin(), df['Moving Average'].min()
    filt = (df['Moving Average'] <= min_avg_loss)
    #return best_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    PATH = os.path.join(ROOT_DIR, args.path)
    main(PATH)
