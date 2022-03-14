import pickle
import numpy as np
import os
from definitions import ROOT_DIR
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import argparse


def main(path, curve_type):
    with open(path, 'rb') as f:
        y_data = pickle.load(f)
    x_data = np.linspace(1, 1000, 1000)
    exp_func = exp_decay if curve_type == 'decay' else log_growth
    params, _ = curve_fit(exp_func, x_data, y_data, maxfev=5000)
    ans = [exp_func(x, *params) for x in x_data]
    plt.scatter(x_data, y_data, s=5, c='red')
    plt.plot(x_data, ans)
    plt.show()


def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def log_growth(x, a, b):
    return a * np.log(x) + b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--curve-type', choices=['decay', 'growth'], default='growth', required=True)
    args = parser.parse_args()
    main(os.path.join(ROOT_DIR, os.path.normpath(args.path)), args.curve_type)
