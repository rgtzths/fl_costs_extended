#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Rafael Teixeira'
__version__ = '0.1'
__email__ = 'rafaelgteixeira@ua.pt'
__status__ = 'Development'

import argparse
import json
import pathlib

import numpy as np
from matplotlib import pyplot as plt


def plot_model_history(results_folder, history_file):

    plt.rcParams.update({'font.size': 22})
    folder = pathlib.Path(results_folder)

    model_history = json.load(open(folder/history_file))

    for key in model_history:
        if key != "times":
            state = 0
            points = []
            values = list(np.arange(0.50,1,0.05))
            for i in range(len(model_history[key])):
                if model_history[key][i] >= values[state]:
                    while state < len(values) and model_history[key][i] >= values[state]:
                        state += 1
                    try:
                        points.append((i+1, model_history[key][i], model_history["times"]["global_times"][i]))
                    except:
                        points.append((i+1, model_history[key][i], model_history["times"][i]))

                    if state == len(values):
                        break

            n_epochs = len(model_history[key])

            fig = plt.figure(figsize=(12, 8))

            plt.plot(range(1, n_epochs+1), model_history[key], label="Validation " +key)
            for i in range(len(points)):
                x = points[i][0]
                y = points[i][1]
                plt.plot(x, y, 'bo')
                plt.text(x + 2.5,  y-0.02, "%.2f" % (points[i][2] / 60), fontsize=18)

            plt.xlabel("Nº Epochs")
            plt.ylabel(key)
            plt.title("Evolution of the "+key+" of the model")
            plt.legend()

            fig.savefig(folder /(key+".pdf"), dpi=300.0, bbox_inches='tight', format="pdf", orientation="landscape")

    plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the model')
    parser.add_argument('-f', type=str, help='Folder for the results', default='results/')
    parser.add_argument('-j', type=str, help='Json file to analyze', default='server.json')
    args = parser.parse_args()

    plot_model_history(args.f, args.j)