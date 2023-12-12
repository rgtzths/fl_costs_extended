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
import pandas as pd
from functools import partial


def plot_model_history(input_folder, file_name):

    plt.rcParams.update({'font.size': 22})
    output = pathlib.Path(input_folder)

    model_history = json.load(open(output/ file_name))

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
                if key == "mcc_val" or key == "mcc":
                    print("%6.2f %6.2f" %(y, points[i][2] / 60))

            plt.xlabel("Nº Epochs")
            plt.ylabel(key)
            plt.title("Evolution of the "+key+" of the model")
            plt.legend()

            fig.savefig(output /(key+".pdf"), dpi=300.0, bbox_inches='tight', format="pdf", orientation="landscape")

    plt.close('all')

def get_average_times(folder, n_batches):
    output = pathlib.Path(folder)
    plt.rcParams.update({'font.size': 16})

    results_list = []
    for folder in output.glob("*"):
        if folder.is_dir():
            average_results = {
                "train" : [],
                "comm_send" : [],
                "comm_recv" : [],
                "conv_send" : [],
                "conv_recv" : [],
                "epochs" : []
            }
            for file in folder.glob("**/worker*"):
                model_history = json.load(open(file))
                for key in average_results:
                    if key != "epochs":
                        if "decentralized" in str(folder):
                            average_results[key].append(sum(model_history["times"][key])/len(model_history["times"][key]))
                        else:
                            temp_array = []
                            for i in range(0, len(model_history["times"][key]), n_batches):
                                temp_array.append(sum(model_history["times"][key][i:i+n_batches]))
                            average_results[key].append(sum(temp_array)/len(temp_array))
            
            results = [np.log10(10000*sum(average_results[key])/len(average_results[key]) )for key in average_results if key != "epochs" ]
            #results = [sum(average_results[key])/len(average_results[key])  for key in average_results if key != "epochs" ]

            results_list.append([str(folder).split("/")[-1].replace("_", " ").capitalize()] + results)

    df = pd.DataFrame(results_list,
                  columns=['FL approach', 'Training', 'Conv. send', 'Comm. send.', 'Comm. recv.', 'Conv. recv.'])
    
    ax = df.plot(x='FL approach', kind='bar', stacked=True, figsize=(12, 8))
    plt.xticks(rotation=0)
    ax.get_yaxis().set_ticklabels([]) 
    ax.get_figure().savefig(output /("times_comparison.pdf"), dpi=300.0, bbox_inches='tight', format="pdf", orientation="landscape")

    plt.close('all')

def get_average_times_single(folder_fl, folder_single, n_batches):
    output = pathlib.Path(folder_fl)
    plt.rcParams.update({'font.size': 16})

    results_list = []
    average_results = {
        "train" : [],
        "comm_send" : [],
        "comm_recv" : [],
        "conv_send" : [],
        "conv_recv" : [],
        "epochs" : []
    }
    for file in output.glob("**/worker*"):
        model_history = json.load(open(file))
        for key in average_results:
            if key != "epochs":
                if "decentralized" in str(output):
                    average_results[key].append(sum(model_history["times"][key])/len(model_history["times"][key]))
                else:
                    temp_array = []
                    for i in range(0, len(model_history["times"][key]), n_batches):
                        temp_array.append(sum(model_history["times"][key][i:i+n_batches]))
                    average_results[key].append(sum(temp_array)/len(temp_array))
    
    results = [np.log10(10000*sum(average_results[key])/len(average_results[key]) )for key in average_results if key != "epochs" ]

    results_list.append([str(output).split("/")[-1].replace("_", " ").capitalize()] + results)

    single_folder = pathlib.Path(folder_single)

    model_history = json.load(open(single_folder/ "train_history.json"))

    times = [model_history["times"][0]]
    for idx, time in enumerate(model_history["times"][1:]):
        times.append(time - model_history["times"][idx-1])

    results = [0]*len(results)
    results[0] = np.log10(10000*sum(times)/len(times))
    results_list.append(["Single host"] + results)
    df = pd.DataFrame(results_list,
                  columns=['Training approach', 'Training', 'Conv. send', 'Comm. send.', 'Comm. recv.', 'Conv. recv.'])
    
    ax = df.plot(x='Training approach', kind='bar', stacked=True, figsize=(12, 8))
    plt.xticks(rotation=0)
    ax.get_yaxis().set_ticklabels([]) 
    ax.get_figure().savefig(output /("times_comparison.pdf"), dpi=300.0, bbox_inches='tight', format="pdf", orientation="landscape")

    plt.close('all')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the model')
    parser.add_argument('-f', type=str, help='Input/output folder', default='results/')
    parser.add_argument('-s', type=str, help='Server file name', default='server.json')
    parser.add_argument('-n', type=bool, help='Number of batches', default=8)
    parser.add_argument('-g', type=str, help='Graphic type', default="bar")


    args = parser.parse_args()

    if args.g == "linear":
        plot_model_history(args.f, args.s)
    elif args.g == "bar":
        # If IOT and 128 batch size n_batch = 299
        # If slicing n_batch=292
        get_average_times(args.f, args.n)
    else:
        get_average_times_single(args.f, args.s, args.n)
    