import os
import sys
import fire
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt



def read_efficiency_txt(result_file):
    with open(result_file, 'r') as f:
        lines = f.readlines()
    results = [line.strip().split(',') for line in lines]

    return results



def plot_efficiency_saving(result_file,
                           folder_name='2011_09_26',
                           phases=['0001','0020','0035','0084']):
    """
    Reading the result file,
    Plot the ratio with different name
    """
    results = read_efficiency_txt(result_file)
    fig, ax = plt.subplots()
    for phase in phases:
        dist = []
        ratios = []
        file_name = '{}_drive_{}_sync'.format(folder_name, phase)
        for j in range(len(results)):
            if results[j][0] == file_name:
                dist.append(float(results[j][1]))
                ratios.append(float(results[j][4])/float(results[j][3]))

        plt.plot(dist, ratios, marker='o', linestyle='dashed', markersize=12, linewidth=2, label='{}'.format(phase))

    plt.xlabel('Range of interest')
    plt.ylabel('Ratio')
    ax.legend()
    fig_name = result_file.replace('.txt','.png')
    plt.savefig(fig_name)






if __name__=='__main__':
    fire.Fire()
