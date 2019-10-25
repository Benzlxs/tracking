import os
import sys
import fire
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt



#####################################
####*****efficiency saving******#####
#####################################
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



#####################################
####*****ratio affect func******#####
#####################################
def plot_ratio_accuracy():
    total_num_points = 25
    x= [0.02*(i+1) for i in range(total_num_points)]

    # produce the detection accuracy
    det_car = [0.9882]*total_num_points
    det_cyc = [0.9422]*total_num_points
    det_ped = [0.7995]*total_num_points
    det_avg = [( 0.9882+0.9422+0.7995)/3.]*total_num_points

    # the detection performance with tracking
    trk_car = [0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9994, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995]
    trk_cyc = [0.9479, 0.9627, 0.9740, 0.9613, 0.9800, 0.9824, 0.9820, 0.9851, 0.9829, 0.9808, 0.9936, 0.9893, 0.9935, 0.9931, 0.9866, 0.9823, 0.9827, 0.9833, 0.9784, 0.9769, 0.9821, 0.9821, 0.9819, 0.9771, 0.9758]
    trk_ped = [0.8884, 0.9003, 0.9024, 0.8918, 0.9123, 0.9058, 0.9211, 0.9239, 0.9206, 0.9096, 0.9187, 0.9045, 0.8925, 0.8849, 0.8783, 0.8729, 0.8905, 0.8898, 0.8842, 0.8648, 0.8701, 0.8782, 0.8682, 0.8652, 0.8703]

    trk_avg = [np.mean( [trk_car[i], trk_cyc[i], trk_ped[i]] )  for i in range(len(trk_car))]

    fig, ax = plt.subplots()
    # plot the detection performance
    # plt.plot(x, det_car, color='r', linestyle='-', linewidth=2, label='Car_det')
    # plt.plot(x, det_cyc, color='g', linestyle='-', linewidth=2, label='Cyc_det')
    # plt.plot(x, det_ped, color='b', linestyle='-', linewidth=2, label='Ped_det')

    # # plot the tracking performance
    # plt.plot(x, trk_car, color='r', marker='o', linestyle='--', markersize=4, linewidth=2, label='Car_trk')
    # plt.plot(x, trk_cyc, color='g', marker='o', linestyle='--', markersize=4, linewidth=2, label='Cyc_trk')
    # plt.plot(x, trk_ped, color='b', marker='o', linestyle='--', markersize=4, linewidth=2, label='Ped_trk')
    # plt.plot(x, trk_avg, color='c', marker='s', linestyle='--', markersize=6, linewidth=2, label='Avg_trk')

    plt.plot(x, det_car[:total_num_points], color='r', linestyle='-', linewidth=2, label='Car_det')
    plt.plot(x, det_cyc[:total_num_points], color='g', linestyle='-', linewidth=2, label='Cyc_det')
    plt.plot(x, det_ped[:total_num_points], color='b', linestyle='-', linewidth=2, label='Ped_det')
    plt.plot(x, det_avg[:total_num_points], color='c', linestyle='-', linewidth=2, label='Avg_det')

    # plot the tracking performance
    plt.plot(x, trk_car[:total_num_points], color='r', marker='o', linestyle='--', markersize=6, linewidth=2, label='Car_trk')
    plt.plot(x, trk_cyc[:total_num_points], color='g', marker='o', linestyle='--', markersize=6, linewidth=2, label='Cyc_trk')
    plt.plot(x, trk_ped[:total_num_points], color='b', marker='o', linestyle='--', markersize=6, linewidth=2, label='Ped_trk')
    plt.plot(x, trk_avg[:total_num_points], color='c', marker='s', linestyle='--', markersize=8, linewidth=3, label='Avg_trk')



    plt.xlabel('\u03B1')
    plt.ylabel('mAP')
    plt.xlim(0, 0.7)
    # plt.ylim(0.65, 1.02)
    ax.legend()

    plt.savefig('ratio_map.png')

#####################################
####**start_fram affect func****#####
#####################################
def plot_start_frame_accuracy():
    total_num_points = 12
    x = [(i+1) for i in range(total_num_points)]

    # produce the detection accuracy
    det_car = [0.9882]*total_num_points
    det_cyc = [0.9422]*total_num_points
    det_ped = [0.7995]*total_num_points
    det_avg = [( 0.9882+0.9422+0.7995)/3.]*total_num_points

    # the detection performance with tracking
    trk_car = [0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9994, 0.9992, 0.9990, 0.9991, 0.9984, 0.9992, 0.9990, 0.9988, 0.9988, 0.9986]
    trk_cyc = [0.9863, 0.9863, 0.9851, 0.9857, 0.9814, 0.9796, 0.9533, 0.9548, 0.9668, 0.9381, 0.9272, 0.9356, 0.8791, 0.9208, 0.9315, 0.9381]
    trk_ped = [0.8921, 0.8921, 0.9239, 0.9075, 0.8789, 0.8467, 0.8321, 0.8556, 0.7976, 0.8651, 0.8427, 0.7969, 0.8173, 0.8491, 0.8134, 0.7801]

    trk_avg = [np.mean( [trk_car[i], trk_cyc[i], trk_ped[i]] )  for i in range(len(trk_car))]

    fig, ax = plt.subplots()

    plt.plot(x, det_car[:total_num_points], color='m', linestyle='-', linewidth=2, label='Car_det')
    plt.plot(x, det_cyc[:total_num_points], color='y', linestyle='-', linewidth=2, label='Cyc_det')
    plt.plot(x, det_ped[:total_num_points], color='k', linestyle='-', linewidth=2, label='Ped_det')
    plt.plot(x, det_avg[:total_num_points], color='c', linestyle='-', linewidth=2, label='Avg_det')

    # plot the tracking performance
    plt.plot(x, trk_car[:total_num_points], color='m', marker='o', linestyle='--', markersize=6, linewidth=2, label='Car_trk')
    plt.plot(x, trk_cyc[:total_num_points], color='y', marker='o', linestyle='--', markersize=6, linewidth=2, label='Cyc_trk')
    plt.plot(x, trk_ped[:total_num_points], color='k', marker='o', linestyle='--', markersize=6, linewidth=2, label='Ped_trk')
    plt.plot(x, trk_avg[:total_num_points], color='c', marker='s', linestyle='--', markersize=8, linewidth=3, label='Avg_trk')

    plt.xlabel('Start frames')
    plt.ylabel('mAP')
    plt.xlim(0, total_num_points+5)
    # plt.ylim(0.65, 1.02)
    ax.legend()

    plt.savefig('start_frames_map.png')







if __name__=='__main__':
    fire.Fire()
