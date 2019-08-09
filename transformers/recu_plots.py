#!/usr/bin/python
# coding: UTF-8
from __future__ import division, print_function
import pylab as plt
from scipy.spatial.distance import pdist, squareform
import pyedflib
import numpy as np
from datetime import datetime
import os

def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z


def moving_average(s, r=5):
    return np.convolve(s, np.ones((r,))/r, mode='valid')


def get_data(current_edf_file):
    print("Started processing of file ", current_edf_file, " at time :", '{:%H:%M:%S}'.format(datetime.now()))
    data = pyedflib.EdfReader('chb01_01.edf')
    print(int(data.file_duration))
    print(int(data.samplefrequency(0)))
    num_signals = 18
    # data.signals_in_file should be #from pyedflib documentation but we want to choose
    # the 18 channels to use from 0-22
    # num_channels = 256

    num_samples = int(data.getNSamples()[0])  # number of samples
    print(num_samples)
    eeg_data = np.zeros((num_signals, num_samples), dtype=np.float)  # create receptacle for the actual data

    # the following block carries out transformations to ensure same locations across montages
    electrode_array = get_electrodes('chb01_04.edf') # private owned code.

    for electrode_counter in range(num_signals):  # only copy 18 electrode locations
        eeg_data[electrode_counter, :] = data.readSignal(electrode_array[electrode_counter])
    data._close()
    eeg_data = eeg_data.transpose()
    print(eeg_data.shape)
    processed_data = np.zeros((3600, 256, 18))
    for i in range(18):
        _data = eeg_data[:, i]
        _data = _data.reshape(-1, 256)
        processed_data[:, :, i] = _data

    print('Job all done, boss!' + '   with shape: ' + str(processed_data.shape))
    return processed_data
if __name__ == '__main__':
    chb_files = ['chb01_01.edf']
    chb = get_data(chb_files)
    eps = 0.1
    steps = 10

    # plotting randomly
    ru = chb[:, 6, 16]
    ru_filtered = moving_average(ru)

    plt.title("Normal")
    plt.subplot(221)
    plt.plot(ru_filtered)
    plt.title("EEG One Channel")
    plt.subplot(223)
    plt.imshow(rec_plot(ru_filtered, eps=eps, steps=steps))
    plt.savefig('k1')
    # Plot normal dist filtered with moving average
    rn = chb[:, 7, 20]
    rn_filtered = moving_average(rn)

    plt.subplot(222)
    plt.plot(rn_filtered)
    plt.title("Another channel")
    plt.subplot(224)
    plt.imshow(rec_plot(rn_filtered, eps=eps, steps=steps))
    plt.savefig('k2')
    plt.show()