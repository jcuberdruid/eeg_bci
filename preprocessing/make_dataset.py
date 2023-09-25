'''
############################################
the point of this program: to create a processed dataset from eeg data 

steps: 
    1. take in parameters for dataset name 
    2. parse raw data paths 
    3. import raw data into python mne and run preprocessing 
    4. output organized dataset into readible files 

'''
import utils as ut
import os
import shutil
import argparse
import click
import mne 
import pandas as pd
import numpy as np

raw_data_path = "../data/raw/"
datasets_path = "../data/datasets"

def parse_arguments():
    parser = argparse.ArgumentParser(description='program for making a processed dataset')
    parser.add_argument('-name', dest='dataset_name', action='store_true', help='the name of the dataset')
    args = parser.parse_args()
    return args

def make_dir_structure(name):
    new_dataset_path = os.path.join(datasets_path, name)
    if os.path.isdir(new_dataset_path):
        if click.confirm('Dataset name taken, would you like to overwrite it?', default=False):
            print("overwriting...")
            shutil.rmtree(new_dataset_path)
            os.mkdir(new_dataset_path)
        else:
            exit(1)
    os.mkdir(os.path.join(new_dataset_path, "processed"))
    os.mkdir(os.path.join(new_dataset_path, "feature_selected"))

def preprocess(filepath, output_name, dataset_name):

    # open filepath with mne
    raw = mne.io.read_raw_eeglab(filepath, preload=True)
    # process raw_mne

    #Was test
    #mne.export.export_raw("analyze.edf", raw)

    #def need to remove power line noise at 50Hz

    print(raw.info)
    raw_powerline = raw.copy().notch_filter(freqs=50, notch_widths=1)

    raw_filtered = raw_powerline.copy().filter(l_freq=1, h_freq=249)
    
    #raw_powerline = raw.copy().notch_filter(freqs=50, notch_widths=1)

    # ^ remove band pass filter and or make a copy for doing ICA and then apply to non band pass 
    
    # ica to remove heartbeats 
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw_filtered)
    ecg_inds, ecg_scores = ica.find_bads_ecg(raw_powerline, ch_name='ECG1')  
    ica.apply(raw_powerline, exclude=ecg_inds)

    #TODO
    #base line correction 1/4 average / per epoch basis 
    # demeaning 
    # common average referencing 
    # average referencing
    #raw_powerline.set_eeg_reference(ref_channels="average")

    raw_powerline.set_eeg_reference(ref_channels="average")
    '''
    # Demean the data
    for i in range(len(raw_powerline.ch_names)):
        # Select the data for the current channel
        data, times = raw_powerline[i, :]

        # Subtract the mean of the channel's data from each time point
        demeaned_data = data - data.mean()

        # Replace the channel's data in the Raw object with the demeaned data
        raw_powerline._data[i, :] = demeaned_data
    '''
    # demeaning with moving average
    window_size = 50  # Arbitrary value -- might want it to be larger 

    if window_size % 2 == 0:
        window_size += 1
    n_channels = len(raw_powerline.ch_names)

    for i in range(n_channels):
        data, times = raw_powerline[i, :]
        data = data.flatten()  # Flatten the data array
        padding = window_size // 2
        padded_data = np.pad(data, (padding, padding), mode='edge')
        moving_avg = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
        demeaned_data = data - moving_avg
        raw_powerline._data[i, :] = demeaned_data  

    # output data
    data, times = raw[:, :]
    channel_names = raw.info['ch_names']
    df = pd.DataFrame(data.T, columns=channel_names)
    output_path = os.path.join(datasets_path, dataset_name,'processed', output_name+".csv")
    df.to_csv(output_path, index=False)

def main(): 
    args = parse_arguments()
    
    if args.dataset_name:
        name = args.dataset_name
    else:
        name = "untitled_dataset"

    #make directory structure of for dataset 
    make_dir_structure(name)

    #get raw file paths
    raw_file_paths = ut.find_files_with_ext(".set", raw_data_path)

    for x in raw_file_paths:
        print(x)
        preprocess(x, ut.create_short_identifier(x), name) 

main()
