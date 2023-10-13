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
    else:
        os.mkdir(new_dataset_path)
    os.mkdir(os.path.join(new_dataset_path, "processed"))
    os.mkdir(os.path.join(new_dataset_path, "feature_selected"))

def preprocess(filepath, output_name, dataset_name, epoch_length=5):

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
    '''
    # output data
    data, times = raw[:, :]
    channel_names = raw.info['ch_names']
    df = pd.DataFrame(data.T, columns=channel_names)
    output_path = os.path.join(datasets_path, dataset_name,'processed', output_name+".csv")
    df.to_csv(output_path, index=False)
    '''
    # Get the sampling rate
    sfreq = raw.info['sfreq']
    # Calculate the number of points in each epoch
    points_per_epoch = int(epoch_length * sfreq)
    # Calculate the total number of epochs
    num_epochs = len(raw.times) // points_per_epoch

    # Get channel names
    channel_names = raw.ch_names

    for epoch_num in range(num_epochs):
        # Calculate start and end points for the epoch
        start = epoch_num * points_per_epoch
        end = (epoch_num + 1) * points_per_epoch

        # Extract epoch data
        data, times = raw[:, start:end]
        df = pd.DataFrame(data.T, columns=channel_names)

        # Save each epoch as a separate CSV file
        epoch_output_name = f"{output_name}_E{epoch_num+1}.csv"
        output_path = os.path.join(datasets_path, dataset_name, 'processed', epoch_output_name)
        df.to_csv(output_path, index=False)


#Fp1,Fz,F3,F7,FT9,FC5,FC1,C3,T7,ECG1,CP5,CP1,Pz,P3,P7,O1,Oz,O2,P4,P8,TP10,CP6,CP2,FCz,C4,T8,FT10,FC6,FC2,F4,F8,Fp2,AF7,AF3,AFz,F1,F5,FT7,FC3,C1,C5,TP7,CP3,P1,P5,PO7,PO3,POz,PO4,PO8,P6,P2,CPz,CP4,TP8,C6,C2,FC4,FT8,F6,AF8,AF4,F2


def compute_fft_on_csv(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    df_fft = pd.DataFrame()  

    for column in df.columns:
        if not np.issubdtype(df[column].dtype, np.number):
            continue

        waveform = df[column].values.astype(float)
        sampling_rate = 500  # Sampling rate in Hz
        num_samples = len(waveform)
        frequencies = np.fft.fftfreq(num_samples, 1/sampling_rate)
        fft_values = np.fft.fft(waveform)
        fft_magnitudes = np.abs(fft_values)
        positive_freq_idxs = np.where(frequencies > 0)
        df_fft[column] = fft_magnitudes[positive_freq_idxs]

    df_fft.to_csv(output_csv_path, index=False)

def main(): 
    args = parse_arguments()
    
    if args.dataset_name:
        name = args.dataset_name
    else:
        name = "dataset_1"
    print(f"Making dataset: {name}...")
    #make directory structure of for dataset 
    make_dir_structure(name)

    #get raw file paths
    raw_file_paths = ut.find_files_with_ext(".set", raw_data_path)
    
    print(f"Preprocessing...")
    for x in raw_file_paths:
        print(x)
        preprocess(x, ut.create_short_identifier(x), name) 
    fft_input = os.path.join(datasets_path, name,'processed')
    processed_waveforms = ut.find_files_with_ext(".csv", fft_input)

    print(f"Making FFTs...")
    for x in processed_waveforms:
        fft_output = os.path.join(datasets_path, name,'feature_selected', os.path.basename(x))
        compute_fft_on_csv(x, fft_output)

    print(f"annnndddd Done.")
main()
