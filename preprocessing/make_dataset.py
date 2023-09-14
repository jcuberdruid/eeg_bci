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
    print(raw.info)
    raw_filtered = raw.copy().filter(l_freq=1, h_freq=30)
    
    # ica to remove heartbeats 
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw)
    ecg_inds, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG1')  
    ica.apply(raw, exclude=ecg_inds)

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
