import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def load_eeg_csv(csv_file):
    csv_file = classesPath + csv_file
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]
    
        # Group rows based on specified keys while preserving order
        key_columns = ['subject', 'epoch', 'run']
        row_groups = group_rows_preserve_order(rows, key_columns)
        # Process each chunk
        for chunk in row_groups:
            processChunk(chunk)

#TODO run fft on a single waveform 
run_fft(waveform):
    #convert the lines to floats and create a NumPy array
    waveform = np.array([float(line.strip()) for line in lines])

    #run the FFT on the waveform data
    sampling_rate = 500  # Sampling rate in Hz #XXX change for this dataset 
    num_samples = len(waveform)
    frequencies = np.fft.fftfreq(num_samples, 1/sampling_rate)
    fft_values = np.fft.fft(waveform)

    #keep positive frequencies
    positive_freq_idxs = np.where(frequencies > 0)
    frequencies = frequencies[positive_freq_idxs]
    fft_values = np.abs(fft_values[positive_freq_idxs])

    return fft_values

#Function to group rows based on specified keys while preserving order
def group_rows_preserve_order(rows, keys):
    groups = []
    current_group = []
    prev_key_values = None

    for row in rows:
        key_values = [row[key] for key in keys]

        if key_values != prev_key_values and current_group:
            groups.append(current_group)
            current_group = []

        current_group.append(row)
        prev_key_values = key_values

    if current_group:
        groups.append(current_group)

    return groups


array = np.load(projectionPath)


def chunkEach(csv_file):
    csv_file = classesPath + csv_file
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]

        # Group rows based on specified keys while preserving order
        key_columns = ['subject', 'epoch', 'run']
        row_groups = group_rows_preserve_order(rows, key_columns)
        #print(len(row_groups))
        #print(len(row_groups[0])) # 4 seconds
        # Process each chunk
        for chunk in row_groups:
            run_fft(chunk)


def get_csv_files(directory):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            csv_files.append(filename)
    return csv_files
                                                                                                          106,1         86%
