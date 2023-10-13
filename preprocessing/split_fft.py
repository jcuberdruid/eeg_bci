## this script will split the results of the FFT into Delta, Alpha, Beta and return them 
import os
import utils as ut
import numpy as np
import pandas as pd
#human vars for arguments
sample_rate = 500 #Hz
FFT_Path = "../data/datasets/dataset_1/feature_selected/"

#constants
delta = (.5, 4)
alpha = (8, 12) 
beta= (13, 30) 

def delta_alpha_beta(fft):
    fft = fft.to_list()
    sample_length = int(len(fft)/250)
    #XXX assuming all inputs are for 60 second increments 
    delta_split = fft[int(delta[0]*sample_length):int(delta[1])*sample_length]   
    alpha_split = fft[int(alpha[0]*sample_length):int(alpha[1])*sample_length]   
    beta_split = fft[int(beta[0]*sample_length):int(beta[1])*sample_length]   
    return delta_split, alpha_split, beta_split 

def process_file(file):
    df = pd.read_csv(file)
    df_transformed = df.apply(delta_alpha_beta, axis=0)
    data_tuples = {col: (df_transformed[col].iloc[0],
                         df_transformed[col].iloc[1],
                         df_transformed[col].iloc[2]) for col in df_transformed.columns}
    df_result = pd.DataFrame([data_tuples])
    #ut.recursive_print_structure(df_result)
    return df_result

def labels_from_paths(path_list):
    for x in path_list:
        print(os.path.basename(x))
    print("unfinished") 

def get_bands_of_interest(pattern1, pattern2, saveFile):
    print("#################################")
    print(f"Starting {saveFile}")
    files = ut.find_files_with_ext('.csv', FFT_Path)
    
    files = [s for s in files if pattern1 in s and pattern2 in s]

    number_of_files = len(files)
    test = (list(map(process_file, files)))
    ut.recursive_print_structure(test)
    n = max(len(fft_band) for df in test for fft_band in df.iloc[0, 0])
    result_array = np.zeros((number_of_files, 63, 3, n))
    for df_idx, df in enumerate(test):
        for col_idx, col in enumerate(df.columns):
            for band_idx, fft_band in enumerate(df[col].iloc[0]):
                result_array[df_idx, col_idx, band_idx, :len(fft_band)] = fft_band
    print(result_array.shape)
    np.save(saveFile, result_array) 
    print(f"Finished {saveFile}")
    print("#################################")


get_bands_of_interest("S001", "zeroBACK", "subject_1_zeroback")
get_bands_of_interest("S001", "oneBACK", "subject_1_oneback")
get_bands_of_interest("S001", "twoBACK", "subject_1_twoback")


get_bands_of_interest("S002", "zeroBACK", "subject_2_zeroback")
get_bands_of_interest("S002", "oneBACK", "subject_2_oneback")
get_bands_of_interest("S002", "twoBACK", "subject_2_twoback")


#files = ut.find_files_with_ext('.csv', FFT_Path)

#labels_from_paths(files)

'''
##Map of names to shorted names
RS_Beg_EC.set -> S001_SS1_RS
RS_Beg_EO.set -> S001_SS1_RS
twoBACK.set -> S001_SS1_TS      <-
RS_End_EO.set -> S001_SS1_RS
oneBACK.set -> S001_SS1_OS      <-
MATBmed.set -> S001_SS1_MS
PVT.set -> S001_SS1_PS
MATBeasy.set -> S001_SS1_MS
zeroBACK.set -> S001_SS1_ZS     <-
MATBdiff.set -> S001_SS1_MS
Flanker.set -> S001_SS1_FS
RS_End_Ec.set -> S001_SS1_RS
'''

