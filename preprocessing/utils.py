'''
The point of this script: a place for small utility functions to keep things uncluttered 

'''
################################################################
## get all files with x file extenion in y directory: 
################################################################
import os
def find_files_with_ext(extension, directory):
    if not os.path.isdir(directory):
        print("The provided path is not a valid directory. See utils.py")
        return 
    result = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                full_path = os.path.join(root, file)
                result.append(full_path)
    return result
################################################################
## given a path to raw .set file create a shorted unique name
################################################################
import re
import os
def shorten_segment(segment):
    words = re.findall(r'\b\w+\b', segment)
    short = ''.join(word[0].upper() for word in words)
    digits = re.findall(r'\d+$', segment)
    if digits:
        short += digits[0]
    return short

def create_short_identifier(file_path):
    segments = os.path.normpath(file_path).split(os.sep)
    sub_segment = segments[3]
    ses_segment = segments[4]
    filename_with_ext = segments[-1]

    # Extract filename without extension
    filename, _ = os.path.splitext(filename_with_ext)

    return f"{shorten_segment(sub_segment)}_{shorten_segment(ses_segment)}_{filename}"
''' 
old
def shorten_segment(segment):
    words = re.findall(r'\b\w+\b', segment)
    short = ''.join(word[0].upper() for word in words)
    digits = re.findall(r'\d+$', segment)
    if digits:
        short += digits[0]
    return short
def create_short_identifier(file_path): 
    segments = os.path.normpath(file_path).split(os.sep)
    sub_segment = segments[3]
    ses_segment = segments[4]
    filename = segments[-1]
    return f"{shorten_segment(sub_segment)}_{shorten_segment(ses_segment)}_{shorten_segment(filename)}"
'''

################################################################
# Prints structure of array/numpy/pandas 
################################################################
import pandas as pd
import numpy as np

def recursive_print_structure(obj, level=0):
    indent = ' ' * 4 * level

    if isinstance(obj, (int, float, str)):
        print(f"{indent}Type: {type(obj)}, Value: {obj}")

    elif isinstance(obj, pd.DataFrame):
        print(f"{indent}Type: DataFrame, Shape: {obj.shape}")
        recursive_print_structure(obj.iloc[0].tolist(), level+1)

    elif isinstance(obj, pd.Series):
        print(f"{indent}Type: Series, Length: {len(obj)}")
        recursive_print_structure(obj.iloc[0], level+1)

    elif isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            shape = f"{len(obj)}"
        else:
            sub_shapes = [recursive_get_shape(sub_arr) for sub_arr in obj]
            ranges = [f"{min(dim_values)}-{max(dim_values)}" if len(set(dim_values)) > 1 else f"{dim_values[0]}" for dim_values in zip(*sub_shapes)]
            shape = ",".join([str(len(obj))] + ranges)
        print(f"{indent}Type: ndarray, Shape: ({shape})")

    elif isinstance(obj, tuple):
        print(f"{indent}Type: tuple, Length: {len(obj)}")
        for item in obj:
            recursive_print_structure(item, level+1)

    elif isinstance(obj, list):
        print(f"{indent}Type: list, Length: {len(obj)}")
        if obj:
            recursive_print_structure(obj[0], level+1)

    else:
        print(f"{indent}Type: {type(obj)}, Value: {obj}")


def recursive_get_shape(obj):
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return [len(obj)]
        else:
            shapes = [recursive_get_shape(sub_arr) for sub_arr in obj]
            combined = list(zip(*shapes))
            return [len(obj)] + [max(dim) for dim in combined]

    elif isinstance(obj, list):
        return [len(obj)] + recursive_get_shape(obj[0]) if obj else [0]

    elif isinstance(obj, tuple):
        shapes = [recursive_get_shape(item) for item in obj]
        return [len(obj)] + [max(shape) for shape in zip(*shapes)]

    else:
        return []

