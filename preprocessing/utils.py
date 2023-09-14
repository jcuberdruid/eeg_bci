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
    filename = segments[-1]
    return f"{shorten_segment(sub_segment)}_{shorten_segment(ses_segment)}_{shorten_segment(filename)}"
################################################################
################################################################
