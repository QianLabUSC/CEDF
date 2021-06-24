# This FILE is part of multi-legged robot field exploration model
# dataset_making.py to make data collection process dataset from human interaction
#
# This programm is explained by roboLAND in university of southern california.
# Please notify the source if you use it
# 
# Copyright(c) 2021-2025 Ryoma Liu
# Email: 1196075299@qq.com

import json
import os
import argparse
from argparse import RawTextHelpFormatter

###########################################
# Load all user data to training data set #
###########################################

def save_data(user_data):
    data_version = user_data['isAlternativeHypo']      # 0-the right 1-the wrong hypo
    '''
    save init template
    '''
    


def main(source_path):
    user_list = os.listdir(source_path)
    jsondata_list = []
    for i in user_list:
        with open(source_path + i) as f:
            json_obj = json.load(f)
            jsondata_list.append(json_obj)
    for i in jsondata_list:
        save_data(i)  
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-s', dest='source_path', help="please specify your source scenario path with --src sourcepath")
    args = parser.parse_args()
    main(args.source_path)
