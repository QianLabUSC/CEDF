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
import scipy.io as scio


# load erodibility data from dataset      
# erodibility_data = scio.loadmat("erodibility_dataset.mat")
# tech_names = {'mm', 'y_H0', 'y_H1'}
# raw_data = {key: value for key, value in erodibility_data.items() if key in tech_names}
# raw_data['mm'] = raw_data['mm'].tolist()
# raw_data['y_H0'] = raw_data['y_H0'].tolist()
# raw_data['y_H1'] = raw_data['y_H1'].tolist()
# print(type(raw_data))
# with open('raw_data.json', 'w') as f:
#     json.dump(raw_data, f, indent=4)
# Load all user data to training data set 
# def save_data(user_id, user_data):
#     '''
#     save init template
#     '''
#     data_version = user_data['isAlternativeHypo']      # 0-the right 1-the wrong hypo
#     for i in range(len(user_data['rows'])):
#         step =  i
#         if(i == 0):
#             state =  
#             action = 

        


def main(source_path):
    user_list = os.listdir(source_path)
    jsondata_list = []
    for i in user_list:
        with open(source_path + i) as f:
            json_obj = json.load(f)
        with open(source_path + i, "w") as f:
            json.dump(json_obj, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-s', dest='source_path', help="please specify your source scenario path with --src sourcepath")
    args = parser.parse_args()
    main(args.source_path)
