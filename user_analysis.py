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
from rule_based_decision_making import *

source_path = 'usersdata/'
user_list = os.listdir(source_path)
DM = rule_state_machine()
for i in user_list:
    with open(source_path + i) as f:
        json_obj = json.load(f)
        data_version = json_obj['isAlternativeHypo']      # 0-the right 1-the wrong hypo
        state = [[],[]]
        for i in range(len(json_obj['rows'])):
            if(json_obj['rows'][i]['roc'] == "I don\u2019t have enough information to make a confidence judgment"):
                state[0].append(json_obj['rows'][i]['index'])
                state[1].append(json_obj['rows'][i]['measurements'])
            else:
                state[0].append(json_obj['rows'][i]['index'])
                state[1].append(json_obj['rows'][i]['measurements'])
                break
        print(state)
        DM.env.set_state(state)
        DM.handle_information_accuracy()
        DM.handle_information_coverage()
        DM.information_model()