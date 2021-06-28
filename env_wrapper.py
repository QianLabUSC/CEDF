# This FILE is part of multi-legged robot field exploration model
# env_wrapper.py - to obtain user interaction data from website
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
import numpy as np

class ENV:
    def __init__(self):
        # load erodibility data from dataset      
        erodibility_data = scio.loadmat("erodibility_dataset.mat")
        tech_names = {'mm', 'y_H0', 'y_H1'}
        self.raw_data = ({key: value for key, value in erodibility_data.items()
                          if key in tech_names})
        # action/state includes [location index, samples in each index] 
        # here is the initial templates
        self.action = [[1,5,9,13,17,21], [3,3,3,3,3,3]]    
        self.state = [[1,5,9,13,17,21], [3,3,3,3,3,3]]

    def initiate_template(self, action):
        self.action = action
        self.state = action


    def get_data_state(self):
        # get the index of rows and cols
        row_index = []
        col_index = []
        for i in range(len(self.action[0])):
            for j in range(self.action[1][i]):
                row_random = np.random.randint(1,22)
                col_index.append(self.action[0][i])
                row_index.append(row_random)
        mm = - np.ones((30,22))
        mm[row_index, col_index] = self.raw_data['mm'][row_index, col_index]
        y_H0 = - np.ones((30,22))
        y_H0[row_index, col_index] = self.raw_data['mm'][row_index, col_index]
        y_H1 = - np.ones((30,22))
        y_H1[row_index, col_index] = self.raw_data['mm'][row_index, col_index]

class user_data:
    def __init__():

        
if __name__ == "__main__":
    env = ENV()
    env.get_data_state()
