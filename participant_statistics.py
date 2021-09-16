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
from re import T
from numpy import DataSource
import scipy.io as scio
from rule_based_decision_making import *
from matplotlib.pyplot import MultipleLocator
beliefs_level = {}
beliefs_level["I don\u2019t have enough information to make a confidence judgment"] = 0
beliefs_level["I don't have enough information to make a confidence judgment"] = 0
beliefs_level["Not at all confident the data supports the hypothesis"] = 0.2
beliefs_level["Slightly confident the data supports the hypothesis"] = 0.4
beliefs_level["Moderately confident the data supports the hypothesis"] = 0.6
beliefs_level["Very confident the data supports the hypothesis"] = 0.8
myparams = {

    'axes.labelsize': '40',

    'xtick.labelsize': '40',

    'ytick.labelsize': '40',

    'lines.linewidth': 1,

    'legend.fontsize': '40',

    'font.family': 'Times New Roman',

    'figure.figsize': '20, 12'  #图片尺寸

    }
pylab.rcParams.update(myparams)  #更新自己的设置
def plot_line(self, color, data_x, data_y, name):

    # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #线型设置 
    fig1 = plt.figure(1)
    a = plt.plot(data_x, data_y ,marker='o', color=sns.xkcd_rgb[color],
            markersize=5, linestyle='None')
    
    plt.legend(loc="lower right")  #图例位置 右下角
    plt.ylabel('accuracy') 
    plt.xlabel('coverage ') 
    plt.xlim((0, 22))
    plt.ylim((0, 22))
    # plt.axvline(x=1, c="b", ls="--", lw=1)
    # plt.axhline(y=1, c="b", ls="--", lw=1)
    plt.savefig(name)


source_path = 'usersdata/'
user_list = os.listdir(source_path)
DM = rule_state_machine()
json_list = []
user_name = []

        
count = 0
male_number = 0
female_number = 0
for i in range(len(user_list)):
    if_deviate = False
    with open(source_path + user_list[i]) as f:
        json_obj = json.load(f)
        count = count + 1
        step_length = len(json_obj['rows'])
        print(json_obj['form']['gender']['value'])
        if(json_obj['form']['gender']['value'] == "male"):
            male_number += 1
        else:
            female_number += 1
print(male_number)
print(female_number)
        