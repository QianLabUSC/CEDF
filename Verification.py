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

    'figure.figsize': '20, 30'  #图片尺寸

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

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def gauss(mean, scale, x=np.linspace(1,22,22), sigma=3):
    return scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))

source_path = 'usersdata/'
user_list = os.listdir(source_path)
DM = rule_state_machine()
json_list = []
user_name = []

        
count = 0
information_selection_number = {}
discrepancy_selection_number = {}
both_selection_number = {}
total_number = {}
feature_selection_number = {}
variable_selection_number = {}
information_selection_number[0] = 0
information_selection_number[0.2] = 0
information_selection_number[0.4] = 0
information_selection_number[0.6] = 0
information_selection_number[0.8] = 0
discrepancy_selection_number[0] = 0
discrepancy_selection_number[0.2] = 0
discrepancy_selection_number[0.4] = 0
discrepancy_selection_number[0.6] = 0
discrepancy_selection_number[0.8] = 0
both_selection_number[0] = 0
both_selection_number[0.2] = 0
both_selection_number[0.4] = 0
both_selection_number[0.6] = 0
both_selection_number[0.8] = 0
total_number[0] = 0
total_number[0.2] = 0
total_number[0.4] = 0
total_number[0.6] = 0
total_number[0.8] = 0
feature_selection_number[0] = 0
feature_selection_number[0.2] = 0
feature_selection_number[0.4] = 0
feature_selection_number[0.6] = 0
feature_selection_number[0.8] = 0

variable_selection_number[0] = 0
variable_selection_number[0.2] = 0
variable_selection_number[0.4] = 0
variable_selection_number[0.6] = 0
variable_selection_number[0.8] = 0
fig, axs = plt.subplots(3, 1, sharex=True)
for i in range(len(user_list)):
    if_deviate = False
    deviate_length = 0
    print(user_list[i])
    with open(source_path + user_list[i]) as f:
        json_obj = json.load(f)
        count = count + 1
        step_length = len(json_obj['rows'])
        # judge if it is deviated
        for j in range(len(json_obj['rows'])):
            if(json_obj['rows'][j]['type'] == "normal"):
                deviate_length += 1
            elif(json_obj['rows'][j]['type'] == "deviate"):
                if_deviate = True
                deviate_length += 1
                break
            elif(json_obj['rows'][j]['type'] == "discarded"):
                continue
        #print(deviate_length)
        data_version = json_obj['isAlternativeHypo']      # 0-the right 1-the wrong hypo
        user_conclusion = json_obj['concludeQuestions']['support']
        # drop out the non deviate user and step length than 5""
        if True:
            folder = "hypo11"
            
            state = [[],[]]
            status = []
            data_x = []
            data_y = []
            discrepancy = []
            confidence = []
            discrepancy_distribution = []
            converage_distribution = []
            deviate_number = 100
            inter_count = 0
            information_spatial_coverage_total = []
            information_variable_coverage_total = []
            fitting_error_total_set = []

            for j in range(len(json_obj['rows'])):
                if(json_obj['rows'][j]['type'] == "normal"):
                    state[0].append(json_obj['rows'][j]['index'])
                    state[1].append(json_obj['rows'][j]['measurements'])
                    continue
                elif(json_obj['rows'][j]['type'] == "deviate"):
                    state[0].append(json_obj['rows'][j]['index'])
                    state[1].append(json_obj['rows'][j]['measurements'])
                    inter_count += 1
                elif(json_obj['rows'][j]['type'] == "discarded"):
                    break
                DM.env.set_data_version(data_version)
                DM.env.set_state(state)
                DM.handle_information_accuracy()
                DM.handle_information_coverage()
                DM.information_model()
                #print(inter_count)
                if(len(state[0]) >=3):
                    try:
                        # print(count)
                        #print(user_list[i])
                        DM.handle_feature_point_detection()
                        fitting_error_total = DM.rmse_data
                        abs_discrepancy = normalization(np.absolute(DM.fitting_error_matrix))
                    except:
                        abs_discrepancy = np.zeros(22)
                        fitting_error_total = 0
                else:
                    abs_discrepancy = np.zeros(22)
                    fitting_error_total = 0
                discrepany_matrix = np.zeros(22)
                # for ii in range(len(abs_discrepancy)):
                #     interst_length = 2 
                #     value = abs_discrepancy[ii]/interst_length
                #     for jj in range(ii - interst_length, ii + interst_length + 1):
                #         if jj >= 0 and jj <= 21:
                #             discrepany_matrix[jj] += value
                # discrepany_matrix = normalization(discrepany_matrix)
                # print(discrepany_matrix)
                information_spatial_coverage_total.append(np.sum(DM.information_matrix)/22)
                information_variable_coverage_total.append(np.sum(DM.variable_information)/20)
                fitting_error_total_set.append(fitting_error_total)
            step = np.linspace(1,inter_count, inter_count)
            if((data_version==0 and user_conclusion == "Yes")):
                axs[0].plot(step, information_spatial_coverage_total, marker="o", markersize="20", linewidth=3, label="spatial coverage", c="lime")
                axs[1].plot(step, information_variable_coverage_total, marker="H", markersize="20", linewidth=3, label="variable coverage", c="lime" )
                axs[2].plot(step, fitting_error_total_set, marker="+", markersize="20", linewidth=3, label="variable coverage", c="lime" )
            if((data_version==0 and user_conclusion == "No")):
                axs[0].plot(step, information_spatial_coverage_total, marker="o", markersize="20", linewidth=3, label="spatial coverage", c="red")
                axs[1].plot(step, information_variable_coverage_total, marker="H", markersize="20", linewidth=3, label="variable coverage", c="red" )
            if((data_version==1 and user_conclusion == "Yes")):

                axs[0].plot(step, information_spatial_coverage_total, marker="o", markersize="20", linewidth=3, label="spatial coverage", c="blue")
                axs[1].plot(step, information_variable_coverage_total, marker="H", markersize="20", linewidth=3, label="variable coverage", c="blue" )
                axs[2].plot(step, fitting_error_total_set, marker="+", markersize="20", linewidth=3, label="variable coverage", c="blue" )
            if((data_version==1 and user_conclusion == "No")):
                axs[0].plot(step, information_spatial_coverage_total, marker="o", markersize="20", linewidth=3, label="spatial coverage", c="yellow")
                axs[1].plot(step, information_variable_coverage_total, marker="H", markersize="20", linewidth=3, label="variable coverage", c="yellow" )
                axs[2].plot(step, fitting_error_total_set, marker="+", markersize="20", linewidth=3, label="variable coverage", c="yellow" )
axs[0].set_xlabel('step')
axs[0].set_ylabel('information spatial coverage')
axs[0].set_title("information variation tendency", fontsize=40)
axs[1].set_xlabel('step')
axs[1].set_ylabel('information variable coverage')
axs[2].set_xlabel('step')
axs[2].set_ylabel('Discrepancy')
plt.savefig('verification factor tendency')
            
          
            
            


# xxx = np.linspace(0,0.8,5) 
# discrepancy_data = []
# information_data = []
# total_data = []
# both_selection_data =[]
# feature_selection_data = []
# variable_selection_data = []
# for i in [0,0.2,0.4,0.6,0.8]:
#     information_data.append(information_selection_number[i])
#     discrepancy_data.append(discrepancy_selection_number[i])
#     total_data.append(total_number[i])
#     both_selection_data.append(both_selection_number[i])
#     feature_selection_data.append(feature_selection_number[i])
#     variable_selection_data.append(variable_selection_number[i])
# total_width, n = 0.1, 6
# width = total_width/n
# x_shifted = xxx -(total_width - width)/2
# plt.figure(1001)
# plt.bar(x_shifted, information_data, width=width, label="coverage selection",color=["orange"])
# plt.bar(x_shifted+width, discrepancy_data, width=width, label="discrepancy selection",color=["lime"])
# plt.bar(x_shifted+2*width, total_data, width=width, label="total selection",color=["dodgerblue"])
# plt.bar(x_shifted+3*width, both_selection_data, width=width, label="both selection",color=["orangered"])
# plt.bar(x_shifted+4*width, feature_selection_data, width=width, label="feature selection",color=["yellow"])
# plt.bar(x_shifted+5*width, variable_selection_data, width=width, label="variable selection",color=["red"])
# plt.legend()
# plt.ylabel('number of participants')
# plt.xlabel('confidence')
# # x_label=['>0&<0.5','>0.5&<1','>1&<1.5','>1.5&<2','>2&<2.5']
# # plt.xticks(xxx, x_label)  # 绘制x刻度标签plt.title('selection driven mode')
# plt.yticks(range(0,24,1))
# plt.ylim((0,24))
# plt.savefig('./statictics/wrong.png')


