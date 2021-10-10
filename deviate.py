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

    'figure.figsize': '20, 15'  #图片尺寸

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

for i in range(len(user_list)):
    if_deviate = False
    deviate_length = 0
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
        if if_deviate and deviate_length >= 5 and ((data_version==1 and user_conclusion == "No")):
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
            for j in range(len(json_obj['rows'])):
                if(json_obj['rows'][j]['type'] == "normal"):
                    state[0].append(json_obj['rows'][j]['index'])
                    state[1].append(json_obj['rows'][j]['measurements'])
                    inter_count += 1
                elif(json_obj['rows'][j]['type'] == "deviate"):
                    # state[0].append(json_obj['rows'][j]['index'])
                    # state[1].append(json_obj['rows'][j]['measurements'])
                    # inter_count += 1
                    deviate_state_0 = json_obj['rows'][j]['index']
                    deviate_state_1 = json_obj['rows'][j]['measurements']
                    deviate_number = min(deviate_number, inter_count)
                    break
                elif(json_obj['rows'][j]['type'] == "discarded"):
                    continue
            DM.env.set_data_version(data_version)
            DM.env.set_state(state)
            DM.handle_information_accuracy()
            DM.handle_information_coverage()
            DM.information_model()
            #print(inter_count)
            if(inter_count >=3):
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
            for ii in range(len(abs_discrepancy)):
                discrepany_matrix += gauss(ii, abs_discrepancy[ii])
            discrepany_matrix = normalization(discrepany_matrix)


            information_converage = normalization(DM.information_matrix)
            variable_coverage = normalization(DM.mapping_value)
            confidence = beliefs_level[json_obj['rows'][j]['roc']]
            '''
            implement importance level detection here
            '''
            # select the interest area based on information coverage
            information_selection =  np.argwhere(information_converage <= 0.4)
            information_selection = information_selection.reshape(1, -1)[0] + 1
            # select the interest area based on the distributed discrepancy
            discrepancy_selection =  np.argwhere(discrepany_matrix > 0.5)
            discrepancy_selection = discrepancy_selection.reshape(1,-1)[0] + 1
            # select the interest area based on the feature area
            feature_selection = DM.saturation_selection
            # print('123123123', feature_selection)
            # select the interest area based on variable area.
            variable_loc_selection = np.argwhere(variable_coverage < 0.4)


            # # statistic the deviate_state_0 (whether in coverage selection and discrepancy selection)
            total_number[confidence] += 1
            if deviate_state_0 in information_selection:
                information_selection_number[confidence] += 1
            if deviate_state_0 in discrepancy_selection:
                discrepancy_selection_number[confidence] += 1
            if (deviate_state_0 in information_selection) and (deviate_state_0 in discrepancy_selection):
                both_selection_number[confidence] += 1
            if (deviate_state_0 in feature_selection):
                feature_selection_number[confidence] += 1
            if (deviate_state_0 in variable_loc_selection):
                variable_selection_number[confidence] += 1

            # # statistic the deviate_state_0 (whether in coverage selection and discrepancy selection)
            # if(fitting_error_total > 0 and fitting_error_total <= 0.5):
            #     confidence = 0
            # elif(fitting_error_total > 0.5 and fitting_error_total <= 1):
            #     confidence = 0.2
            # elif(fitting_error_total > 1 and fitting_error_total <= 1.5):
            #     confidence = 0.4
            # elif(fitting_error_total > 1.5  and fitting_error_total <= 2):
            #     confidence = 0.6
            # elif(fitting_error_total > 2):
            #     confidence = 0.8

            # total_number[confidence] += 1
            # if deviate_state_0 in information_selection:
            #     information_selection_number[confidence] += 1
            # if deviate_state_0 in discrepancy_selection:
            #     discrepancy_selection_number[confidence] += 1
            # if (deviate_state_0 in information_selection) and (deviate_state_0 in discrepancy_selection):
            #     both_selection_number[confidence] += 1
            # if (deviate_state_0 in feature_selection):
            #     feature_selection_number[confidence] += 1

            x = np.linspace(1,22,22)


            #plot Information spatial coverage 

            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(x, information_converage, marker="o", markersize="20", linewidth=3, label="coverage", c="lime")
           
            axs[0].plot(information_selection, -0.13 * np.ones(len(information_selection)),marker='H',markersize="20", c="blue")
            axs[0].set_xlabel('loc')
            axs[0].set_ylabel('value')
            axs[0].set_xticks(range(0,23,1))
            axs[0].plot(x, -0.7 * np.ones(22), marker='H',markersize="20", label="sample", c="grey")
            axs[0].plot( [j + 1 for j in state[0]], -0.7 * np.ones(len(state[0])), marker='H',markersize="20", label="sample", c="lime")
            axs[0].plot(deviate_state_0 + 1, -0.7, marker='H',markersize="20", label="sample", c="orangered")
            axs[0].set_title(user_list[i] + 'confidence' + str(confidence), fontsize=40)
            state[0].append(deviate_state_0)
            state[1].append(deviate_state_1)
            DM.env.set_state(state)
            mm, erodi = DM.env.get_data_state()
            #print(erodi)
            for kk in range(mm.shape[1]):
                erodi_nonzero = erodi[:,kk][np.nonzero(erodi[:,kk])]
                #print(erodi_nonzero)
                axs[1].scatter(kk*np.ones(len(erodi_nonzero)) + 1, erodi_nonzero, marker='D',s=160, label="sample", c="lime")
            erodi_deviate_nonzero = erodi[:,deviate_state_0][np.nonzero(erodi[:,deviate_state_0])]
            axs[1].scatter(deviate_state_0*np.ones(len(erodi_deviate_nonzero)) + 1, erodi_deviate_nonzero, marker='D',s=160, label="sample", c="orangered")
            axs[1].set_xlabel('loc')
            axs[1].set_ylabel('erodi')
            axs[1].xaxis.set_major_locator(MultipleLocator(1))
            axs[1].yaxis.set_major_locator(MultipleLocator(2))
            plt.savefig('./deviate_analysis/hypo11/user'+str(i+1)+"ISC")

            # plot Information variable coverage 
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(x, variable_coverage, marker="o", markersize="20", linewidth=3, label="coverage", c="lime")
           
            axs[0].plot(variable_loc_selection, -0.13 * np.ones(len(variable_loc_selection)),marker='H',markersize="20", c="blue")
            axs[0].set_xlabel('loc')
            axs[0].set_ylabel('value')
            axs[0].set_xticks(range(0,23,1))
            axs[0].plot(x, -0.7 * np.ones(22), marker='H',markersize="20", label="sample", c="grey")
            axs[0].plot( [j + 1 for j in state[0]], -0.7 * np.ones(len(state[0])), marker='H',markersize="20", label="sample", c="lime")
            axs[0].plot(deviate_state_0 + 1, -0.7, marker='H',markersize="20", label="sample", c="orangered")
            axs[0].set_title(user_list[i] + 'confidence' + str(confidence), fontsize=40)
            state[0].append(deviate_state_0)
            state[1].append(deviate_state_1)
            DM.env.set_state(state)
            mm, erodi = DM.env.get_data_state()
            #print(erodi)
            for kk in range(mm.shape[1]):
                erodi_nonzero = erodi[:,kk][np.nonzero(erodi[:,kk])]
                #print(erodi_nonzero)
                axs[1].scatter(kk*np.ones(len(erodi_nonzero)) + 1, erodi_nonzero, marker='D',s=160, label="sample", c="lime")
            erodi_deviate_nonzero = erodi[:,deviate_state_0][np.nonzero(erodi[:,deviate_state_0])]
            axs[1].scatter(deviate_state_0*np.ones(len(erodi_deviate_nonzero)) + 1, erodi_deviate_nonzero, marker='D',s=160, label="sample", c="orangered")
            axs[1].set_xlabel('loc')
            axs[1].set_ylabel('erodi')
            axs[1].xaxis.set_major_locator(MultipleLocator(1))
            axs[1].yaxis.set_major_locator(MultipleLocator(2))
            plt.savefig('./deviate_analysis/hypo11/user'+str(i+1)+"IVC")

             # plot Discrepancy 
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(x, abs_discrepancy, marker="s", markersize="20", linewidth=3, label="distribut_discrepancy", c="pink")
            axs[0].plot(x, discrepany_matrix, marker="s", markersize="20", linewidth=3, label="discrepancy", c="orangered")
            axs[0].plot(discrepancy_selection, -0.26 * np.ones(len(discrepancy_selection)),marker='H',markersize="20", c="black")
            axs[0].set_xlabel('loc')
            axs[0].set_ylabel('value')
            axs[0].set_xticks(range(0,23,1))
            axs[0].plot(x, -0.7 * np.ones(22), marker='H',markersize="20", label="sample", c="grey")
            axs[0].plot( [j + 1 for j in state[0]], -0.7 * np.ones(len(state[0])), marker='H',markersize="20", label="sample", c="lime")
            axs[0].plot(deviate_state_0 + 1, -0.7, marker='H',markersize="20", label="sample", c="orangered")
            axs[0].set_title(user_list[i] + 'confidence' + str(confidence), fontsize=40)
            state[0].append(deviate_state_0)
            state[1].append(deviate_state_1)
            DM.env.set_state(state)
            mm, erodi = DM.env.get_data_state()
            #print(erodi)
            for kk in range(mm.shape[1]):
                erodi_nonzero = erodi[:,kk][np.nonzero(erodi[:,kk])]
                #print(erodi_nonzero)
                axs[1].scatter(kk*np.ones(len(erodi_nonzero)) + 1, erodi_nonzero, marker='D',s=160, label="sample", c="lime")
            erodi_deviate_nonzero = erodi[:,deviate_state_0][np.nonzero(erodi[:,deviate_state_0])]
            axs[1].scatter(deviate_state_0*np.ones(len(erodi_deviate_nonzero)) + 1, erodi_deviate_nonzero, marker='D',s=160, label="sample", c="orangered")
            axs[1].set_xlabel('loc')
            axs[1].set_ylabel('erodi')
            axs[1].xaxis.set_major_locator(MultipleLocator(1))
            axs[1].yaxis.set_major_locator(MultipleLocator(2))
            plt.savefig('./deviate_analysis/hypo11/user'+str(i+1)+"DC")

            # plot Discrepancy 
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(feature_selection, -0.39 * np.ones(len(feature_selection)),marker='H',markersize="20", c="yellow")
            axs[0].set_xlabel('loc')
            axs[0].set_ylabel('value')
            axs[0].set_xticks(range(0,23,1))
            axs[0].plot(x, -0.7 * np.ones(22), marker='H',markersize="20", label="sample", c="grey")
            axs[0].plot( [j + 1 for j in state[0]], -0.7 * np.ones(len(state[0])), marker='H',markersize="20", label="sample", c="lime")
            axs[0].plot(deviate_state_0 + 1, -0.7, marker='H',markersize="20", label="sample", c="orangered")
            axs[0].set_title(user_list[i] + 'confidence' + str(confidence), fontsize=40)
            state[0].append(deviate_state_0)
            state[1].append(deviate_state_1)
            DM.env.set_state(state)
            mm, erodi = DM.env.get_data_state()
            #print(erodi)
            for kk in range(mm.shape[1]):
                erodi_nonzero = erodi[:,kk][np.nonzero(erodi[:,kk])]
                #print(erodi_nonzero)
                axs[1].scatter(kk*np.ones(len(erodi_nonzero)) + 1, erodi_nonzero, marker='D',s=160, label="sample", c="lime")
            erodi_deviate_nonzero = erodi[:,deviate_state_0][np.nonzero(erodi[:,deviate_state_0])]
            axs[1].scatter(deviate_state_0*np.ones(len(erodi_deviate_nonzero)) + 1, erodi_deviate_nonzero, marker='D',s=160, label="sample", c="orangered")
            axs[1].set_xlabel('loc')
            axs[1].set_ylabel('erodi')
            axs[1].xaxis.set_major_locator(MultipleLocator(1))
            axs[1].yaxis.set_major_locator(MultipleLocator(2))
            plt.savefig('./deviate_analysis/hypo11/user'+str(i+1)+"Feature")
          
            
            


xxx = np.linspace(0,0.8,5) 
discrepancy_data = []
information_data = []
total_data = []
both_selection_data =[]
feature_selection_data = []
variable_selection_data = []
for i in [0,0.2,0.4,0.6,0.8]:
    information_data.append(information_selection_number[i])
    discrepancy_data.append(discrepancy_selection_number[i])
    total_data.append(total_number[i])
    both_selection_data.append(both_selection_number[i])
    feature_selection_data.append(feature_selection_number[i])
    variable_selection_data.append(variable_selection_number[i])
total_width, n = 0.1, 6
width = total_width/n
x_shifted = xxx -(total_width - width)/2
plt.figure(1001)
plt.bar(x_shifted, information_data, width=width, label="coverage selection",color=["orange"])
plt.bar(x_shifted+width, discrepancy_data, width=width, label="discrepancy selection",color=["lime"])
plt.bar(x_shifted+2*width, total_data, width=width, label="total selection",color=["dodgerblue"])
plt.bar(x_shifted+3*width, both_selection_data, width=width, label="both selection",color=["orangered"])
plt.bar(x_shifted+4*width, feature_selection_data, width=width, label="feature selection",color=["yellow"])
plt.bar(x_shifted+5*width, variable_selection_data, width=width, label="variable selection",color=["red"])
plt.legend()
plt.ylabel('number of participants')
plt.xlabel('confidence')
# x_label=['>0&<0.5','>0.5&<1','>1&<1.5','>1.5&<2','>2&<2.5']
# plt.xticks(xxx, x_label)  # 绘制x刻度标签plt.title('selection driven mode')
plt.yticks(range(0,24,1))
plt.ylim((0,24))
plt.savefig('./statictics/wrong.png')


