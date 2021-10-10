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
from matplotlib.pyplot import MultipleLocator, axis
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

    'legend.fontsize': '10',

    'font.family': 'Times New Roman',

    'figure.figsize': '20, 40'  #图片尺寸

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
information_level_set = {}
discrepancy_level_set = {}
fitting_error_set = {}
colormap = {}
colormap[0] = 'blue'
colormap[0.2] = 'purple'
colormap[0.4] = 'green'
colormap[0.6] = 'red'
colormap[0.8] = 'black'
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
information_level_set[0] = [[],[],[],[]]
information_level_set[0.2] = [[],[],[],[]]
information_level_set[0.4] = [[],[],[],[]]
information_level_set[0.6] = [[],[],[],[]]
information_level_set[0.8] = [[],[],[],[]]
discrepancy_level_set[0] = [[],[],[],[]]
discrepancy_level_set[0.2] = [[],[],[],[]]
discrepancy_level_set[0.4] = [[],[],[],[]]
discrepancy_level_set[0.6] = [[],[],[],[]]
discrepancy_level_set[0.8] = [[],[],[],[]]
fitting_error_set[0] = [[],[],[],[]]
fitting_error_set[0.2] = [[],[],[],[]]
fitting_error_set[0.4] = [[],[],[],[]]
fitting_error_set[0.6] = [[],[],[],[]]
fitting_error_set[0.8] = [[],[],[],[]]
variable_coverage_set = [[],[],[],[]]
fitting_error_set_single = [[],[],[],[]]
fig, axs = plt.subplots(4, 1, sharex=True)
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
        # drop out the non deviate user and step length than 5
        if True:
            # fig, axs = plt.subplots(2, 1, sharex=True)

            data_version = json_obj['isAlternativeHypo']      # 0-the right 1-the wrong hypo
            user_conclusion = json_obj['concludeQuestions']['support']
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
            variable_coverage_set = []
            fitting_error_set_single = []
            for j in range(len(json_obj['rows'])):
                if(json_obj['rows'][j]['type'] == "normal"):
                    state[0].append(json_obj['rows'][j]['index'])
                    state[1].append(json_obj['rows'][j]['measurements'])
                    inter_count += 1
                elif(json_obj['rows'][j]['type'] == "deviate"):
                    state[0].append(json_obj['rows'][j]['index'])
                    state[1].append(json_obj['rows'][j]['measurements'])
                    inter_count += 1
                    # deviate_state_0 = json_obj['rows'][j]['index']
                    # deviate_state_1 = json_obj['rows'][j]['measurements']
                    # deviate_number = min(deviate_number, inter_count)
                    # break
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
                        #print(count)
                        #print(user_list[i])
                        DM.handle_feature_point_detection()
                        
                        # if(j == len(json_obj['rows']) - 1):
                        # if(data_version == 0 and user_conclusion == "Yes"):
                        #     fitting_error_set[confidence][0].append(DM.rmse_data)
                        # elif(data_version == 0 and user_conclusion == "No"):
                        #     fitting_error_set[confidence][1].append(DM.rmse_data)
                        # elif(data_version == 1 and user_conclusion == "Yes"):
                        #     fitting_error_set[confidence][2].append(DM.rmse_data)
                        # elif(data_version == 1 and user_conclusion == "No"):
                        #     fitting_error_set[confidence][3].append(DM.rmse_data)

                        fitting_error_set_single.append(DM.rmse_data)
                        variable_coverage_set.append(DM.variable_coverage)
                        
                        abs_discrepancy = normalization(np.absolute(DM.fitting_error_matrix))
                    except:
                        abs_discrepancy = np.zeros(22)
                else:
                    abs_discrepancy = np.zeros(22)
                
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
                # feature_selection = DM.saturation_selection
                # print('123123123', feature_selection)

                # evaluate the information level and discrepancy level
                information_level = 1 - len(information_selection)/22
                discrepancy_level = 1 - len(discrepancy_selection)/22
                if(j == len(json_obj['rows']) - 1):
                    if(data_version == 0 and user_conclusion == "Yes"):
                        information_level_set[confidence][0].append(information_level)
                        discrepancy_level_set[confidence][0].append(discrepancy_level)
                    if(data_version == 0 and user_conclusion == "No"):
                        information_level_set[confidence][1].append(information_level)
                        discrepancy_level_set[confidence][1].append(discrepancy_level)
                    if(data_version == 1 and user_conclusion == "Yes"):
                        information_level_set[confidence][2].append(information_level)
                        discrepancy_level_set[confidence][2].append(discrepancy_level)
                    if(data_version == 1 and user_conclusion == "No"):
                        information_level_set[confidence][3].append(information_level)
                        discrepancy_level_set[confidence][3].append(discrepancy_level)
            # print('lllllllllllllllllllllllll', fitting_error_set_single)
            # print('lllllllllllllllllllllllll', variable_coverage_set)
            if(data_version == 0 and user_conclusion == "Yes"):
                axs[0].plot( variable_coverage_set,fitting_error_set_single, 'r')
            if(data_version == 0 and user_conclusion == "No"):
                axs[1].plot( variable_coverage_set,fitting_error_set_single, 'g')
            if(data_version == 1 and user_conclusion == "Yes"):
                axs[2].plot( variable_coverage_set,fitting_error_set_single, 'b')
            if(data_version == 1 and user_conclusion == "No"):
                axs[3].plot( variable_coverage_set,fitting_error_set_single, 'y')

plt.xlabel('variable space coverage')
plt.ylabel('fitting error')
plt.savefig('./confidence/fittingerror vs variable space each coverage')
            # # statistic the deviate_state_0 (whether in coverage selection and discrepancy selection)
            # total_number[confidence] += 1
            # if deviate_state_0 in information_selection:
            #     information_selection_number[confidence] += 1
            # if deviate_state_0 in discrepancy_selection:
            #     discrepancy_selection_number[confidence] += 1
            # if (deviate_state_0 in information_selection) and (deviate_state_0 in discrepancy_selection):
            #     both_selection_number[confidence] += 1
            # if (deviate_state_0 in feature_selection):
            #     feature_selection_number[confidence] += 1
            

            # x = np.linspace(1,22,22)
            # axs[0].plot(x, abs_discrepancy, marker="s", markersize="20", linewidth=3, label="distribut_discrepancy", c="pink")
            # axs[0].plot(x, discrepany_matrix, marker="s", markersize="20", linewidth=3, label="discrepancy", c="orangered")
            # axs[0].plot(x, information_converage, marker="o", markersize="20", linewidth=3, label="coverage", c="lime")
            # axs[0].plot(information_selection, -0.13 * np.ones(len(information_selection)),marker='H',markersize="20", c="blue")
            # axs[0].plot(discrepancy_selection, -0.26 * np.ones(len(discrepancy_selection)),marker='H',markersize="20", c="black")
            # axs[0].plot(feature_selection, -0.39 * np.ones(len(feature_selection)),marker='H',markersize="20", c="yellow")
            # # axs[0].legend()
            # axs[0].set_xlabel('loc')
            # axs[0].set_ylabel('value')
            # axs[0].set_xticks(range(0,23,1))
            # axs[0].plot(x, -0.5 * np.ones(22), marker='H',markersize="20", label="sample", c="grey")
            # axs[0].plot( [j + 1 for j in state[0]], -0.5 * np.ones(len(state[0])), marker='H',markersize="20", label="sample", c="lime")
            # axs[0].plot(deviate_state_0 + 1, -0.5, marker='H',markersize="20", label="sample", c="orangered")
            # axs[0].set_title(user_list[i] + 'confidence' + str(confidence), fontsize=40)
            # state[0].append(deviate_state_0)
            # state[1].append(deviate_state_1)
            # DM.env.set_state(state)
            # mm, erodi = DM.env.get_data_state()
            # print(erodi)
            # for kk in range(mm.shape[1]):
            #     erodi_nonzero = erodi[:,kk][np.nonzero(erodi[:,kk])]
            #     print(erodi_nonzero)
            #     axs[1].scatter(kk*np.ones(len(erodi_nonzero)) + 1, erodi_nonzero, marker='D',s=160, label="sample", c="lime")
            # erodi_deviate_nonzero = erodi[:,deviate_state_0][np.nonzero(erodi[:,deviate_state_0])]
            # axs[1].scatter(deviate_state_0*np.ones(len(erodi_deviate_nonzero)) + 1, erodi_deviate_nonzero, marker='D',s=160, label="sample", c="orangered")
            # axs[1].set_xlabel('loc')
            # axs[1].set_ylabel('erodi')
            # axs[1].xaxis.set_major_locator(MultipleLocator(1))
            # axs[1].yaxis.set_major_locator(MultipleLocator(2))
            # # plt.savefig('user'+str(i+1))
# xxx = np.linspace(0,0.8,5) 
# discrepancy_data = []
# information_data = []
# total_data = []
# both_selection_data =[]
# feature_selection_data = []
# for i in [0,0.2,0.4,0.6,0.8]:
#     information_data.append(information_selection_number[i])
#     discrepancy_data.append(discrepancy_selection_number[i])
#     total_data.append(total_number[i])
#     both_selection_data.append(both_selection_number[i])
#     feature_selection_data.append(feature_selection_number[i])

# total_width, n = 0.1, 5
# width = total_width/n
# x_shifted = xxx -(total_width - width)/2
# plt.figure(1001)
# plt.bar(x_shifted, information_data, width=width, label="coverage selection",color=["orange"])
# plt.bar(x_shifted+width, discrepancy_data, width=width, label="discrepancy selection",color=["lime"])
# plt.bar(x_shifted+2*width, total_data, width=width, label="total selection",color=["dodgerblue"])
# plt.bar(x_shifted+3*width, both_selection_data, width=width, label="both selection",color=["orangered"])
# plt.bar(x_shifted+4*width, feature_selection_data, width=width, label="feature selection",color=["yellow"])
# plt.legend()
# plt.ylabel('number of participants')
# plt.xlabel('confidence')
# plt.title('selection driven mode')
# plt.yticks(range(0,18,1))
# plt.ylim((0,18))
# plt.savefig('statictics')

# plt.figure(1000+10* i, figsize=(20, 10))
# i=0
# plt.scatter(information_level_set[i][0], discrepancy_level_set[i][0], s=200, c=colormap[0], marker='H', label='hypo00')
# plt.scatter(information_level_set[i][1], discrepancy_level_set[i][1], s=200, c=colormap[0.2], marker='H', label='hypo01')
# plt.scatter(information_level_set[i][2], discrepancy_level_set[i][2], s=200, c=colormap[0.4], marker='H', label='hypo10')
# plt.scatter(information_level_set[i][3], discrepancy_level_set[i][3], s=200, c=colormap[0.6], marker='H', label='hypo11')
# # fig, axs = plt.subplots(5, 1, sharex=True)
# for i in [0.2,0.4,0.6,0.8]:

#     plt.scatter(information_level_set[i][0], discrepancy_level_set[i][0], s=200, c=colormap[0], marker='H')
#     plt.scatter(information_level_set[i][1], discrepancy_level_set[i][1], s=200, c=colormap[0.2], marker='H')
#     plt.scatter(information_level_set[i][2], discrepancy_level_set[i][2], s=200, c=colormap[0.4], marker='H')
#     plt.scatter(information_level_set[i][3], discrepancy_level_set[i][3], s=200, c=colormap[0.6], marker='H')
#     plt.legend(fontsize=40)
#     plt.ylabel('discrepancy')
#     plt.xlabel('information')
#     plt.title('confidence driven mode',fontsize=40)
#     plt.ylim((0,1))
#     plt.xlim((0,1))
    
# plt.savefig('./confidence/two-degree')

# plt.figure(1999, figsize=(20, 10))
# i = 0
# plt.scatter((i-0.05)*np.ones(len(fitting_error_set[i][0])), fitting_error_set[i][0], s=200, c=colormap[0], marker='H', label='dataversion0, yes')
# plt.scatter((i-0.025)*np.ones(len(fitting_error_set[i][1])), fitting_error_set[i][1], s=200, c=colormap[0.2], marker='H', label='dataversion0, no')
# plt.scatter((i)*np.ones(len(fitting_error_set[i][2])), fitting_error_set[i][2], s=200, c=colormap[0.4], marker='H', label='dataversion1, yes')
# plt.scatter((i+0.025)*np.ones(len(fitting_error_set[i][3])), fitting_error_set[i][3], s=200, c=colormap[0.6], marker='H', label='dataversion1, no')
# for i in [0.2,0.4,0.6,0.8]:
#     plt.scatter((i-0.05)*np.ones(len(fitting_error_set[i][0])), fitting_error_set[i][0], s=200, c=colormap[0], marker='H')
#     plt.scatter(i-0.025*np.ones(len(fitting_error_set[i][1])), fitting_error_set[i][1], s=200, c=colormap[0.2], marker='H')
#     plt.scatter(i*np.ones(len(fitting_error_set[i][2])), fitting_error_set[i][2], s=200, c=colormap[0.4], marker='H')
#     plt.scatter((i+0.025)*np.ones(len(fitting_error_set[i][3])), fitting_error_set[i][3], s=200, c=colormap[0.6], marker='H')
#     plt.legend(fontsize=40)
#     plt.ylabel('fitting error')
#     plt.xlabel('confidence')
#     plt.title('fitting error vs confidence',fontsize=40)
#     plt.xlim((-0.15,1))
# plt.savefig('./confidence/fittingerror vs confidece')



