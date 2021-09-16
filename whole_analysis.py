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

    'axes.labelsize': '20',

    'xtick.labelsize': '20',

    'ytick.labelsize': '20',

    'lines.linewidth': 1,

    'legend.fontsize': '20',

    'font.family': 'Times New Roman',

    'figure.figsize': '10, 20'  #图片尺寸

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
for i in range(len(user_list)):
    with open(source_path + user_list[i]) as f:
        json_obj = json.load(f)
        count = count + 1
        step_length = len(json_obj['rows'])
        fig, axs = plt.subplots(5, 1, sharex=True)

        
        axs[0].set_title(user_list[i])
        axs[0].set_ylabel('information') 
        axs[0].set_ylim((0, 1.2))
        axs[0].xaxis.set_major_locator(MultipleLocator(1))
        axs[1].set_ylabel('discrepancy') 
        axs[1].set_ylim((0, 22))
        axs[1].xaxis.set_major_locator(MultipleLocator(1))
        axs[2].set_ylabel('converage') 
        axs[2].set_ylim((0, 22))
        axs[2].xaxis.set_major_locator(MultipleLocator(1))
        axs[3].set_ylabel('confidence') 
        axs[3].set_ylim((-0.2, 1.2))
        axs[3].xaxis.set_major_locator(MultipleLocator(1))
        axs[4].set_xlabel('step')
        axs[4].set_ylabel('location')
        axs[4].set_xlim((0, step_length+1))
        axs[4].set_ylim((0, 22))
        axs[4].xaxis.set_major_locator(MultipleLocator(1))
        

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
        initial_belief_number = 100
        inter_count = 0
        for j in range(len(json_obj['rows'])):
            
            if(json_obj['rows'][j]['type'] == "normal"):
                state[0].append(json_obj['rows'][j]['index'])
                state[1].append(json_obj['rows'][j]['measurements'])
                inter_count += 1
            elif(json_obj['rows'][j]['type'] == "deviate"):
                state[0].append(json_obj['rows'][j]['index'])
                state[1].append(json_obj['rows'][j]['measurements'])
                inter_count += 1
                deviate_number = min(deviate_number, inter_count)
            elif(json_obj['rows'][j]['type'] == "discarded"):
                continue

            if((json_obj['rows'][j]['roc'] != 
            "I don't have enough information to make a confidence judgment" or
            json_obj['rows'][j]['roc'] != 
            "I don\u2019t have enough information to make a confidence judgment")):
                initial_belief_number = min(initial_belief_number, inter_count)

            # plot state
            axs[4].scatter((inter_count)*np.ones(len(state[0])), state[0], marker='s', color=sns.xkcd_rgb['leaf'],
                s=40)
            axs[4].scatter((inter_count), state[0][len(state[0])-1], marker='s', color=sns.xkcd_rgb['orangered'],
            s=40)

            x = np.linspace(1,inter_count,inter_count)
            # plot the trajectory of user information
            print(state)
            DM.env.set_state(state)
            DM.handle_information_accuracy()
            DM.handle_information_coverage()
            DM.information_model()
            print(inter_count)
            if(inter_count >= initial_belief_number and inter_count >=3 and initial_belief_number < 100):
                try:
                    print(count)
                    print(user_list[i])
                    DM.handle_feature_point_detection()
                    discrepancy.append(DM.rmse_data)
                    print('sdfasdafasdfasf', DM.fitting_error_matrix)
                    abs_discrepancy = np.absolute(DM.fitting_error_matrix)
                    axs[1].scatter((inter_count)*np.ones(22),np.linspace(1,22,22), c=abs_discrepancy,s=1000*abs_discrepancy, cmap='Wistia')
                except:
                    discrepancy.append(0)
                    axs[1].scatter((inter_count)*np.ones(22),np.linspace(1,22,22), c=np.zeros(22),s=40*np.zeros(22), cmap='Wistia')
            else:
                discrepancy.append(0)
                axs[1].scatter((inter_count)*np.ones(22),np.linspace(1,22,22), c=np.zeros(22),s=40*np.zeros(22), cmap='Wistia')
            axs[2].scatter((inter_count)*np.ones(22),np.linspace(1,22,22), c=DM.information_matrix,s=40*DM.information_matrix, cmap='Wistia')
            data_x.append(DM.coverage_criteria)
            data_y.append(DM.accuracy_criteria)
            confidence.append(beliefs_level[json_obj['rows'][j]['roc']])
        total_width, n = 0.4, 3
        width = total_width/n
        x_shifted = x -(total_width - width)/2
        print(x_shifted)
        print(data_x)
        axs[0].axvline(x=max(initial_belief_number,3) + 0.5, c="b", ls="--", lw=1)
        axs[1].axvline(x=max(initial_belief_number,3) + 0.5, c="b", ls="--", lw=1)
        axs[2].axvline(x=max(initial_belief_number,3) + 0.5, c="b", ls="--", lw=1)
        axs[3].axvline(x=max(initial_belief_number,3) + 0.5, c="b", ls="--", lw=1)
        axs[4].axvline(x=max(initial_belief_number,3) + 0.5, c="b", ls="--", lw=1)
        (axs[3].text(max(initial_belief_number, 3) +0.5, -0.25   ,
        '----exploration | verification----', fontdict={'size': 16, 'color': 'b'},
        horizontalalignment='center', verticalalignment='top'))
        if(deviate_number < 100):
            axs[0].axvline(x=deviate_number - 0.5, c="r", ls="--", lw=1)
            axs[1].axvline(x=deviate_number - 0.5, c="r", ls="--", lw=1)
            axs[2].axvline(x=deviate_number - 0.5, c="r", ls="--", lw=1)
            axs[3].axvline(x=deviate_number - 0.5, c="r", ls="--", lw=1)
            axs[4].axvline(x=deviate_number - 0.5, c="r", ls="--", lw=1)
            (axs[2].text(deviate_number -0.5, -1   ,
            '----initial | deviate----', fontdict={'size': 16, 'color': 'r'},
            horizontalalignment='center', verticalalignment='top'))
        axs[0].bar(x_shifted, data_x, width=width, label="coverage",color=["orange"])
        axs[0].bar(x_shifted+width, data_y, width=width, label="accuracy",color=["lime"])
        axs[0].bar(x_shifted+2*width, discrepancy, width=width, label="discrepancy", color=["dodgerblue"])
        axs[0].legend()
        axs[3].plot(x, confidence, marker="s", markersize="10", linewidth=3, label="user confindence", c="orangered")
        axs[3].legend()
        plt.savefig('user'+str(count))
            



        # # print('-------------------------------------------------------------')
        # # print(state)
        # DM.env.set_state(state)
        # DM.handle_information_accuracy()
        # DM.handle_information_coverage()
        # DM.information_model()
        # if(data_version == False and user_conclusion == 'Yes'):
        #     DM.plot('cool green', 'confidence.png')
        # if(data_version == True and user_conclusion == 'No'):
        #     DM.plot('cool green', 'confidence.png')
        # if(data_version == False and user_conclusion == "No"):
        #     DM.plot('dull red', 'confidence.png')
        # if(data_version == True and user_conclusion == "Yes"):
        #     DM.plot('dull red', 'confidence.png')



        # if(data_version == False and user_conclusion == 'Yes'):
        #     DM.plot_line('cool green', data_x, data_y, 'trajectory.png')
        # if(data_version == True and user_conclusion == 'No'):
        #     DM.plot_line('cool green', data_x, data_y, 'trajectory.png')
        # if(data_version == False and user_conclusion == "No"):
        #     DM.plot_line('dull red', data_x, data_y, 'trajectory.png')
        # if(data_version == True and user_conclusion == "Yes"):
        #     DM.plot_line('dull red', data_x, data_y, 'trajectory.png')



