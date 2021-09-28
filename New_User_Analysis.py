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
from matplotlib import markers
from numpy import DataSource
from numpy.core.fromnumeric import size
import scipy.io as scio
from rule_based_decision_making import *
from matplotlib.pyplot import MultipleLocator
import types
myparams = {

    'axes.labelsize': '40',

    'xtick.labelsize': '40',

    'ytick.labelsize': '40',

    'lines.linewidth': 1,

    'legend.fontsize': '20',

    'font.family': 'Times New Roman',

    'figure.figsize': '40, 100'  #图片尺寸

    }
pylab.rcParams.update(myparams)  #更新自己的设置
colormap = {}
colormap[0] = 'blue'
colormap[0.2] = 'purple'
colormap[0.4] = 'green'
colormap[0.6] = 'red'
colormap[0.8] = 'black'
source_path = 'newuserdata/'
user_list = os.listdir(source_path)
for i in range(len(user_list)):
    with open(source_path + user_list[i]) as f:
        json_obj = json.load(f)

        # plot the initial strategy
        N = int(len(json_obj['initialStrategy']['transects']))
        # plot the title and any other information
        subtitle = ("people: " + str(user_list[i]) +
                    " dataversion: " + str(json_obj['dataVersion']) )
        #             " user_conclusion- " + "local hypo:" + str(json_obj['initialStrategy']['localHypothesis']['nullHypothesis']) + 
        #             str(json_obj['initialStrategy']['localHypothesis']['alternativeHypothesis1']) + 
        #             str(json_obj['initialStrategy']['localHypothesis']['alternativeHypothesis2']) +
        #             " global: " + str(json_obj['initialStrategy']['globalHypothesis']['nullHypothesis']) + 
        #             str(json_obj['initialStrategy']['globalHypothesis']['alternativeHypothesis1']) + 
        #             str(json_obj['initialStrategy']['globalHypothesis']['alternativeHypothesis2'])) 
        fig1, axs = plt.subplots(figsize=(30,13))
        transects_sum = len(json_obj['initialStrategy']['samples'])
        for transects_index in range(transects_sum):
            transects = json_obj['initialStrategy']['samples'][transects_index]
            index = []
            sample_number = []
            
            for locs_number  in range(len(transects)):
                width = 1*transects_index/transects_sum
                locs = transects[locs_number]
                index.append(locs['index'] + width)                    
                sample_number.append(locs['measurements'])
            axs.scatter(index, sample_number, label="transect" + str(transects_index), marker="H", s=100)   
            axs.set_xlabel('loc')
            axs.set_ylabel('samples')
            axs.set_xticks(range(0,23,1))
            axs.set_yticks(range(0,10,1))
            axs.set_title(subtitle, fontsize="40")
            plt.setp( axs.xaxis.get_majorticklabels(), rotation=-45, ha="left" )
            plt.legend()
            
        plt.savefig('./initial_strategy/Initial strategy' + str(i))

        unifying_transect = []
        fig2, axs = plt.subplots(figsize=(33, 20))
        for transects_index in range(transects_sum):
            transects = json_obj['initialStrategy']['samples'][transects_index]
            transects_name = json_obj['initialStrategy']['transects'][transects_index]
            unifying_transect.append("transect" + str(transects_name['number']))
            index = []
            sample_number = []    
            for locs_number  in range(len(transects)):
                width = 1*transects_index/transects_sum
                locs = transects[locs_number]
                index.append(locs['index'] + width)                    
                sample_number.append(locs['measurements'])
            plt.scatter((transects_index+1)*np.ones(len(index)), index, marker="H", s=200)
        plt.ylabel('location')
        plt.title('initial strategy',fontsize=40)
        plt.yticks(range(0,23,1))
        plt.xticks(np.linspace(1, transects_sum, transects_sum), unifying_transect)
        plt.savefig('./initial_strategy/initial_loc_transect' + str(i))


        # plot the actual strategy
        fig, axs = plt.subplots(4,1,figsize=(30,60))
        cmap = plt.get_cmap("tab10")
        # plot the title and any other information
        subtitle = ("people: " + str(user_list[i]) +
                    " dataversion: " + str(json_obj['dataVersion']))
                    # " user_conclusion- " + "local hypo- null:" + str(json_obj['initialStrategy']['localHypothesis']['nullHypothesis']) + 
                    # " alt1: " + str(json_obj['initialStrategy']['localHypothesis']['alternativeHypothesis1']) + 
                    # " alt2: " + str(json_obj['initialStrategy']['localHypothesis']['alternativeHypothesis2']) +
                    # " global hypo- null:" + str(json_obj['initialStrategy']['globalHypothesis']['nullHypothesis']) + 
                    # " alt1: " + str(json_obj['initialStrategy']['globalHypothesis']['alternativeHypothesis1']) + 
                    # " alt2: " + str(json_obj['initialStrategy']['globalHypothesis']['alternativeHypothesis2'])) 
        fig.suptitle(subtitle, fontsize=40)
        
        for transects_index in range(len(json_obj['actualStrategy']['transects'])):
            transects = json_obj['actualStrategy']['transects'][transects_index]
           
            # plot the moisture with location 
            axs[0].scatter(0,-10,s=200,  c=cmap(transects_index), label="transects" + str(transects['number']))
            axs[1].scatter(0,-10,s=200,  c=cmap(transects_index), label="transects" + str(transects['number']))
            axs[2].scatter(0,-10,s=200,  c=cmap(transects_index), label="transects" + str(transects['number']))
            axs[3].scatter(0,-10,s=200,  c=cmap(transects_index), label="transects" + str(transects['number']))
            for sample in transects['samples']:
                if(sample['type'] == "planned"):
                    axs[0].scatter(sample['index']*np.ones(len(sample['moisture'])), np.array(sample['moisture']), marker='D',s=200, c=cmap(transects_index))
                    axs[1].scatter(np.array(sample['moisture']), np.array(sample['shear']), marker='D',s=200, c=cmap(transects_index))
                    axs[2].scatter(np.array(sample['moisture']), np.array(sample['grain']), marker='D',s=200, c=cmap(transects_index))
                    axs[3].scatter(sample['index'], sample['batteryLevelBefore'], marker='D',s=200, c=cmap(transects_index))
                elif(sample['type'] == "deviated" ):
                    axs[transects_index].scatter(sample['index']*np.ones(len(sample['moisture'])), np.array(sample['moisture']), marker='H',s=200, c=cmap(transects_index))
                    axs[transects_index].scatter(sample['index']*np.ones(len(sample['shear'])), np.array(sample['shear']), marker='H',s=200, c=cmap(transects_index))
                    axs[transects_index].scatter(sample['index']*np.ones(len(sample['grain'])), np.array(sample['grain']), marker='H',s=200, c=cmap(transects_index))
                    axs[transects_index].scatter(sample['index'], sample['batteryLevelBefore'], marker='H',s=200, c=cmap(transects_index))
            axs[0].set_xlabel('loc')
            axs[0].set_ylabel('moisture')
            axs[0].set_xticks(range(0,23,1))
            axs[0].set_ylim((-4,20))
            axs[0].yaxis.set_major_locator(MultipleLocator(1))
            axs[0].set_title('moisture vs location', fontsize="40")   
            axs[0].legend()

            axs[1].set_xlabel('moisture')
            axs[1].set_ylabel('shear')
            axs[1].set_ylim((0,10))
            axs[1].yaxis.set_major_locator(MultipleLocator(1))
            axs[1].set_title('shear vs location', fontsize="40")   
            axs[1].legend()

            axs[2].set_xlabel('moisture')
            axs[2].set_ylabel('grain')
            axs[2].set_ylim((0,1))
            axs[2].yaxis.set_major_locator(MultipleLocator(1))
            axs[2].set_title('grain vs location', fontsize="40")   
            axs[2].legend()

            axs[3].set_xlabel('location')
            axs[3].set_ylabel('battery')
            axs[3].set_ylim((-0.2,1))
            axs[3].yaxis.set_major_locator(MultipleLocator(1))
            axs[3].set_title('battery vs location', fontsize="40")   
            axs[3].legend()


                
        plt.savefig('./actual_stratege/actual strategy' + str(i))

        # plot the confidence tendency
        local_y_0 = []
        local_y_1 = []
        local_y_2 = []
        global_y_0 = []
        global_y_1 = []
        global_y_2 = []
        unifying_x = []
        local_y_0.append(json_obj['initialStrategy']['localHypothesis']['nullHypothesis'] - 3)
        local_y_1.append(json_obj['initialStrategy']['localHypothesis']['alternativeHypothesis1'] - 3)
        local_y_2.append(json_obj['initialStrategy']['localHypothesis']['alternativeHypothesis2'] - 3)
        global_y_0.append(json_obj['initialStrategy']['globalHypothesis']['nullHypothesis'] - 3)
        global_y_1.append(json_obj['initialStrategy']['globalHypothesis']['alternativeHypothesis1'] - 3)
        global_y_2.append(json_obj['initialStrategy']['globalHypothesis']['alternativeHypothesis2'] - 3) 
        unifying_x.append('initial')      
        for transects in json_obj['actualStrategy']['transects']:
            local_y_0.append(transects['localHypotheses']['nullHypothesis'] - 3)
            local_y_1.append(transects['localHypotheses']['nullHypothesis'] - 3)
            local_y_2.append(transects['localHypotheses']['nullHypothesis'] - 3)
            global_y_0.append(transects['globalHypotheses']['nullHypothesis'] - 3)
            global_y_1.append(transects['globalHypotheses']['nullHypothesis'] - 3)
            global_y_2.append(transects['globalHypotheses']['nullHypothesis'] - 3)
            unifying_x.append('transect' + str(transects['number']))
        local_y_0.append(json_obj['finalHypotheses']['localHypothesis']['nullHypothesis'] - 3)
        local_y_1.append(json_obj['finalHypotheses']['localHypothesis']['alternativeHypothesis1'] - 3)
        local_y_2.append(json_obj['finalHypotheses']['localHypothesis']['alternativeHypothesis2'] - 3)
        global_y_0.append(json_obj['finalHypotheses']['globalHypothesis']['nullHypothesis'] - 3)
        global_y_1.append(json_obj['finalHypotheses']['globalHypothesis']['alternativeHypothesis1'] - 3)
        global_y_2.append(json_obj['finalHypotheses']['globalHypothesis']['alternativeHypothesis2'] - 3) 
        unifying_x.append('final') 
        plt.figure(figsize=(40, 20))
        plt.plot(np.linspace(1, len(local_y_0), len(local_y_0)), local_y_0, 'ro--', linewidth=5, markersize=40, label="null hypo")
        plt.plot(np.linspace(1, len(local_y_1), len(local_y_1)), local_y_1, 'go--', linewidth=5, markersize=40, label="alte1")
        plt.plot(np.linspace(1, len(local_y_2), len(local_y_2)), local_y_2, 'bo--', linewidth=5, markersize=40, label="alte2")
        plt.ylim((-3, 4))
        plt.xticks(np.linspace(1, len(local_y_0), len(local_y_0)), unifying_x)
        plt.ylabel('confidence')
        plt.legend(fontsize= 40)
        plt.savefig('./hypothesis/local/local_hypothesis_trajectory' + str(i))

        plt.figure(figsize=(40, 20))
        plt.plot(np.linspace(1, len(global_y_0), len(global_y_0)), global_y_0, 'ro--', linewidth=5, markersize=40, label="null hypo")
        plt.plot(np.linspace(1, len(global_y_1), len(global_y_1)), global_y_1, 'go--', linewidth=5, markersize=40, label="alte1")
        plt.plot(np.linspace(1, len(global_y_2), len(global_y_2)), global_y_2, 'bo--', linewidth=5, markersize=40, label="alte2")

        plt.ylim((-3, 4))
        plt.ylabel('confidence')
        plt.legend(fontsize= 40)
        plt.xticks(np.linspace(1, len(local_y_0), len(local_y_0)), unifying_x)
        plt.savefig('./hypothesis/global/global_hypothesis_trajectory' + str(i))
