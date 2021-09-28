# This FILE is part of multi-legged robot field exploration model
# env_wrapper.py - to obtain user interaction data from website
#
# This programm is explained by roboLAND in university of southern california.
# Please notify the source if you use it
# 
# Copyright(c) 2021-2025 Ryoma Liu
# Email: 1196075299@qq.com

from env_wrapper import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize
import random
import matplotlib.pylab as pylab
import numpy as np
from PIL import Image
from math import *

class rule_state_machine:
    def __init__(self):
        '''Initial env info and parameters for decision making
        '''
        self.states = ['Initial', 'Exploration', 'Verification']
        self.current_state = 0
        self.env = ENV()
        self.hypo_locations = (['No','Feature_low','Feature_middle',
                                'Feature_high'])
        self.hypo_location = 0
        self.hypo_samples = (['No','Feature_low', 'Feature_middle',
        'Feature_high'])
        self.hypo_sample = 0
        self.information_matrix = []
        self.accuracy_matrix = []
        self.fitting_error_matrix = []


    def set_init_hypo(self, hypo_location, hypo_sample):
        self.hypo_location = hypo_location
        self.hypo_sample = hypo_sample

    def choose_initial_template(self):
        '''choose initial template

        According to the initial knowledge and hypothesis, human will select a
        experience data sample distribution

        Args:
            self.hypo_location: inital hypo about data location feature
            self.hypo_sample : initial hypo about data sample feature

        Returns:
            change the initial template in env wrapper
        '''
        if(self.hypo_location == 0):
            location_index = [1,9,13,21]
        elif(self.hypo_location == 1):
            location_index = [1,4,7,11,16,21]
        elif(self.hypo_location == 2):
            location_index = [1,5,9,12,15,21]
        elif(self.hypo_location == 3):
            location_index = [1,6,11,14,17,20]
        if(self.hypo_sample == 0):
            sample_index = [3,3,3,3]
        elif(self.hypo_sample == 1):
            sample_index = [5,5,3,3,3,3]
        elif(self.hypo_sample == 2):
            sample_index = [3,3,5,5,3,3]
        elif(self.hypo_sample == 3):
            sample_index = [3,3,3,3,5,5]
        initial_action = [location_index, sample_index]
        self.env.initiate_template(initial_action)

    def handle_information_coverage(self):
        sample_state = self.env.get_state()
        self.information_matrix = np.zeros(22)     #information matrix in location
        self.variable_coverage = np.zeros(20)
        for i in range(len(sample_state[0])):
            
            scale = 0.1 * sample_state[1][i] + 1
            locs =  sample_state[0][i] + 1
            self.information_matrix += gauss(locs, scale)
            # print(self.information_matrix)
            # print(gauss(locs, scale))
            # self.plot_line('cool green', np.linspace(1,22,22), gauss(locs, scale), 'test'+str(i))
        # print("coverage_matrix: ", self.information_matrix)
        mm, erodi = self.env.get_data_state()
        mm_mean = np.mean(mm, axis=0)
        mm_nonzero = mm[np.nonzero(mm)]
        start = -10  # 区间左端点
        number_of_interval = 20  # 区间个数
        length = 1  # 区间长度
        intervals = {'{}~{}'.format(length*x+start, length*(x+1)+start): 0 for x in range(number_of_interval)}  # 生成区间
        result = interval_statistics(mm_nonzero, intervals)
        result_number = np.linspace(-10, 9, 20)
        variable_information = np.zeros(20)
        for i in range(len(result_number)):
            single_converage = gauss_variable(result_number[i] +0.5, result[i])
            variable_information += single_converage
        loc_variable_information = []
        for i in mm_mean:
            interval_i = int(i)
            loc_variable_information.append(variable_information[interval_i])
        self.loc_variable_coverage = np.zeros(22)
        for i in range(len(sample_state[0])):
            scale = loc_variable_information[i]
            locs =  sample_state[0][i] + 1
            self.loc_variable_coverage += gauss(locs, scale)

    def handle_information_accuracy(self):
        accuracy_matrix = []
        mm, data_state = self.env.get_data_state()
        loc_state = self.env.get_state()
        # error_cost = np.std(data_state, axis=0)
        for col in range(data_state.shape[1]): 
            if col in loc_state[0]:
                effective_data = data_state[:,col][np.nonzero(data_state[:,col])]
                # print(effective_data)
                median = np.median(effective_data) 
                k1 = 1.4826
                mad = k1 * np.median(np.abs(effective_data-median))
                lower_limit = median - (3*mad)
                upper_limit = median + (3*mad)
                outlier_data_num = (len(effective_data[(effective_data> 
                                upper_limit) & (effective_data<lower_limit)]))
                data_samples = len(effective_data)
                if(data_samples == 0):
                    total_cost = 0
                elif(data_samples > 0):
                    total_cost = 1 - 1/(1+ (data_samples - 0.99)/(3*outlier_data_num + 1))
                    accuracy_matrix.append(total_cost)
            else:
                accuracy_matrix.append(0)
        self.accuracy_matrix = accuracy_matrix
        # print('accuracy_matrix: ', self.accuracy_matrix)


    def handle_feature_point_detection(self):
        loc_state = self.env.get_state()[0]
        print(self.env.get_state())
        self.fitting_error_matrix = np.zeros(22)
        mm, erodi = self.env.get_data_state()
        mm_mean = np.mean(mm, axis=0)
        mm_nonzeroindex = (mm_mean != 0)
        erodi_mean = np.mean(erodi, axis=0)
        self.loc_index = np.linspace(1,22,22)[mm_nonzeroindex]
        data_index = mm_mean[mm_nonzeroindex]
        data_mean = erodi_mean[mm_nonzeroindex]
        p , e = optimize.curve_fit(piecewise_linear, data_index, data_mean)
        # print('dfadfaaf', p)
        xd = np.linspace(0, np.max(data_index), 22)
        fit_curve = piecewise_linear(xd, *p)
        fitting_results = piecewise_linear(data_index, *p)
        self.fitting_results = fitting_results
        fitting_error = fitting_results - data_mean
        mm_mean[mm_nonzeroindex] = fitting_error
        self.data_index = data_index
        self.fitting_error_matrix[mm_nonzeroindex] = fitting_error

        # print(data_mean)
        nonzero_data_mean = data_mean[np.nonzero(data_mean != 0)]
        rmse_data = (sqrt(np.sum(np.power(nonzero_data_mean, 2))/
                                    np.size(nonzero_data_mean)))
        # print(rmse_data)
        self.rmse_data = rmse_data
        # plt.plot(xd, fit_curve)
        # plt.plot(data_index, data_mean, "o")
        # plt.plot(data_index, fitting_results, "*")
        # #plt.plot(data_index, fitting_error, "+")
        # plt.show()
        # plt.savefig('123.png')


        # find the feature point location
        array = np.asarray(data_index)
        idx = (np.abs(array - p[0])).argmin()
        loc_indx = loc_state[idx]
        saturation_estimated = int(loc_indx * (p[0]/array[idx]))
        self.saturation_selection = np.arange(saturation_estimated - 2, saturation_estimated + 3, 1)
        

    def confidence_model(self):
        non_zero_matrix = (self.fitting_error_matrix[np.nonzero
                                    (self.fitting_error_matrix != 0)])
        rmse = (sqrt(np.sum(np.power(non_zero_matrix, 2))/
                                    np.size(non_zero_matrix)))
        # print(rmse)
        # print(self.fitting_error_matrix)
        # print(non_zero_matrix)
        whole_rmse_percentage = rmse/self.rmse_data
        # print(whole_rmse_percentage)
        confindence = (0.04 - whole_rmse_percentage) * 30 * self.coverage_criteria
        # print(confindence)

    def handle_state_judge(self):
        if(self.current_state == 0):
            self.current_state = 1
        elif(self.current_state == 1):
            if(np.min(self.accuracy_matrix) > 0.7 and 
            len(self.information_matrix[self.information_matrix > 0.8]) > 15):
                self.current_state = 2
            else: 
                self.current_state = 1
        elif(self.current_state == 2):
            if(len(self.fitting_error_matrix[self.fitting_error_matrix > 0.8]) > 0):
                self.current_state = 1
            elif():
                self.current_state = 2
    
    def information_model(self):
        self.coverage_criteria = (len(self.information_matrix[self.information_matrix
                             > 0.3]) / 22)
        accuracy_matrix = np.array(self.accuracy_matrix)
        # print(accuracy_matrix)
        self.accuracy_criteria = (len(accuracy_matrix[(accuracy_matrix > 0.6) & (accuracy_matrix != 0)]) /
                        len(accuracy_matrix[accuracy_matrix != 0]))
        
        # print('accuracy_value:', self.accuracy_criteria)    # percentage of locs which the accuracy is lower than 0.6
        # print('coverage_value:', self.coverage_criteria)    # percentage of locs which the information is lower than 0.8
        

    def take_action(self):
        if(self.current_state == 0):
            self.choose_initial_template()
        elif(self.current_state == 1):
            action_loc = np.argmin(self.information_matrix)
            self.env.set_action([action_loc],[3])
            accuracy_loc = np.where(self.accuracy_matrix < 0.7)
            accuracy_samples = np.ones(len(accuracy_loc))
            self.env.set_action(accuracy_loc,accuracy_samples) 
        elif(self.current_state == 2):
            fitting_error_loc = np.where(self.fitting_error_matrix > 0.8)
            add_loc = []
            add_samples = []
            current_state = self.env.get_state()
            for i in fitting_error_loc:
                if not i+1 in current_state[0]:
                    add_loc.append(i+1)
                    add_samples.append(3)
                if not i-1 in current_state[0]:
                    add_loc.append(i-1)
                    add_samples.append(3)
            self.env.set_action(add_loc, add_samples)

    def plot(self, color, name):
        myparams = {

        'axes.labelsize': '10',

        'xtick.labelsize': '10',

        'ytick.labelsize': '10',

        'lines.linewidth': 1,

        'legend.fontsize': '3',

        'font.family': 'Times New Roman',

        'figure.figsize': '9, 5'  #图片尺寸

        }

        pylab.rcParams.update(myparams)  #更新自己的设置
        # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #线型设置
        
        fig1 = plt.figure(1)
        a = plt.plot(self.coverage_criteria, self.accuracy_criteria ,marker='o', color=sns.xkcd_rgb[color],
                markersize=5)
        
        plt.legend(loc="lower right")  #图例位置 右下角
        plt.ylabel('accuracy') 
        plt.xlabel('coverage ') 
        plt.xlim((0, 1.1))
        plt.ylim((0, 1.1))
        plt.axvline(x=1, c="b", ls="--", lw=1)
        plt.axhline(y=1, c="b", ls="--", lw=1)
        plt.savefig(name)

        #注意.show()操作后会默认打开一个空白fig,此时保存,容易出现保存的为纯白背景,所以请在show()操作前保存fig.
        # plt.show()

def interval_statistics(data, intervals):
    if len(data) == 0:
        return
    for num in data:
        for interval in intervals:
            lr = tuple(interval.split('~'))
            left, right = float(lr[0]), float(lr[1])
            if left <= num <= right:
                intervals[interval] += 1
    results = []
    for key, value in intervals.items():
        print("%10s" % key, end='')  # 借助 end=''可以不换行
        print("%10s" % value, end='')  # "%10s" 右对齐
        print('%16s' % '{:.3%}'.format(value * 1.0 / len(data)))
        results.append(value)
    return results


def piecewise_linear(x, x0, y0, k1):
	# x<x0 ⇒ lambda x: k1*x + y0 - k1*x0
	# x>=x0 ⇒ lambda x: k2*x + y0 - k2*x0
    return np.piecewise(x, [x < x0, x >= x0], [lambda x:k1*x + y0-k1*x0, 
                                   lambda x: y0])

def gauss(mean, scale, x=np.linspace(1,22,22), sigma=1):
    return scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))
def gauss_variable(mean, scale, x=np.linspace(-10,9,20), sigma=1):
    return scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))
if __name__ == "__main__":
    DM = rule_state_machine()
    DM.choose_initial_template()
    # x = np.linspace(1,22,22)
    # information_matrix = gauss(1,0.1).reshape(22,1)
    # print(information_matrix)
    # sns.set()
    # ax = sns.heatmap(information_matrix, vmin=0, vmax=1)
    # plt.title('Information Matrix')
    # plt.savefig("test.png") 
    DM.handle_information_accuracy()
    DM.handle_information_coverage()
    DM.information_model()
    DM.plot('cool green','test') 
    DM.handle_feature_point_detection()
    DM.confidence_model()
  