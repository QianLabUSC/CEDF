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
            location_index = [1,5,9,13,17,21]
        elif(self.hypo_location == 1):
            location_index = [1,4,7,11,16,21]
        elif(self.hypo_location == 2):
            location_index = [1,5,9,12,15,21]
        elif(self.hypo_location == 3):
            location_index = [1,6,11,14,17,20]
        if(self.hypo_sample == 0):
            sample_index = [3,3,3,3,3,3]
        elif(self.hypo_sample == 1):
            sample_index = [5,5,3,3,3,3]
        elif(self.hypo_sample == 2):
            sample_index = [3,3,5,5,3,3]
        elif(self.hypo_sample == 3):
            sample_index = [3,3,3,3,5,5]
        initial_action = [location_index, sample_index]
        self.env.initiate_template(initial_action)

    # def take_action(self):
    #     if(self.current_state == 0):
    #     elif(self.current_state == 1):
    #     elif(self.current_state == 2):

    def handle_information_coverage(self):
        sample_state = env.get_state()
        for i in len(sample_state[0]):
            x = np.linspace(1,22,22)     #information matrix in location
            scale = 0.1 * sample_state[1][i] + 1
            locs =  sample_state[0][i]
            self.information_matrix = gauss(locs, scale).reshape(22, 1)

    # def handle_information_accuracy(self):
    #     data_state = env.get_data_state()


def gauss(mean, scale, x=np.linspace(1,22,22), sigma=1):
    return scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))

if __name__ == "__main__":
    DM = rule_state_machine()
    x = np.linspace(1,22,22)
    information_matrix = gauss(x).reshape(22,1)
    print(information_matrix)
    sns.set()
    ax = sns.heatmap(information_matrix, vmin=0, vmax=1)
    plt.title('Information Matrix')
    plt.savefig("test.png")  
