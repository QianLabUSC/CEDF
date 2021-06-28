# This FILE is part of multi-legged robot field exploration model
# env_wrapper.py - to obtain user interaction data from website
#
# This programm is explained by roboLAND in university of southern california.
# Please notify the source if you use it
# 
# Copyright(c) 2021-2025 Ryoma Liu
# Email: 1196075299@qq.com

from env_wrapper import *

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
        

