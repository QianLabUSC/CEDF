from math import inf
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from rule_based_decision_making import *
# this is the python code for variable instruction
myparams = {

    'axes.labelsize': '40',

    'xtick.labelsize': '40',

    'ytick.labelsize': '40',

    'lines.linewidth': 1,

    'legend.fontsize': '40',

    'font.family': 'Times New Roman',

    'figure.figsize': '20, 10'  #图片尺寸

    }
def gauss(mean, scale, x=np.linspace(1,22,22), sigma=1.5):
    return 0.4 * scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))
pylab.rcParams.update(myparams)  #更新自己的设置
sample_location = [1,4,7,12,16,21]
sample_numbers = [3,3,5,7,3,4]
# plt.bar(sample_location, sample_numbers,width = 0.4)
# plt.title('Example for information coverage', fontsize=40)
# plt.xlabel('location')
# plt.ylabel('sample numbers')
# plt.xticks(range(0,23,1))
# plt.yticks(range(0,8,1))
# information_converage = np.zeros((22))
location = np.linspace(1,22,22)
# for i in range(len(sample_location)):
#     single_converage = gauss(sample_location[i],sample_numbers[i])
#     information_converage += single_converage
#     plt.plot(location, single_converage,'go')
#     plt.plot(location, single_converage,'g')
# plt.plot(location, information_converage, 'ro')
# plt.plot(location, information_converage, 'r')
# plt.show()

DM = rule_state_machine()
DM.choose_initial_template()
DM.env.set_state([sample_location, sample_numbers])
mm, erodi = DM.env.get_data_state()
DM.handle_feature_point_detection()
for kk in range(mm.shape[1]):
    erodi_nonzero = erodi[:,kk][np.nonzero(erodi[:,kk])]
    plt.scatter(kk*np.ones(len(erodi_nonzero)) + 1, erodi_nonzero, marker='D',s=160, label="sample", c="lime")
plt.title('Example for discrepancy', fontsize=40)
plt.xlabel('location')
plt.ylabel('sample value')
plt.xticks(range(0,23,1))
plt.show()
