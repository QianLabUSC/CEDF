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

    'figure.figsize': '20, 20'  #图片尺寸

    }
def gauss(mean, scale, sigma, k, x=np.linspace(0,19,20)):
    return k * scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))
pylab.rcParams.update(myparams)  #更新自己的设置
sample_location = [1,4,7,12,16,21]
sample_numbers = [3,3,5,7,3,4]
# for k in [0.2, 0.4, 0.6, 0.8, 1]:
#     for sigma in [0.5,1,1.5,2,2.5]:
#         name = 'k' + str(k) + 'sigma' + str(sigma)
#         plt.figure()
#         plt.bar(sample_location, sample_numbers,width = 0.4)
#         plt.title('Example for information coverage' + name, fontsize=40)
#         plt.xlabel('location')
#         plt.ylabel('sample numbers')
#         plt.xticks(range(0,23,1))
#         plt.yticks(range(0,8,1))
#         information_converage = np.zeros((22))
#         location = np.linspace(1,22,22)
#         for i in range(len(sample_location)):
#             single_converage = gauss(sample_location[i],sample_numbers[i], sigma, k)
#             information_converage += single_converage
#             plt.plot(location, single_converage,'go')
#             plt.plot(location, single_converage,'g')
#         plt.plot(location, information_converage, 'ro')
#         plt.plot(location, information_converage, 'r')
#         # plt.show()"
#         plt.savefig("./variable_instruction/"+str(k) + str(sigma) + ".jpg")

# DM = rule_state_machine()
# DM.choose_initial_template()
# DM.env.set_state([sample_location, sample_numbers])
# mm, erodi = DM.env.get_data_state()
# DM.handle_feature_point_detection()
# for kk in range(mm.shape[1]):
#     erodi_nonzero = erodi[:,kk][np.nonzero(erodi[:,kk])]
#     plt.scatter(kk*np.ones(len(erodi_nonzero)) + 1, erodi_nonzero, marker='D',s=160, label="sample", c="lime")
# plt.title('Example for discrepancy', fontsize=40)
# plt.xlabel('location')
# plt.ylabel('sample value')
# plt.xticks(range(0,23,1))
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
        #print("%10s" % key, end='')  # 借助 end=''可以不换行
        #print("%10s" % value, end='')  # "%10s" 右对齐
        #print('%16s' % '{:.3%}'.format(value * 1.0 / len(data)))
        results.append(value)
    return results




DM = rule_state_machine()
DM.choose_initial_template()
DM.env.set_state([sample_location, sample_numbers])
mm, erodi = DM.env.get_data_state()
mm_nonzero = mm[np.nonzero(mm)]

# start = 0  # 区间左端点
# number_of_interval = 20  # 区间个数
# length = 1  # 区间长度
# intervals = {'{}~{}'.format(length*x+start, length*(x+1)+start): 0 for x in range(number_of_interval)}  # 生成区间
# result = interval_statistics(mm_nonzero, intervals)
# result_number = np.linspace(0, 19, 20)
# for k in [0.2, 0.4, 0.6, 0.8, 1]:
#     for sigma in [0.5,1,1.5,2,2.5]:
#         name = 'k' + str(k) + 'sigma' + str(sigma)
#         plt.figure()
#         plt.barh(result_number + 0.5, result)
#         plt.title('Example for Variable coverage' + name, fontsize=40)
#         plt.xlabel('variable')
#         plt.ylabel('sample numbers')

#         information_converage = np.zeros((20))
#         for i in range(len(result_number)):
#             single_converage = gauss(result_number[i] +0.5, result[i], sigma, k)
#             information_converage += single_converage
#             plt.plot(single_converage,result_number, 'go')
#             plt.plot( single_converage,result_number,'g')
#         plt.plot( information_converage, result_number,'ro')
#         plt.plot( information_converage, result_number,'r')
#         plt.yticks(range(0, 20))
#         ax = plt.gca()
#         ax.invert_xaxis()
#         ax.yaxis.set_ticks_position('right')
#         # plt.show()"
#         plt.savefig("./variable_instruction/variable_space/"+str(k) + str(sigma) + ".jpg")

plt.figure()
# plot moisture vs location
for index in sample_location:
    mm_loc_nonzero = mm[:,index][np.nonzero(mm[:,index])]
    print(mm_loc_nonzero)
    mm_number = len(mm_loc_nonzero)
    plt.scatter(np.ones(mm_number) * index, mm_loc_nonzero, marker='D',s=160)
plt.xticks(range(1, 23))
plt.yticks(range(0, 20))
plt.xlabel('loc')
plt.ylabel('moisture')
plt.title('moisture vs location')
plt.savefig("./variable_instruction/variable_space/moisvsloc.jpg")


# plt.hist(mm_nonzero)
# plt.xticks(range(0,23,1))
# plt.show()
# plt.title('Example for discrepancy', fontsize=40)
# plt.xlabel('location')
# plt.ylabel('sample value')
# plt.xticks(range(0,23,1))
# plt.show()