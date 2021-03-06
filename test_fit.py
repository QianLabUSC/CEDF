from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 1.1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15], dtype=float)
y = np.array([5, 3, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 
	                 84.47, 98.36, 112.25, 126.14, 140.03])

# 一个输入序列，4个未知参数，2个分段函数
def piecewise_linear(x, x0, y0, k1, k2):
	# x<x0 ⇒ lambda x: k1*x + y0 - k1*x0
	# x>=x0 ⇒ lambda x: k2*x + y0 - k2*x0
    return np.piecewise(x, [x < x0, x >= x0], [lambda x:k1*x + y0-k1*x0, 
                                   lambda x:k2*x + y0-k2*x0])

def gauss(mean, scale, x=np.linspace(1,22,22), sigma=4):
    return scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))

# # 用已有的 (x, y) 去拟合 piecewise_linear 分段函数
# p , e = optimize.curve_fit(piecewise_linear, x, y)

# xd = np.linspace(0, 15, 100)
# plt.plot(x, y, "o")
# plt.plot(xd, piecewise_linear(xd, *p))
# plt.savefig('123.png')
xi = np.linspace(1,22,22)
information_matrix = np.zeros((22))
x = [1, 13]
for i in range(len(x)):
	information_matrix += gauss(x[i],1)
	# plt.plot(xi, information_matrix)
plt.plot(xi, information_matrix)
plt.show()