import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from numba import njit , vectorize
from numba.typed import List
startTime = time.time()


#data#
R = 1
V = 1
d = 0.025



#points initialization#
def calc_p(R, d):
    points_lst = []
    for y in range(-math.ceil((R - (d / 2)) / d), math.ceil((R - (d / 2)) / d) + 1):
        for x in range(0, math.ceil((R - (d / 2)) / d) + 1):
            if math.sqrt(((d / 2) + x * d) ** 2 + ((d / 2) - y * d) ** 2) < R:#half circle #
                points_lst.append(((d / 2) + x * d, (d / 2) - y * d))
                points_lst.append((-((d / 2) + x * d), (d / 2) - y * d))#other half#
    return(List(points_lst))

points_lst = calc_p(R, d)


# Discretion to a quarter circle #
def mat_to_discrete_circle(points_lst, d, D):
    mat_aa = np.zeros(shape=(len(points_lst), len(points_lst)))
    mat_ab = np.zeros(shape=(len(points_lst), len(points_lst)))
    diag=0
    k = 8987551788
    approx = 3.1685*(10**10)# approximation from "moment method instructions" 0.8814/(pi * eps0)
    for i in range(0, len(points_lst)):#fill L_mn matrix#
        for j in range(diag, len(points_lst)):
            if i == j:#Main diagonal#
                mat_aa[i][j]=(d*approx) #laa
                mat_ab[i][j]=((k*d**2) / math.sqrt((((points_lst[i][0] - points_lst[j][0])) ** 2 + ((points_lst[i][1] - points_lst[j][1])) ** 2 + D ** 2))) #lab#
            else:#Organs outside the main diagonal#
                mat_aa[i][j]=((k*d**2) / math.sqrt((((points_lst[i][0] - points_lst[j][0])) ** 2 + ((points_lst[i][1] - points_lst[j][1])) ** 2)))
                mat_ab[i][j] = ((k*d**2) / math.sqrt((((points_lst[i][0] - points_lst[j][0])) ** 2 + ((points_lst[i][1] - points_lst[j][1])) ** 2 + D ** 2)))
        diag += 1
    return (mat_aa, mat_ab) #Returns the matrix lmn#



#Fills a symetric top triangular matrix#
def complete_top_tri_mat(tri_mat):
    n = tri_mat.shape[0]
    for r in range(1, n):
        for c in range(r):
            tri_mat[r, c] = tri_mat[c, r]
    return tri_mat



#Creating a vector of potential and solving a system of equations#
def solve_vector_matrix(matrix, lst, v1, v2):
    v_vec1 = np.array([v1 for i in range(math.ceil(len(matrix) / 2))], dtype ='float32')
    v_vec2 = np.array([v2 for i in range(math.ceil(len(matrix) / 2))], dtype='float32')
    v_vec = np.concatenate((v_vec1,v_vec2), axis=0)
    sol = np.linalg.solve(matrix, v_vec)
    return sol



#parts a,b#
#D_lst = [R / 2, R / 5] #list of distances between plates#
D_lst = [R / 2]
Q_lst = [] #total charge list#
c_lst = [] #capacity list#
Err_lst = [] #error list#
A_e = 2.78162514 * (10 ** -11)#(Circle area * epsilon) with R=1#
for D in D_lst:
    mat_aa, mat_ba = mat_to_discrete_circle(points_lst, d, D)
    mat_laa = complete_top_tri_mat(mat_aa)
    mat_lba = complete_top_tri_mat(mat_ba)
    mat_lab = mat_lba.transpose()
    matrix1 = np.concatenate((mat_laa,mat_lba), axis=1)
    matrix2 = np.concatenate((mat_lab,mat_laa), axis=1)
    matrix = np.concatenate((matrix1,matrix2), axis=0)#mat L_mn#
    Xj1 = (solve_vector_matrix(matrix, points_lst, V / 2, -V / 2))
    Xj_p = np.split(Xj1,2)
    Q_lst.append(((Xj_p[0].sum()) * ((d) ** 2)))
    c_lst = Q_lst * int(1 / V)  # C=q/V
    Err_lst.append((abs(c_lst[D_lst.index(D)] - A_e / D) / (A_e / D)) * 100)#compute error#

for Q in range(len(Q_lst)):
    print('in question 2 part ' + str(Q+1) +', the computed Charge is: ' + str(Q_lst[Q]) + ', the computed Capcitance is: ' + str(c_lst[Q]) +',and the error is ' + str(Err_lst[Q]))





# #part c#
# c_lst2 = [] #capacity list#
# peri = 0.025 #period#
# for D_c in np.arange(d/3, 1, peri):
#     mat_aa, mat_ba = mat_to_discrete_circle(points_lst, d, D_c)
#     mat_laa = complete_top_tri_mat(mat_aa)
#     mat_lba = complete_top_tri_mat(mat_ba)
#     mat_lab = mat_lba.transpose()
#     matrix1 = np.concatenate((mat_laa, mat_lba), axis=1)
#     matrix2 = np.concatenate((mat_lab, mat_laa), axis=1)
#     matrix = np.concatenate((matrix1, matrix2), axis=0) #mat L_mn#
#     Xj2 = (solve_vector_matrix(matrix, points_lst, V / 2, -V/2))
#     Xj_2p = np.split(Xj2, 2)
#     c_lst2.append(((Xj_2p[0].sum()) * ((d) ** 2)))# C=q/V
#
# real_val = [A_e/elem for elem in np.arange(d/3, 1, peri)] #theoretical capacity value#
#
# ##### plot part c #####
#
# x1 = np.arange(d/3, 1, peri)
# x2 = np.arange(d/3, 1, peri)
#
# y1 = c_lst2
# y2 = real_val
#
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'r.-')
# plt.title('Capcitance as function of D')
# plt.ylabel('computed C(D) [F]')
# plt.grid()
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, 'b.-')
# plt.xlabel('D [M]')
# plt.ylabel('real C(D) [F]')
# plt.grid()
# plt.show()





#part d#
D_d = R/2
mat_aa, mat_ba = mat_to_discrete_circle(points_lst, d, D_d)
mat_laa = complete_top_tri_mat(mat_aa)
mat_lba = complete_top_tri_mat(mat_ba)
mat_lab = mat_lba.transpose()
matrix1 = np.concatenate((mat_laa,mat_lba), axis=1)
matrix2 = np.concatenate((mat_lab,mat_laa), axis=1)
matrix = np.concatenate((matrix1,matrix2), axis=0)#mat L_mn#
Xj_d = (solve_vector_matrix(matrix, points_lst, V , 0))
Q_d = ((Xj_d.sum()) * ((d) ** 2))# Q=C

print('in question 2 part a the computed Charge was: ' + str(c_lst[0]) + '.  in part d, for a grounded disc - the total charge on both disks is: ' + str(Q_d))
print('the difference is: |' + str(abs(c_lst[0]-Q_d))+'|')