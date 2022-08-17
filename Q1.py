import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from numba import njit , vectorize
from numba.typed import List
startTime = time.time()

##### data #####
R = 1
v = 1

#points initialization#
def calc_p(R, d):
    points_lst = []
    for y in range(-math.ceil((R - (d / 2)) / d), math.ceil((R - (d / 2)) / d) + 1):
        for x in range(0, math.ceil((R - (d / 2)) / d) + 1):
            if math.sqrt(((d / 2) + x * d) ** 2 + ((d / 2) - y * d) ** 2) < R:#half circle #
                points_lst.append(((d / 2) + x * d, (d / 2) - y * d))
                points_lst.append((-((d / 2) + x * d), (d / 2) - y * d))#other half#
    return(List(points_lst))



#Fills a symetric top triangular matrix#
def complete_top_tri_mat(tri_mat):
    n = tri_mat.shape[0]
    for r in range(1, n):
        for c in range(r):
            tri_mat[r, c] = tri_mat[c, r]
    return tri_mat



# Discretion to a quarter circle #
def mat_to_discrete_circle(points_lst, d):
    mat = np.zeros(shape=(len(points_lst), len(points_lst)))
    diag=0
    k = 8987551788
    approx = 3.16865 * (10 ** 10) # approximation from "moment method instructions" 0.8814/(pi * eps0)
    for i in range(0, len(points_lst)):#fill L_mn matrix#
        for j in range(diag, len(points_lst)):
            if i == j:#Main diagonal#
                mat[i][j]=(d * approx)
            else:#Organs outside the main diagonal#
                mat[i][j]=((k * d ** 2) / math.sqrt((((points_lst[i][0] - points_lst[j][0])) ** 2 + ((points_lst[i][1] - points_lst[j][1])) ** 2)))
        diag += 1
    return (mat) #Returns the matrix lmn#




#Creating a vector of potential and solving a system of equations#
def solve_vector_matrix(matrix, len, v):
    v_vec= np.array([v for i in range(len)],dtype = 'float32')
    sol = np.linalg.solve(matrix, v_vec)
    return sol


#### main code ####

d_lst = [0.02, 0.025, 0.05, 0.075, 0.1, 0.12, 0.15, 0.25]
q_lst = []
for d in d_lst:
    points = calc_p(R, d)
    mat_lmn = complete_top_tri_mat(mat_to_discrete_circle(points, d))
    n = len(points)
    Xj = solve_vector_matrix(mat_lmn, n, v) # --Xj-- Question b sector 1#
    q_lst.append(Xj.sum() * ((d) ** 2))
for Q in range(len(q_lst)):
    print('in question 1 part b.2, for d = ' + str(d_lst[Q]) + ', the computed Charge is: ' + str(q_lst[Q]))


##### plot #####
plt.plot(d_lst, q_lst)
plt.title("Q(d)")
plt.xlabel("d [length]")
plt.ylabel("Q [c]")
plt.show()