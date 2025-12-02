from tkinter import Label
import matplotlib.pyplot as plt 
import cvxpy as cp
import numpy as np
import time

def tsp_path(UGV_location_all, user_location_all, D_all, v_all, M):
    
    # plot starting point
    p0 = plt.plot(UGV_location_all[0,0], UGV_location_all[1,0],'ob', markersize=15, label="Starting point")
    # plot all vertices
    p1 = plt.plot(UGV_location_all[0,:], UGV_location_all[1,:],'s', markersize=4, label="Vertices")
    # plot IoT users
    # p2 = plt.plot(user_location_all[0,:], user_location_all[1,:],'+', markersize=4, label="IoT users")

    D = D_all[:,:]
    v_opt = v_all

    # plot the shortest path visiting all vertices
    # solve the TSP using Gurobi
    x = cp.Variable((M, M), boolean=True)
    u = cp.Variable((M, 1))

    cost = 0
    constraints = []
        
    cost += cp.trace(np.transpose(D) @ x) # minimize path length

    # divergence constraint
    for m in range(M):
        constraints += [sum(x[m,:])==1] 
        constraints += [sum(x[:,m])==1] 
        constraints += [x[m,m]==0]

    # subtour elimination constraint
    for m in range(1, M):
        for j in range(1, M):
            if m!= j:
                constraints += [ u[m]-u[j]+(M-1)*x[m,j]+(M-3)*x[j,m]<=M-2 ]

    for m in range(1, M):
        u[m] >= 1
        u[m] <= M-1

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # Sm = np.trace(np.transpose(D) @ x.value) / a 
     
    # plot the proposed path 
    MM=sum(v_opt)
    if MM == 1:
        Sm = 0
        Em = 0
        x= np.zeros((M, M))
        # p4=plt.plot(UGV_location_all[0,0],UGV_location_all[1,0],'-k', label='Optimal path1')

    else:
        epsilon=1e3

        x = cp.Variable((M, M), boolean=True)
        u = cp.Variable((M, 1))

        cost = 0
        constraints = []
        
        cost += cp.trace(np.transpose(D) @ x) # minimize path length

        for m in range(M):
            if v_opt[m]==1:
                constraints += [sum(x[m,:])>=1] 
                constraints += [sum(x[:,m])>=1] 
            else:
                constraints += [sum(x[m,:])==0] 
                constraints += [sum(x[:,m])==0]

            constraints += [ x[m, m] == 0 ]

        for m in range(1, M):
            for j in range(1, M):
                if m!=j and v_opt[m]==1 and v_opt[j]==1:
                   constraints+= [ u[m]-u[j]+(MM-1)*x[m,j]+(MM-3)*x[j,m]<=MM-2 ]   

        for m in range(1, M):
            u[m] >= v_opt[m]
            u[m] <= (MM- 1) * v_opt[m]
        
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        point_list = []
        flag = 0
        num = 0
        m = 0

        print(UGV_location_all)
        print(UGV_location_all[1,3])

        while (num <= sum(v_opt)):
                for j in range(M):
                    if x.value[m, j] > 0.5:
                        tan_theta  = ( UGV_location_all[1,j] - UGV_location_all[1,m] ) / ( UGV_location_all[0,j] - UGV_location_all[0,m]  )
                        
                        if (UGV_location_all[1,j] - UGV_location_all[1,m] >= 0 ):
                            theta = np.arctan(tan_theta)
                        else: 
                            theta = 3.1415926 + np.arctan(tan_theta)

                        if (theta >= 3.1415926):
                            theta = theta - 2*3.1415926

                        if m == 0 and (theta >= 1.57 or theta <= -1.57):
                            pass
                        else:
                            if flag == 0:
                                p4 = plt.plot(UGV_location_all[0,[m,j]],UGV_location_all[1,[m,j]],'--m', label='Global path')
                                flag =1
                                point_list.append([UGV_location_all[0,m], UGV_location_all[1,m], theta])
                                point_list.append([UGV_location_all[0,j], UGV_location_all[1,j], theta])
                                num += 2
                            else:
                                p4 = plt.plot(UGV_location_all[0,[m,j]],UGV_location_all[1,[m,j]],'--m')
                                point_list.append([UGV_location_all[0,j], UGV_location_all[1,j], theta])
                                num += 1

                            m = j
                            break

        np.savetxt('point_list_ref.txt', np.array(point_list), fmt='%s', delimiter=' ')

    plt.legend()
    # plt.show()
    # print('Figure will close in 3 sceonds *** ')
    # time.sleep(3)
    # plt.close()
