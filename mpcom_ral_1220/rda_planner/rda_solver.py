
'''
RDA solver 
Author: Han Ruihua
'''
import csv
import cvxpy as cp
import numpy as np
import os
import math
from pathos.multiprocessing import Pool
from math import sin, cos, tan, inf

import time
from collections import namedtuple

# para_obstacle = namedtuple('obstacle', ['At', 'bt', 'cone_type'])
pool = None
global iteration_n
iteration_n = 0
iot_file_path = "./iot_Tjunc.txt"

with open(iot_file_path, "r") as file:
    iot_list = [line.strip().split(",") for line in file]

nnn_iot = len(iot_list)  # 设备数量

if nnn_iot > 0:
    iotpt_value = [20] * nnn_iot
    iot_x_value = [0] * nnn_iot
    iot_y_value = [0] * nnn_iot
    iotbk_value = [0] * nnn_iot

    for noi, line in enumerate(iot_list):
        try:
            iot_x_value[noi], iot_y_value[noi], iotbk_value[noi] = map(float, line)
        except ValueError:
            print(f"无效的坐标数据: {line}")
else:
    print("文件为空或无有效数据。")






class RDA_solver:
    def __init__(self, receding, car_tuple, obstacle_template_list=[{'edge_num': 10, 'obstacle_num': 10, 'cone_type': 'norm2'}, {'edge_num': 4, 'obstacle_num': 1, 'cone_type': 'Rpositive'}], 
                        iter_num=2, step_time=0.1, iter_threshold=0.2, process_num=4, **kwargs) -> None:

        '''
        obstacle_template_dict: the template for the obstacles to construct the problem, 
            edge_num: number of convex obstacle edges; 
            obstacle_num: number of convex obstacles; 
            cone_type: Rpositive, norm2
        '''

        # setting
        self.Iot_com_all = 0
        self.single_iot_com = [0] * nnn_iot
        self.single_iot = [0] * nnn_iot
        self.Itera_time = 0
        self.T = receding
        self.car_tuple = car_tuple # car_tuple: 'G h cone wheelbase max_speed max_acce'
        self.L = car_tuple.wheelbase
        self.max_speed = np.c_[self.car_tuple.max_speed]
        self.obstacle_template_list = obstacle_template_list
        self.obstacle_template_num = sum([ ot['obstacle_num'] for ot in obstacle_template_list])        
        self.iter_num = iter_num
        self.dt = step_time
        self.acce_bound = np.c_[car_tuple.max_acce] * self.dt 
        self.iter_threshold = iter_threshold
        

        # independ variable and cvxpy parameters definition
        self.definition(obstacle_template_list, **kwargs)

        # flag
        # self.init_flag = True
        self.process_num = process_num

        if process_num == 1:
            self.construct_problem(**kwargs)
        elif process_num > 1:
            global pool 
            pool = self.construct_mp_problem(process_num, **kwargs)
    
    # region: definition of variables and parameters
    def definition(self, obstacle_template_list, **kwargs):

        self.state_variable_define()
        self.dual_variable_define(obstacle_template_list)
        
        self.state_parameter_define()
        self.dual_parameter_define(obstacle_template_list)

        self.obstacle_parameter_define(obstacle_template_list)

        self.adjust_parameter_define(**kwargs)

        self.combine_parameter_define(obstacle_template_list)

    def state_variable_define(self):
        # decision variables
        self.indep_s = cp.Variable((3, self.T+1), name='state')
        
        self.indep_u = cp.Variable((2, self.T), name='vel')
        self.indep_dis = cp.Variable((1, self.T), name='distance', nonneg=True)
        self.iot_dis_var = cp.Variable((3, self.T+1), name='iot_var')
        self.indep_s.value = np.zeros((3, self.T+1))  

        self.indep_rot_list = [cp.Variable((2, 2), name='rot_'+str(t))  for t in range(self.T)]  # the variable of rotation matrix

    def dual_variable_define(self, obstacle_template_list):
        
        '''
        define the indep_lam; indep_mu; indep_z
        ''' 
        self.indep_lam_list = []
        self.indep_mu_list = []
        self.indep_z_list = []

        for ot in obstacle_template_list:
            self.indep_lam_list += [ cp.Variable((ot['edge_num'], self.T+1), name='lam_'+ str(ot['edge_num']) + '_' + str(index)) for index in range(ot['obstacle_num'])]
            self.indep_mu_list += [ cp.Variable((self.car_tuple.G.shape[0], self.T+1), name='mu_'+ str(ot['edge_num']) + '_' + str(index)) for index in range(ot['obstacle_num'])]
            self.indep_z_list += [ cp.Variable((1, self.T), nonneg=True, name='z_'+ str(ot['edge_num']) + '_' + str(index)) for index in range(ot['obstacle_num'])]
    
    def state_parameter_define(self):
        
        self.para_ref_s = cp.Parameter((3, self.T+1), name='para_ref_state')
        self.para_ref_speed = cp.Parameter(name='para_ref_state')
        self.para_iot_dis = cp.Parameter((3, self.T+1), name='para_iot_dis')
        self.para_ref_s.value = np.zeros((3, self.T+1))

        self.ref_state = cp.Parameter((3, self.T+1), name='ref_state')
        self.ref_state.value = np.zeros((3, self.T+1))

        self.ref_state_mid = cp.Parameter((3, self.T+1), name='ref_state')
        self.ref_state_mid.value = np.zeros((3, self.T+1))

        self.para_s = cp.Parameter((3, self.T+1), name='para_state')
        self.para_u = cp.Parameter((2, self.T), name='para_vel')
        self.para_dis = cp.Parameter((1, self.T), nonneg=True, value=np.ones((1, self.T)), name='para_dis')

        self.para_rot_list = [cp.Parameter((2, 2), name='para_rot_'+str(t)) for t in range(self.T)]
        self.para_drot_list = [cp.Parameter((2, 2), name='para_drot_'+str(t)) for t in range(self.T)]
        self.para_drot_phi_list = [cp.Parameter((2, 2), name='para_drot_phi_'+str(t)) for t in range(self.T)]

        self.para_A_list = [ cp.Parameter((3, 3), name='para_A_'+str(t)) for t in range(self.T)]
        self.para_B_list = [ cp.Parameter((3, 2), name='para_B_'+str(t)) for t in range(self.T)]
        self.para_C_list = [ cp.Parameter((3, 1), name='para_C_'+str(t)) for t in range(self.T)]

    def dual_parameter_define(self, obstacle_template_list):
        # define the parameters related to obstacles
        self.para_lam_list = []
        self.para_mu_list = []
        self.para_z_list = []
        self.para_xi_list = []
        self.para_zeta_list = []

        for ot in obstacle_template_list:
            for index in range(ot['obstacle_num']):
                oen = ot['edge_num'] # obstacle edge number
                ren = self.car_tuple.G.shape[0]  # robot edge number

                self.para_lam_list += [ cp.Parameter((oen, self.T+1), value=0.1*np.ones((oen, self.T+1)), name='para_lam_'+ str(oen) + '_'  + str(index)) ]
                self.para_mu_list += [ cp.Parameter((ren, self.T+1), value=np.ones((ren, self.T+1)), name='para_mu_'+ str(oen) + '_'  + str(index)) ]
                self.para_z_list += [ cp.Parameter((1, self.T), nonneg=True, value=0.01*np.ones((1, self.T)), name='para_z_'+ str(oen) + '_'  + str(index))]
                self.para_xi_list += [ cp.Parameter((self.T+1, 2), value=np.zeros((self.T+1, 2)), name='para_xi_'+ str(oen) + '_'  + str(index))]
                self.para_zeta_list += [ cp.Parameter((1, self.T), value = np.zeros((1, self.T)), name='para_zeta_'+ str(oen) + '_' + str(index))]

    def obstacle_parameter_define(self, obstacle_template_list):

        self.para_obstacle_list = []

        for ot in obstacle_template_list: 
            for index in range(ot['obstacle_num']):                
                oen = ot['edge_num']

                A_list = [ cp.Parameter((oen, 2), value=np.zeros((oen, 2)), name='para_A_t'+ str(t)) for t in range(self.T+1)]
                b_list = [ cp.Parameter((oen, 1), value=np.zeros((oen, 1)), name='para_b_t'+ str(t)) for t in range(self.T+1)]
                para_obstacle={'A': A_list, 'b': b_list, 'cone_type': ot['cone_type'], 'edge_num': oen, 'assign': False}

                self.para_obstacle_list.append(para_obstacle)

    def combine_parameter_define(self, obstacle_template_list):
        self.para_obsA_lam_list = []   # lam.T @ obsA
        self.para_obsb_lam_list = []   # lam.T @ obsb
        self.para_obsA_rot_list = []   # obs.A @ rot
        self.para_obsA_trans_list = []   # obs.A @ trans

        for ot in obstacle_template_list: 
            for index in range(ot['obstacle_num']): 

                oen = ot['edge_num']

                para_obsA_lam = cp.Parameter((self.T+1, 2), value=np.zeros((self.T+1, 2)), name='para_obsA_lam')
                para_obsb_lam = cp.Parameter((self.T+1, 1), value=np.zeros((self.T+1, 1)), name='para_obsb_lam')

                self.para_obsA_lam_list.append(para_obsA_lam)
                self.para_obsb_lam_list.append(para_obsb_lam)

                para_obsA_rot = [ cp.Parameter((oen, 2), value=np.zeros((oen, 2)), name='para_obsA_rot_t'+ str(t)) for t in range(self.T+1) ]
                para_obsA_trans = [ cp.Parameter((oen, 1), value=np.zeros((oen, 1)), name='para_obsA_trans_t'+ str(t)) for t in range(self.T+1) ]

                self.para_obsA_rot_list.append(para_obsA_rot)
                self.para_obsA_trans_list.append(para_obsA_trans)


    def adjust_parameter_define(self, **kwargs):
        # ws: 1
        # wu: 1
        # ro1: 200
        # ro2: 1
        # slack_gain: 8
        # max_sd: 1.0
        # min_sd: 0.1

        # self.para_ws = cp.Parameter(value=1, nonneg=True)
        # self.para_wu = cp.Parameter(value=1, nonneg=True)
        self.para_slack_gain = cp.Parameter(value=kwargs.get('slack_gain', 10), nonneg=True)
        self.para_max_sd = cp.Parameter(value=kwargs.get('max_sd', 1), nonneg=True)
        self.para_min_sd = cp.Parameter(value=kwargs.get('min_sd', 0.1), nonneg=True)

        self.itera_0 = cp.Parameter((10,4),value=np.zeros((10,4)),nonneg=True)
        self.wrm = cp.Parameter(value=0)

        

    # endregion

    # region: construct the problem
    def construct_problem(self, **kwargs):
        self.prob_su = self.construct_su_prob(**kwargs)
        self.prob_LamMuZ_list = self.construct_LamMuZ_prob(**kwargs)


    def construct_mp_problem(self, process_num, **kwargs):
        self.prob_su = self.construct_su_prob(**kwargs)
        pool = Pool(processes=process_num, initializer=self.init_prob_LamMuZ, initargs=(kwargs, )) 
        return pool
    
    def construct_su_prob(self, **kwargs):
        
        ws = kwargs.get('ws', 1)
        wu = kwargs.get('wu', 1)

        ro1 = kwargs.get('ro1', 200)
        ro2 = kwargs.get('ro2', 1)
        

        wid = 0
        wm = 0.01
        # wm = 0.1
        
        nav_cost, nav_constraints = self.nav_cost_cons(ws, wu)
        su_cost, su_constraints = self.update_su_cost_cons(self.para_slack_gain, ro1, ro2)

        # prob_su = cp.Problem(cp.Minimize(nav_cost + su_cost + wid*I_dis_cost), su_constraints+nav_constraints) 
        prob_su = cp.Problem(cp.Minimize(nav_cost + su_cost), su_constraints + nav_constraints)
        # prob_su = cp.Problem(cp.Minimize(nav_cost + su_cost + wid*mm_cost), su_constraints+nav_constraints) 
        # prob_su = cp.Problem(cp.Minimize(nav_cost + su_cost + cp.sqrt(self.wrm ** 2)*I_dis_cost), su_constraints+nav_constraints) 

        # self.prob_su1 = cp.Problem(cp.Minimize(nav_cost + su_cost - wm * mm_cost), su_constraints+nav_constraints) 
        self.prob_su1 = prob_su
        assert prob_su.is_dcp()
        assert self.prob_su1.is_dcp()

        return prob_su

    def construct_LamMuZ_prob(self, **kwargs):
        
        ro1 = kwargs.get('ro1', 200)
        ro2 = kwargs.get('ro2', 1) 
        prob_list = []

        for obs_index in range(self.obstacle_template_num):

            indep_lam = self.indep_lam_list[obs_index]
            indep_mu = self.indep_mu_list[obs_index]
            indep_z = self.indep_z_list[obs_index]

            para_xi = self.para_xi_list[obs_index]
            para_zeta = self.para_zeta_list[obs_index]

            para_obs = self.para_obstacle_list[obs_index]

            para_obsA_rot = self.para_obsA_rot_list[obs_index]
            para_obsA_trans = self.para_obsA_trans_list[obs_index]

            cost, constraints = self.LamMuZ_cost_cons(indep_lam, indep_mu, indep_z, self.para_s, self.para_rot_list, para_xi, self.para_dis, para_zeta, para_obs, para_obsA_rot, para_obsA_trans, self.T, ro1, ro2)
            
            prob = cp.Problem(cp.Minimize(cost), constraints)

            assert prob.is_dcp(dpp=True)
            
            prob_list.append(prob)

        return prob_list

    def init_prob_LamMuZ(self, kwargs):
        global prob_LamMuZ_list, para_xi_list, para_zeta_list, para_s, para_rot_list, para_dis, para_obsA_rot_list, para_obsA_trans_list, para_obstacle_list

        para_xi_list = self.para_xi_list
        para_zeta_list = self.para_zeta_list
        para_s = self.para_s
        para_rot_list = self.para_rot_list
        para_dis = self.para_dis 

        para_obsA_rot_list = self.para_obsA_rot_list
        para_obsA_trans_list = self.para_obsA_trans_list
        para_obstacle_list = self.para_obstacle_list

        prob_LamMuZ_list = self.construct_LamMuZ_prob_parallel(para_xi_list, para_zeta_list, para_s, para_rot_list, para_dis, para_obstacle_list, para_obsA_rot_list, para_obsA_trans_list, **kwargs)

    def construct_LamMuZ_prob_parallel(self, para_xi_list, para_zeta_list, para_s, para_rot_list, para_dis, para_obstacle_list, para_obsA_rot_list, para_obsA_trans_list, **kwargs):

        ro1 = kwargs.get('ro1', 200)
        ro2 = kwargs.get('ro2', 1) 

        prob_list = []

        for obs_index in range(self.obstacle_template_num):

            indep_lam = self.indep_lam_list[obs_index]
            indep_mu = self.indep_mu_list[obs_index]
            indep_z = self.indep_z_list[obs_index]

            para_xi = para_xi_list[obs_index]
            para_zeta = para_zeta_list[obs_index]

            para_obs = para_obstacle_list[obs_index]
            para_obsA_rot = para_obsA_rot_list[obs_index]
            para_obsA_trans = para_obsA_trans_list[obs_index]

            cost, constraints = self.LamMuZ_cost_cons(indep_lam, indep_mu, indep_z, para_s, para_rot_list, para_xi, para_dis, para_zeta, para_obs, para_obsA_rot, para_obsA_trans, self.T, ro1, ro2)
            
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob_list.append(prob)

        return prob_list


    def nav_cost_cons(self, ws=1, wu=1):
        
        # path tracking objective cost constraints
        # indep_s: cp.Variable((3, self.receding+1), name='state')
        # indep_u: cp.Variable((2, self.receding), name='vel')
        # para_ref_s: cp.Parameter((3, self.T+1), name='para_ref_state')

        cost = 0
        constraints = []
        cost += self.C0_cost(self.para_ref_s, self.para_ref_speed, self.indep_s, self.indep_u[0, :], ws, wu)
        #iot_dis
        # cost += self.dis_cost(self.para_iot_dis,self.iot_dis_var)
        constraints += self.dynamics_constraint(self.indep_s, self.indep_u, self.T)
        #动力学约束
        constraints += self.bound_su_constraints(self.indep_s, self.indep_u, self.para_s, self.max_speed, self.acce_bound)
        #速度与加速度约束
        self.cm_constr = constraints
        self.cm_cost1 = cost
        

        return cost, constraints
    
    def update_su_cost_cons(self, slack_gain, ro1=200, ro2=1):
        cost = 0
        constraints = []

        if self.obstacle_template_num == 0:
            return cost, constraints

        cost += self.C1_cost(self.indep_dis, slack_gain)

        Im_su_list = []
        Hm_su_list = []
        
        for obs_index, para_obs in enumerate(self.para_obstacle_list):  
            
            para_xi = self.para_xi_list[obs_index]

            para_lam = self.para_lam_list[obs_index]
            para_mu = self.para_mu_list[obs_index]
            para_z = self.para_z_list[obs_index]
            para_zeta = self.para_zeta_list[obs_index]

            para_obsA_lam = self.para_obsA_lam_list[obs_index]
            para_obsb_lam = self.para_obsb_lam_list[obs_index]

            Imsu = self.Im_su(self.indep_s, self.indep_dis, para_lam, para_mu, para_z, para_zeta, para_obs, para_obsA_lam, para_obsb_lam)
            Hmsu = self.Hm_su(self.indep_rot_list, para_mu, para_lam, para_xi, para_obs, self.T, para_obsA_lam)
            
            Im_su_list.append(Imsu)
            Hm_su_list.append(Hmsu)

        rot_diff_list = []
        for t in range(self.T):

            indep_phi = self.indep_s[2, t+1:t+2]
            indep_rot_t = self.indep_rot_list[t]

            rot_diff_list.append(self.para_rot_list[t] - self.para_drot_phi_list[t] + self.para_drot_list[t] * indep_phi - indep_rot_t)

        rot_diff_array = cp.vstack(rot_diff_list)
        Im_su_array = cp.vstack(Im_su_list)
        Hm_su_array = cp.vstack(Hm_su_list)

        constraints += [cp.constraints.zero.Zero(rot_diff_array)]
        
        cost += 0.5*ro1 * cp.sum_squares(cp.neg(Im_su_array))
        # constraints += [Im_su_array >= 0]
        cost += 0.5*ro2 * cp.sum_squares(Hm_su_array)

        constraints += self.bound_dis_constraints(self.indep_dis)

        return cost, constraints

    def LamMuZ_cost_cons(self, indep_lam, indep_mu, indep_z, para_s, para_rot_list, para_xi, para_dis, para_zeta, para_obs, para_obsA_rot, para_obsA_trans, receding, ro1, ro2):

        cost = 0
        constraints = []

        Hm_array = self.Hm_LamMu(indep_lam, indep_mu, para_rot_list, para_xi, para_obs, receding, para_obsA_rot)
        Im_array = self.Im_LamMu(indep_lam, indep_mu, indep_z, para_s, para_dis, para_zeta, para_obs, para_obsA_trans)

        cost += 0.5*ro1 * cp.sum_squares(cp.neg(Im_array))
        # constraints += [ Im_array >= 0 ]
        cost += 0.5*ro2 * cp.sum_squares(Hm_array)

        temp_list = []
        for t in range(self.T):
            para_obsAt = para_obs['A'][t+1]
            indep_lam_t = indep_lam[:, t+1:t+2]
            temp_list.append( cp.norm(para_obsAt.T @ indep_lam_t) )
        
        temp = cp.max(cp.vstack(temp_list))

        constraints += [ temp <= 1 ]
        # constraints += [ cp.norm(para_obs.A.T @ indep_lam, axis=0) <= 1 ]
        constraints += [ self.cone_cp_array(-indep_lam, para_obs['cone_type']) ]
        constraints += [ self.cone_cp_array(-indep_mu, self.car_tuple.cone_type) ]

        return cost, constraints
    # endregion

    # region: assign value for parameters
    def assign_adjust_parameter(self, **kwargs):
        # self.para_ws.value = kwargs.get('ws', 1)
        # self.para_wu.value = kwargs.get('wu', 1) 
        self.para_slack_gain = kwargs.get('slack_gain', 8)
        self.para_max_sd.value = kwargs.get('max_sd',1)
        self.para_min_sd.value = kwargs.get('min_sd',0.1)
    

    def assign_state_parameter(self, nom_s, nom_u, nom_dis):

        self.para_s.value = nom_s
        self.para_u.value = nom_u
        self.para_dis.value = nom_dis
        
        for t in range(self.T):
            nom_st = nom_s[:, t:t+1]
            state_ugv = nom_s[:,0:1]
            state_ugv_x = nom_s[0:,0:1]
            state_ugv_y = nom_s[1:,0:1]
            nom_ut = nom_u[:, t:t+1]
            
            A, B, C = self.linear_ackermann_model(nom_st, nom_ut, self.dt, self.L)

            self.para_A_list[t].value = A
            self.para_B_list[t].value = B
            self.para_C_list[t].value = C

            nom_phi = nom_st[2, 0]
            self.para_rot_list[t].value = np.array([[cos(nom_phi), -sin(nom_phi)],  [sin(nom_phi), cos(nom_phi)]])
            self.para_drot_list[t].value = np.array( [[-sin(nom_phi), -cos(nom_phi)],  [cos(nom_phi), -sin(nom_phi)]] )
            self.para_drot_phi_list[t].value = nom_phi * np.array( [[-sin(nom_phi), -cos(nom_phi)],  [cos(nom_phi), -sin(nom_phi)]] )

    def assign_state_parameter_parallel(self, input):

        nom_s, nom_dis = input

        para_s.value = nom_s
        para_dis.value = nom_dis
        
        for t in range(self.T):
            nom_st = nom_s[:, t:t+1]
            
            nom_phi = nom_st[2, 0]
            para_rot_list[t].value = np.array([[cos(nom_phi), -sin(nom_phi)],  [sin(nom_phi), cos(nom_phi)]])

    def assign_dual_parameter(self, LamMuZ_list):

        for index, LamMuZ in enumerate(LamMuZ_list):
            self.para_lam_list[index].value = LamMuZ[0]
            self.para_mu_list[index].value = LamMuZ[1]
            self.para_z_list[index].value = LamMuZ[2]


    def assign_obstacle_parameter(self, obstacle_list):
        
        # self.obstacle_template_list
        self.obstacle_num = len(obstacle_list)

        for obs in obstacle_list:
            for para_obs in self.para_obstacle_list:

                para_obs_edge_num = para_obs['edge_num']

                if isinstance(obs.A, list):
                    obs_edge_num = obs.A[0].shape[0]
                    if obs_edge_num == para_obs_edge_num and para_obs['assign'] is False:
                        for t in range(len(para_obs['A'])):
                            para_obs['A'][t].value = obs.A[t]
                            para_obs['b'][t].value = obs.b[t]

                        para_obs['assign'] = True
                        break
                           
                else:
                    obs_edge_num = obs.A.shape[0]

                    if obs_edge_num == para_obs_edge_num and para_obs['assign'] is False:
                        for t in range(len(para_obs['A'])):
                            para_obs['A'][t].value = obs.A
                            para_obs['b'][t].value = obs.b

                        para_obs['assign'] = True
                        break
                        
        for para_obs in self.para_obstacle_list:
            para_obs['assign'] = False


    def assign_combine_parameter_lamobs(self):
        
        for n in range(self.obstacle_template_num):

            para_lam_value = self.para_lam_list[n].value
            para_obs = self.para_obstacle_list[n]

            for t in range(self.T):
                lam = para_lam_value[:, t+1:t+2]
                obsA = para_obs['A'][t+1].value
                obsb = para_obs['b'][t+1].value

                self.para_obsA_lam_list[n].value[t+1, :] = lam.T @ obsA
                self.para_obsb_lam_list[n].value[t+1, :] = lam.T @ obsb
                    
    def assign_combine_parameter_stateobs(self):
        
        # self.para_obsA_lam_list = []   # lam.T @ obsA
        # self.para_obsb_lam_list = []   # lam.T @ obsb
        # self.para_obsA_rot_list = []   # obs.A @ rot
        # self.para_obsA_trans_list = []   # obs.A @ trans

        for n in range(self.obstacle_template_num):

            para_obs = self.para_obstacle_list[n]

            for t in range(self.T):
                obsA = para_obs['A'][t+1].value

                rot = self.para_rot_list[t].value
                trans = self.para_s.value[0:2, t+1:t+2]
                
                self.para_obsA_rot_list[n][t+1].value = obsA @ rot
                self.para_obsA_trans_list[n][t+1].value = obsA @ trans

    # endregion
    
    # region: solve the problem
    def iterative_solve(self, nom_s, nom_u, ref_states, ref_speed, obstacle_list, **kwargs):
        global iteration_n

        # obstacle_list: a list of obstacle instance
        #   obstacle: (A, b, cone_type)

        # start_time = time.time()
        
        self.para_ref_s.value = np.hstack(ref_states)[0:3, :]
        self.para_ref_speed.value = ref_speed
        # random.shuffle(obstacle_list)

        self.assign_state_parameter(nom_s, nom_u, self.para_dis.value)
        self.assign_obstacle_parameter(obstacle_list)
        
        iteration_time = time.time()

        # print(self.iter_num)
        for i in range(self.iter_num):

            start_time = time.time()
            opt_state_array, opt_velocity_array, resi_dual, resi_pri = self.rda_solver()
            print('iteration ' + str(i) + ' time: ', time.time()-start_time)
            
            if resi_dual < self.iter_threshold and resi_pri < self.iter_threshold:
                print('iteration early stop: '+ str(i))
                break

        print('-----------------------------------------------')
        print('iteration time:', time.time() - iteration_time)
      
        print('==============================================')
        
        if not os.path.exists('results'):
            os.mkdir('results')
            print('mkdir log finished.')

        self.save_path_point(opt_state_array[0][0],opt_state_array[1][0],opt_state_array[2][0])
        self.save_path_dir(iteration_n, opt_velocity_array[0][0], opt_velocity_array[1][0])

        # # info for debug
        opt_state_list = [state[:, np.newaxis] for state in opt_state_array.T ]
        info = {'ref_traj_list': ref_states, 'opt_state_list': opt_state_list}
        info['iteration_time'] = time.time() - start_time
        info['resi_dual'] = resi_dual
        info['resi_pri'] = resi_pri    

        iteration_n += 1
        
        return opt_velocity_array, info 

    def rda_solver(self):
        
        resi_dual, resi_pri = 0, 0
        
        # start_time = time.time()
        nom_s, nom_u, nom_dis = self.su_prob_solve()
        # print('- su problem solve:', time.time() - start_time)
        
        self.assign_state_parameter(nom_s, nom_u, nom_dis)
        # start_time = time.time()
        self.assign_combine_parameter_stateobs()
        # print('- other1:', time.time() - start_time)

        if self.obstacle_num != 0:
        # if self.obstacle_template_num != 0:
            # start_time = time.time()
            LamMuZ_list, resi_dual = self.LamMuZ_prob_solve()
            # print('- LamMu problem solve:', time.time() - start_time)

            self.assign_dual_parameter(LamMuZ_list)
            self.assign_combine_parameter_lamobs()

            resi_pri = self.update_xi()
            self.update_zeta()
            
        return nom_s, nom_u, resi_dual, resi_pri

    def update_zeta(self):

        for obs_index, para_obs in enumerate(self.para_obstacle_list):

            Im_list = []
            zeta = self.para_zeta_list[obs_index].value
            z = self.para_z_list[obs_index].value

            for t in range(self.T):

                lam_t = self.para_lam_list[obs_index].value[:, t+1:t+2]
                mu_t = self.para_mu_list[obs_index].value[:, t+1:t+2]
                z_t = self.para_z_list[obs_index].value[:, t:t+1]
                zeta_t = self.para_zeta_list[obs_index].value[:, t:t+1]

                trans_t = self.para_s.value[0:2, t+1:t+2]

                para_obsAt = para_obs['A'][t+1].value
                para_obsbt = para_obs['b'][t+1].value

                Im = lam_t.T @ para_obsAt @ trans_t - lam_t.T @ para_obsbt - mu_t.T @ self.car_tuple.h

                Im_list.append(Im)

            Im_array = np.hstack(Im_list)

            # temp= zeta + (Im_array - self.para_dis.value - z)  
            self.para_zeta_list[obs_index].value = zeta + (Im_array - self.para_dis.value - z)    
            
    def update_xi(self): 

        hm_list = []

        for obs_index, obs in enumerate(self.para_obstacle_list):
            for t in range(self.T):

                lam_t = self.para_lam_list[obs_index].value[:, t+1:t+2]
                mu_t = self.para_mu_list[obs_index].value[:, t+1:t+2]
                rot_t = self.para_rot_list[t].value
                xi_t = self.para_xi_list[obs_index].value[t+1:t+2, :]
                
                obsAt = obs['A'][t+1].value

                Hmt = mu_t.T @ self.car_tuple.G + lam_t.T @ obsAt @ rot_t
                self.para_xi_list[obs_index].value[t+1:t+2, :] = Hmt + xi_t    

                hm_list.append(Hmt)

        hm_array = np.vstack(hm_list)
        resi_pri = np.linalg.norm(hm_array)

        return resi_pri
    
    def su_prob_solve(self):

        self.ref_state_mid.value = self.indep_s.value
        # self.prob_su.solve()
        self.prob_su.solve(solver=cp.SCS)
        #MM_begin
        self.ref_state.value = self.indep_s.value
        self.indep_s.value = self.ref_state_mid.value
        self.calc_itera_0()
        # # self.prob_su1.solve(verbose=True)
        for i in range (nnn_iot):
            iotpt_value[i] = 20
        self.prob_su1.solve(solver=cp.SCS)
            
        
        if self.prob_su.status == cp.OPTIMAL:
            return self.indep_s.value, self.indep_u.value, self.indep_dis.value
        else:
            print('No update of state and control vector')
            return self.para_s.value, self.para_u.value, self.para_dis.value

    def LamMuZ_prob_solve(self):
        
        input_args = []
        if self.process_num > 1:
            for obs_index in range(self.obstacle_template_num):

                nom_s = self.para_s.value
                nom_dis = self.para_dis.value
                nom_xi = self.para_xi_list[obs_index].value
                nom_zeta = self.para_zeta_list[obs_index].value
                receding = self.T
                nom_lam = self.para_lam_list[obs_index].value
                nom_mu = self.para_mu_list[obs_index].value
                nom_z = self.para_z_list[obs_index].value
                
                nom_obs_A = [o.value for o in self.para_obstacle_list[obs_index]['A']]    
                nom_obs_b = [o.value for o in self.para_obstacle_list[obs_index]['b']]   
                nom_obsA_rot = [p.value for p in self.para_obsA_rot_list[obs_index]]
                nom_obsA_trans = [p.value for p in self.para_obsA_trans_list[obs_index]] 

                input_args.append((obs_index, nom_s, nom_dis, nom_xi, receding, nom_lam, nom_mu, nom_z, nom_zeta, nom_obs_A, nom_obs_b, nom_obsA_rot, nom_obsA_trans))
            
            LamMuZ_list = pool.map(RDA_solver.solve_parallel, input_args)

        else:
            for obs_index in range(self.obstacle_template_num):
                prob = self.prob_LamMuZ_list[obs_index]
                input_args.append((prob, obs_index))
            
            LamMuZ_list = list(map(self.solve_direct, input_args))
        
        # update
        if len(LamMuZ_list) != 0:
            resi_dual_list = ([LamMu[3] for LamMu in LamMuZ_list])
            resi_dual = sum(resi_dual_list) / len(resi_dual_list)
        else:
            resi_dual = 0

        return LamMuZ_list, resi_dual

    def solve_parallel(input):
        
        obs_index, nom_s, nom_dis, nom_xi, receding, nom_lam, nom_mu, nom_z, nom_zeta, nom_obs_A, nom_obs_b, nom_obsA_rot, nom_obsA_trans = input
        
        prob = prob_LamMuZ_list[obs_index]

        # update parameter
        para_s.value = nom_s
        para_dis.value = nom_dis
        para_xi_list[obs_index].value = nom_xi
        para_zeta_list[obs_index].value = nom_zeta

        for t in range(receding):
            nom_st = nom_s[:, t:t+1]
            nom_phi = nom_st[2, 0]
            para_rot_list[t].value = np.array([[cos(nom_phi), -sin(nom_phi)],  [sin(nom_phi), cos(nom_phi)]])
            
            para_obsA_rot_list[obs_index][t+1].value = nom_obsA_rot[t+1]
            para_obsA_trans_list[obs_index][t+1].value = nom_obsA_trans[t+1]

            para_obstacle_list[obs_index]['A'][t+1].value = nom_obs_A[t+1]
            para_obstacle_list[obs_index]['b'][t+1].value = nom_obs_b[t+1]
        
        prob.solve(solver=cp.ECOS)

        for variable in prob.variables():
            if 'lam_' in variable.name():
                indep_lam_value = variable.value
            elif 'mu_' in variable.name():
                indep_mu_value = variable.value
            elif 'z_' in variable.name():
                indep_z_value = variable.value
                
        if prob.status == cp.OPTIMAL:

            lam_diff = np.linalg.norm(indep_lam_value - nom_lam)
            mu_diff = np.linalg.norm(indep_mu_value - nom_mu)
            
            z_diff = np.linalg.norm(indep_z_value - nom_z)
            residual = lam_diff**2 + mu_diff**2 + z_diff**2

            return indep_lam_value, indep_mu_value, indep_z_value, residual

        else:
            print('Update Lam Mu Fail')
            return nom_lam, nom_mu, nom_z, inf

    def solve_direct(self, input):
        
        prob, obs_index = input
        prob.solve(solver=cp.ECOS)

        indep_lam = self.indep_lam_list[obs_index]
        indep_mu = self.indep_mu_list[obs_index]
        indep_z = self.indep_z_list[obs_index]

        para_lam = self.para_lam_list[obs_index]
        para_mu = self.para_mu_list[obs_index]
        para_z = self.para_z_list[obs_index]

        if prob.status == cp.OPTIMAL:

            lam_diff = np.linalg.norm(indep_lam.value - para_lam.value)
            mu_diff = np.linalg.norm(indep_mu.value - para_mu.value)
            z_diff = np.linalg.norm(indep_z.value - para_z.value)
            residual = lam_diff**2 + mu_diff**2 + z_diff**2

            return indep_lam.value, indep_mu.value, indep_z.value, residual
        else:
            print('Update Lam Mu Fail')
            return para_lam.value, para_mu.value, para_z.value, inf
        
    # endregion

    # region: formula， Hm, Im
    def Im_su(self, state, distance, para_lam, para_mu, para_z, para_zeta, para_obs, para_obsA_lam, para_obsb_lam):
        
        Im_list = []

        for t in range(self.T):
            para_lam_t = para_lam[:, t+1:t+2]
            indep_trans_t = state[0:2, t+1:t+2]
            para_mu_t = para_mu[:, t+1:t+2]

            para_obsAt = para_obs['A'][t+1]
            para_obsbt = para_obs['b'][t+1]

            para_obsA_lam_t = para_obsA_lam[t+1:t+2, :]
            para_obsb_lam_t = para_obsb_lam[t+1:t+2, :]
             
            Im = para_obsA_lam_t @ indep_trans_t - para_obsb_lam_t - para_mu_t.T @ self.car_tuple.h
            Im_list.append(Im)

        Im_array = cp.hstack(Im_list)

        return Im_array[0, :] - distance[0, :] - para_z[0, :] + para_zeta[0, :]

    def Hm_su(self, rot, para_mu, para_lam, para_xi, para_obs, receding, para_obsA_lam):
        
        Hm_list = []

        for t in range(receding):
    
            lam_t = para_lam[:, t+1:t+2]
            mu_t = para_mu[:, t+1:t+2]
            para_xi_t = para_xi[t+1:t+2, :]
            indep_rot_t = rot[t]

            para_obsAt = para_obs['A'][t+1]

            para_obsA_lam_t = para_obsA_lam[t+1:t+2, :]

            Hmt = mu_t.T @ self.car_tuple.G + para_obsA_lam_t @ indep_rot_t + para_xi_t

            Hm_list.append(Hmt)

        return cp.vstack(Hm_list)

    def Hm_LamMu(self, indep_lam, indep_mu, para_rot_list, para_xi, para_obs, receding, para_obsA_rot):

        Hm_list = []
        for t in range(receding):
            indep_lam_t = indep_lam[:, t+1:t+2]
            indep_mu_t = indep_mu[:, t+1:t+2]
            
            para_rot_t = para_rot_list[t]
            para_xi_t = para_xi[t+1:t+2, :]

            para_obsA_rot_t = para_obsA_rot[t+1]
        
            Hmt = indep_mu_t.T @ self.car_tuple.G + indep_lam_t.T @ para_obsA_rot_t + para_xi_t
            Hm_list.append(Hmt)

        return cp.vstack(Hm_list)

    def Im_LamMu(self, indep_lam, indep_mu, indep_z, para_s, para_dis, para_zeta, para_obs, para_obsA_trans):

        # Im_array = cp.diag( indep_lam.T @ obs.A @ para_s[0:2] - indep_lam.T @ obs.b - indep_mu.T @ self.car_tuple.h ) 
        Im_list = []

        for t in range(self.T):
            indep_lam_t = indep_lam[:, t+1:t+2]
            indep_mu_t = indep_mu[:, t+1:t+2]
            para_obsbt = para_obs['b'][t+1]
            para_obsA_trans_t = para_obsA_trans[t+1]

            Im = indep_lam_t.T @ para_obsA_trans_t - indep_lam_t.T @ para_obsbt - indep_mu_t.T @ self.car_tuple.h
            Im_list.append(Im)

        Im_array = cp.hstack(Im_list)

        Im_lammu = Im_array[0, :] - para_dis[0, :] - indep_z[0, :] + para_zeta[0, :]

        return Im_lammu

    def dynamics_constraint(self, state, control_u, receding):

        temp_s1_list = []

        for t in range(receding):
            indep_st = state[:, t:t+1]
            indep_ut = control_u[:, t:t+1]

            ## dynamic constraints
            A = self.para_A_list[t]
            B = self.para_B_list[t]
            C = self.para_C_list[t]
            
            temp_s1_list.append(A @ indep_st + B @ indep_ut + C)
        
        constraints = [ state[:, 1:] == cp.hstack(temp_s1_list) ]

        return constraints
        
    def bound_su_constraints(self, state, control_u, para_s, max_speed, acce_bound):

        constraints = []

        constraints += [ cp.abs(control_u[:, 1:] - control_u[:, :-1] ) <= acce_bound ]  # constraints on speed accelerate
        constraints += [ cp.abs(control_u) <= max_speed]
        constraints += [ state[:, 0:1] == para_s[:, 0:1] ]

        return constraints

    def bound_dis_constraints(self, indep_dis):

        constraints = []

        constraints += [ cp.max(indep_dis) <= self.para_max_sd ] 
        constraints += [ cp.min(indep_dis) >= self.para_min_sd ]

        return constraints
    
    def linear_ackermann_model(self, nom_state, nom_u, dt, L):
        
        phi = nom_state[2, 0]
        v = nom_u[0, 0]
        psi = nom_u[1, 0]

        A = np.array([ [1, 0, -v * dt * sin(phi)], [0, 1, v * dt * cos(phi)], [0, 0, 1] ])

        B = np.array([ [cos(phi)*dt, 0], [sin(phi)*dt, 0], 
                        [ tan(psi)*dt / L, v*dt/(L * (cos(psi))**2 ) ] ])

        C = np.array([ [ phi*v*sin(phi)*dt ], [ -phi*v*cos(phi)*dt ], 
                        [ -psi * v*dt / ( L * (cos(psi))**2) ]])

        return A, B, C

    def C0_cost(self, ref_s, ref_speed, state, speed, ws, wu):
        
        #distance_sum_s
        # num_points = ref_s.value.shape[1]
        # iot_distances = np.zeros(num_points)

        # iot_distances = []
        # for i in range(num_points):
        #     x = state[0, i] 
        #     y = state[1, i]  
        #     iot_distances.append(self.Iot_distance(x, y) )           
        # sum_iot_cost = cp.sum(iot_distances)
       
        #distance_sum_e
        # diff_s = (state - ref_s)
        diff_s = (state[0:2]-ref_s[0:2])
        diff_u = (speed - ref_speed)
        # 0.3832
        # return ws * cp.sum_squares(diff_s) + wu*cp.sum_squares(diff_u) + 0.15*sum_iot_cost
        return ws * cp.sum_squares(diff_s) + wu*cp.sum_squares(diff_u)
    #CM_begin
    def CM_cost(self,ref_state):

        # num_points = ref_state.value.shape[1]
        num_points = 10
        MM_iot_sum = []
        for i in range(num_points-1):
            MM_iot_sum.append(self.MM_Iot(self.indep_s[0,i],self.indep_s[1,i],i) )      #（基础位置（x，y）（变量））     
        sum_MM = cp.sum(MM_iot_sum)

        return sum_MM
        
    #CM_end

    def dis_cost(self, iot_dis,iot_var):
        diff_iot = ( iot_dis[2, :]-iot_var[2,:] ) 
        sum_iot_dis = cp.sum_squares(diff_iot)
        # convex_sum = cp.pos(sum_iot_dis)
        return 1*sum_iot_dis


    def C1_cost(self, indep_dis, slack_gain):
        return ( -slack_gain * cp.sum(indep_dis) )

    def cone_cp_array(self, array, cone='Rpositive'):
        # cone for cvxpy: R_positive, norm2
        if cone == 'Rpositive':
            return cp.constraints.nonpos.NonPos(array)
        elif cone == 'norm2':
            return cp.constraints.nonpos.NonPos( cp.norm(array[0:-1], axis = 0) - array[-1]  )

    def Iot_distance(self, x, y):
        total_distance = 0
        for i in range(nnn_iot):
            distance = cp.norm(cp.vstack([x - iot_x_value[i], y - iot_y_value[i]]))
            total_distance += distance
               

        return total_distance

    

    def MM_Iot(self, x, y, k):
        t1 = 0.1  # time step
        sigma1 = 0.01
        alpha1 = 2
        
        
        # 归一化参数
        scale_factor = 1e-3  # 通过该缩放因子调整距离和功率相关数值的大小

        
        i1 = 0
        total_com1 = 0
        
        
        for i in range(nnn_iot):
            iot_x= iot_x_value[i]
            iot_y= iot_y_value[i]
            iot_B_k= iotbk_value[i]
                
            # 计算设备间的距离并进行缩放
            itera1 = cp.norm(cp.vstack([x - iot_x, y - iot_y]))**2 * scale_factor  # 对距离进行缩放
                
            # 缩放参数计算，保证数值稳定性
            scaled_power = (iotpt_value[i] / sigma1**2) * scale_factor
                
            scaled_itera_0 = cp.power(self.itera_0[k, i1], -alpha1 / 2) * scale_factor
                
            # 调整后的通信公式，避免过大的数值波动
            Iot_com1 = t1 * iot_B_k * (
                    cp.log(1 + scaled_power * (
                        2 * scaled_itera_0 - 
                        cp.power(self.itera_0[k, i1], -alpha1) * cp.power(itera1, alpha1 / 2)
                    )) / np.log(2)
                )

            i1 += 1
            total_com1 += Iot_com1
        
        return total_com1


    def fairness_Iot(self, x, y):
        alpha1 = 2
        t = 0.1
        N0 = 1
        iot_B_k_const = 1
        # 读取IoT设备数量
        
        # 定义p_k作为优化变量
        P_k = cp.Variable(nnn_iot, nonneg=True)  # 每个P_k > 0
        min_com1 = cp.Variable()  # 用于存储min_com1的优化变量
        # 限制条件：p_k的和要小于n * 20, average is 20mW

        constraints = [cp.sum(P_k) <= nnn_iot * 20]
        
        com_list = []
        itera1_list = []
        com_list_debug = []

        # 遍历每个IoT设备并设置通信公式
        for i in range(nnn_iot):

            iot_x= iot_x_value[i]
            iot_y= iot_y_value[i]
            
            # 计算设备间的距离
            # itera1 = cp.norm(cp.vstack([x - iot_x, y - iot_y])) 
            itera1 = np.linalg.norm([x - iot_x, y - iot_y])
            
            # 使用辅助变量简化公式
            # scaled_itera_0 = 50*cp.power(itera1, -alpha1 )
            scaled_itera_0 = 50* itera1**(-alpha1)
             
            # 用新的变量表示通信速率，简化计算
            Iot_com1 = scaled_itera_0*P_k[i]
            #debug辅助变量，查看优化前后的量

            Iot_com2 = t*iot_B_k_const*(cp.log(1+Iot_com1)/math.log(2))

            Iotdebug = scaled_itera_0
            com_list_debug.append(Iotdebug)
            # 添加到通信速率列表
            # com_list.append(Iot_com1)
            com_list.append(Iot_com2)
            
            itera1_list.append(itera1)
        
        # 将com_list转换为cvxpy可以处理的形式
       

        # constraints.append(min_com1 <= cp.min(cp.vstack(com_list)))
        
        # 优化目标：最大化 min_com1
        objective1 = cp.Maximize(cp.min(cp.vstack(com_list)))
        # objective1 = cp.Maximize(cp.min(com_list))
        objective2 = cp.Maximize(cp.sum(com_list)/3)
        zeta = 1
        objective = zeta * objective1 + (1-zeta) * objective2
        # objective = objective2
        
        # if x <= 5 and x >= 4:
        #     com_assistance = [com_list[0] >= 0.7 ]
        #     constraints += com_assistance
       

        # 定义并求解优化问题
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # debug_youhua = [0]*nnn_iot
        # debug_yuanlai = [0]*nnn_iot
        # for i in range(nnn_iot):
        #     iotpt_value[i] = P_k[i].value
        #     debug_youhua[i]=com_list_debug[i].value * iotpt_value[i]
        #     debug_yuanlai[i]=com_list_debug[i].value * 20

        # 返回最终的总通信速率和优化后的P_k
            
        # itera1_list[0]
        # [com_list[0].value,com_list[1].value,com_list[2].value], P_k.value
        return [com_list[0].value,com_list[1].value,com_list[2].value], P_k.value


    
    def Iot_distance_cost(self,state):
        num_points = state.value.shape[1]
        iot_distances = np.zeros(num_points)

        iot_distances = []
        for i in range(num_points):
            x = state[0, i] 
            y = state[1, i]  
            iot_distances.append(self.Iot_distance(x, y) )           
        sum_iot_cost = cp.sum(iot_distances)
        return sum_iot_cost
       

    def Iot_com(self,x, y):
        t = 0.1   #time step
        N_0 = 1 
        alpha = 2
        niot = 0
        # P_k = 1000 #0.1
        iot_file_path = "./iot_list.txt" 
        debug_iot = [0]*nnn_iot
        total_com = 0
        with open(iot_file_path, "r") as file:
            for line in file:
                iot_coords = line.strip().split(",")
                iot_x, iot_y,iot_B_k = map(float, iot_coords)
                iot_x_const = cp.Parameter(value=iot_x)
                iot_y_const = cp.Parameter(value=iot_y)
                iot_B_k_const = cp.Parameter(value=iot_B_k) #1
                # iot_x_const = 2.5
                Iot_com = t*iot_B_k_const*(cp.log(1+(cp.norm(cp.vstack([x - iot_x_const, y - iot_y_const])))**(-alpha)*iotpt_value[niot]*50/N_0)/cp.log(2))
                Iot_com_debug = t*iot_B_k_const*(cp.log(1+(cp.norm(cp.vstack([x - iot_x_const, y - iot_y_const])))**(-alpha)*20*50/N_0)/cp.log(2))
                total_com += Iot_com
                debug_iot[niot] = Iot_com_debug
                self.single_iot[niot] = Iot_com

                niot+=1
        
        return total_com.value
    

    def calc_itera_0(self):
        new_itera_0_value = np.array(self.itera_0.value)
        for k in range(10):
            iot_file_path = "./iot_Tjunc.txt" 
            i = 0
            with open(iot_file_path, "r") as file:
                for line in file:
                    iot_coords = line.strip().split(",")
                    iot_x, iot_y,iot_B_k = map(float, iot_coords)
                    # self.itera_0[k,i].value=(cp.norm(cp.vstack([self.indep_s[0,k].value-iot_x, self.indep_s[1,k].value - iot_y])))**2
                    a = np.array([self.ref_state[0,k].value-iot_x, self.ref_state[1,k].value - iot_y])
                    squares = np.square(a)
                    sum_of_squares = np.sum(squares)
                    new_itera_0_value[k,i] = sum_of_squares
                    i += 1
        # new_itera_0 = cp.Parameter((10, 3), value=new_itera_0_value, nonneg=True)
        new_itera_0 = cp.Parameter((10, 4), value=new_itera_0_value, nonneg=True)
        self.itera_0.value = new_itera_0.value
        return ()
    

    def save_path_point(self, x, y, dir):
    # 打开文件，如果不存在则创建新文件
        with open("./results/traj.txt", "a") as file:
        # 将坐标和方向写入文件，以空格分隔，并以换行符结束
            file.write(f"{x} {y} {dir}\n")
        return ()

    def save_power(self, p1, p2, p3):
    # 打开文件，如果不存在则创建新文件
        with open("./results/power.txt", "a") as file:
        # 将坐标和方向写入文件，以空格分隔，并以换行符结束
            file.write(f"{p1} {p2} {p3}\n")
        return ()

    def save_path_dir(self,t,speed,dir):
    # 打开文件，如果不存在则创建新文件
        with open("./results/cmd_vel.txt", "a") as file:
        # 将坐标和方向写入文件，以空格分隔，并以换行符结束
            file.write(f"{t} {speed} {dir}\n")
        return ()

  
def get_data_from_csv(x, y, csv_file):
    # 根据输入的坐标计算行和列索引
    row_index = int(y * 10) - 1  # 行索引
    col_index = int(x * 10) - 1  # 列索引
    
    # 读取 CSV 文件
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    
    # 获取对应区域的数据
    try:
        value = float(data[row_index][col_index])
        if np.isinf(value):
            result = -160
        else:
            result = value
    except (IndexError, ValueError):
        result = -160  # 如果索引超出了范围或者数据为-inf，则返回 -160
    
    return result



    # endregion
    
    


    


    


    

