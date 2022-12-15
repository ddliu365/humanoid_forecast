
import os
from datetime import datetime
import numpy as np
from numpy.linalg import norm, inv, pinv, eig, svd
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import fmin_slsqp

import pybullet as p
import pybullet_data
import pinocchio as se3
from pinocchio.utils import *
import cv2
import configparser

from py_pinocchio_bullet.wrapper import PinBulletWrapper
from robot_properties_solo.config import SoloConfig

from os.path import join, dirname

import py_lqr.model as lhm
import py_lqr.planner as pl
import py_lqr.trajectory_generation as tg

import gc

import sys

from model import Biped

import inspect
import concurrent.futures
import pickle


bipd = Biped(doFallSimulation = False, rendering = 0)
bipd.rendering = 0
bipd.useTorqueCtrl = False

def walking_data_collect(args):

    counter, parameter = args[0], args[1]

    angle, force_magnitude, force_apply_time = parameter[0], parameter[1], parameter[2]


    print(counter, parameter, angle, force_magnitude, force_apply_time)
    start_time = time.time()

    vec2list = lambda m: np.array(m.T).reshape(-1).tolist()
    np.set_printoptions(precision=2, suppress=True)

    # Setup simulation length
    horizon_length = 700

    # Setup simulation output
    isShowVelocityAccelerationResult = True
    isShowComResult = True
    isShowCopResult = True
    isRecordValues = True
    doPrediction = False 
    
    # Setup external force
    isApplyForce = True
    gc.collect()

    iter_name = 'Force_magnitude_vs_Force_apply_time'

    # Setup pybullet for the quadruped and a wrapper to pinocchio.


    # Get the current state and modify the joints to have the legs
    # bend inwards.
    q, dq = bipd.get_state()
    false_prediction_list = []
        
    # Setup initial condition
    if bipd.doFallSimulation == False:
        q[7] = q[13] = 0
        q[8] = q[14] = 0
        q[9] = q[15] = 25./180*3.14
        q[10] = q[16] = -50./180*3.14
        q[11] = q[17] = -25./180*3.14
        q[12] = q[18] = 0
    else:

        q[7] = q[13] = 0
        q[8] = q[14] = 0
        q[9] = q[15] = 30./180*3.14
        q[10] = q[16] = 0./180*3.14
        q[11] = q[17] = 30./180*3.14
        q[12] = q[18] = 0

    # Reset environment
    # if j>0:
    q[0] = 0
    q[1] = 0
    q[2] = 0.0014875
    q[3] = 0
    q[4] = 0
    q[5] = 0
    q[6] = 1
    dq = np.zeros((18,1))
    bipd.reset_state(q, dq)
    p.resetBaseVelocity(bipd.robotId, [0, 0, 0], [0, 0, 0])

    if bipd.doFallSimulation == True:
        # Setup initial velocity while falling
        v_h = 2.3
        v_v = v_h/np.cos(4/180*3.14)*np.sin(4/180*3.14)
    else:
        v_h = 0.
        v_v = 0.
        dq[0] = v_h
        dq[1] = v_v

    # Take the initial joint states as desired state.
    q_des = q[7:].copy()

    # Update the simulation state to the new initial configuration.
    bipd.reset_state(q, dq)

    # For test: foot distance difference: 0.079
    stat = p.getLinkState(bipd.robotId, bipd.bullet_endeff_ids[0])
    pos = stat[0]
    # print('Simulator: Initial Feet Height:', pos)

    # Call calculated trajectory
    # region
    _isOnlineCompute = False
    urdfPath = str(
        join(bipd.pack_path,
        "urdf", "humanoid_pinocchio.urdf") 
        )
    meshPath = str(bipd.pack_path)
    dataPath = str(os.path.abspath(os.path.join(
    bipd.pack_path, '../humanoid_control/data')))
    _resultPath = str(os.path.abspath(os.path.join(
    bipd.pack_path, '../humanoid_simulation/data')))
    str_=datetime.now().strftime("%Y_%m_%d")
    resultPath = _resultPath + "/" + str_ + "/" + str(counter)



    #python program to check if a directory exists


    

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")




    # path = "directory"
    # Check whether the specified path exists or not
    isExist = os.path.exists(resultPath)  
    #printing if the path exists or not
    # print(isExist)

    if not isExist:
        os.makedirs(resultPath)

    else:
        resultPath +="/"+dt_string
        os.makedirs(resultPath)

    robot = lhm.loadHumanoidModel(
    isDisplay=False, urdfPath=urdfPath, meshPath=meshPath)
    foot_width = robot.foot_width
    foot_length = robot.foot_length
    planner = pl.stepPlanner(robot)
    foot_print = planner.foot_print
    traj = tg.trajectoryGenerate(
    robot, planner, isDisplay=False, data_path=dataPath)
    simulator = tg.simulator(robot, traj, isRecompute=False, isOnlineCompute=_isOnlineCompute)
    if _isOnlineCompute == False:
        joint_traj = simulator.joint_traj
        com_traj = simulator.com_traj
        cop_traj = simulator.cop_traj
    
    # model = simulator.model
    # data = simulator.data
    horizon_length = simulator.horizon_length + 300
    horizon_length_data = simulator.horizon_length

    # planned trajectory
    joint_traj_arr = np.zeros([np.size(joint_traj, 0),horizon_length])
    joint_traj_arr[:,0:np.size(joint_traj,1)] = joint_traj
    com_traj_arr = np.zeros([np.size(com_traj, 0),horizon_length])
    com_traj_arr[:,0:np.size(com_traj,1)] = com_traj
    cop_traj_arr = np.zeros([np.size(cop_traj, 0),horizon_length])
    cop_traj_arr[:,0:np.size(cop_traj,1)] = cop_traj
    # endregion

    # simulated result trajectory
    # torso_v_arr = np.zeros([3, horizon_length])
    # torso_a_arr = np.zeros([4, horizon_length])
    com_arr = np.zeros([3, horizon_length])

    # add the resultant cop force
    cop_arr = np.zeros([12, horizon_length])

    # torso_orientation_arr = np.zeros([3,horizon_length])
    # j_position_arr = np.zeros([19,horizon_length])
    # j_velocity_arr = np.zeros([18,horizon_length])
    # external_force_applied = np.zeros((horizon_length,1))
    # linear_momentum_arr = np.zeros([3,horizon_length])
    # angular_momentum_arr = np.zeros([3,horizon_length])
    # linear_momentum_deri_arr = np.zeros([3,horizon_length])
    # angular_momentum_deri_arr = np.zeros([3,horizon_length])

    
    # Run the simulator for 2000 steps = 2 seconds.
    # previous dq
    dq0 = np.zeros(18)
    # fall_predicted = False
    # fall_start_time = 0
    # fall_predict_lead = 0

    force_direction_angles = angle/180*np.pi

    force_direction = np.array([ np.cos(force_direction_angles), np.sin(force_direction_angles) ])

    for i in range(horizon_length):
        # Get the current state (position and velocity)
        q, dq = bipd.get_state()
        active_contact_frames, contact_forces, contact_cop,contact_cop_force, contact_cop_ft_l, contact_cop_ft_r = bipd.get_force()
        # active_contact_frames_link, contact_forces_link = bipd.get_force_link()

        q, dq = bipd.get_state_update_pinocchio()
        # Get the current center of mass
        com = se3.centerOfMass(bipd.pinocchio_robot.model, bipd.pinocchio_robot.data, q)           

        cop_arr[:3, i] = bipd.get_foot_pos(q)["leftfoot"]

        cop_arr[3:5, i] = bipd.get_left_foot_cop()

        cop_arr[5, i] = bipd.get_left_foot_normal_force()

        cop_arr[6:9, i] =  bipd.get_foot_pos(q)["rightfoot"]

        cop_arr[9:11, i] = bipd.get_right_foot_cop()

        cop_arr[11, i] = bipd.get_right_foot_normal_force()





        cop_arr[:3, i] = contact_cop
        cop_arr[3, i] = contact_cop_force
        # torso_v_arr[:, i] = dq[0:3].transpose()
        # if i > 0:
        #     torso_a_arr[:3, i] = (torso_v_arr[:, i] - torso_v_arr[:, i-1])/0.008
        #     torso_a_arr[3, i] = np.sqrt(torso_a_arr[0, i]**2 + torso_a_arr[1, i]**2 + torso_a_arr[2, i]**2)
        # endregion
        # com orientation is actually torso orientation
        # torso_orientation_arr[:,i] = p.getEulerFromQuaternion(q[3:7])
        # j_position_arr[:,i] = q.T
        # j_velocity_arr[:,i] = dq.T
        com_arr[:, i] = vec2list(com)
        # linear_momentum_arr[:,i] = vec2list(bipd.pinocchio_robot.data.hg.linear)
        # angular_momentum_arr[:,i] = vec2list(bipd.pinocchio_robot.data.hg.angular)
        # linear_momentum_deri_arr[:,i] = vec2list(bipd.pinocchio_robot.data.dhg.linear)
        # angular_momentum_deri_arr[:,i] = vec2list(bipd.pinocchio_robot.data.dhg.angular)
        

        # Send the commands to the robot.
        if bipd.doFallSimulation == False:
            if i < horizon_length_data:
                q_des = joint_traj[:, i]
            else:
                q_des = joint_traj[:, horizon_length_data-1]
        bipd.send_joint_command(q_des)

    
        if i > force_apply_time and i < force_apply_time+2:
            # Apply external force
            force = [-force_magnitude * force_direction[0], -force_magnitude * force_direction[1], 0]
            com = vec2list(com)
            forcePos = [0, 0, 0]  # define point in front of torso           
            p.applyExternalForce(objectUniqueId=bipd.robotId, linkIndex=0, forceObj=force, posObj=forcePos, flags=p.LINK_FRAME)

        # Step the simulator and sleep.
        p.stepSimulation()

        #checking state
        if fall_start_time == 0 and com[2] < 0.2:
            fall_start_time = i

    if com_arr[2,-1] < 0.2:
        robotFall = True
        if fall_predicted == False and doPrediction == True:
            false_predict = True
            gain_balance = False
            false_prediction_list.append(j+1) 
        elif fall_predicted == True and doPrediction == True:
            false_predict = False
            gain_balance = False
    else:
        robotFall = False
        if fall_predicted == True and doPrediction == True:
            false_predict = True
            gain_balance = True
            false_prediction_list.append(j+1) 
        elif fall_predicted == False and doPrediction == True:
            false_predict = False
            gain_balance = False

    if isRecordValues == True:
        # torso_position = np.append(external_force_applied.T, j_position_arr[0:3,:], axis=0)
        # torso_orientation = np.append(external_force_applied.T, torso_orientation_arr, axis=0)
        # torso_translation = np.append(external_force_applied.T, j_velocity_arr[0:3,:], axis=0)
        # torso_rotation = np.append(external_force_applied.T, j_velocity_arr[3:6,:], axis=0)
        # com_position = np.append(external_force_applied.T,com_arr, axis=0)
        # j_position = np.append(external_force_applied.T, j_position_arr[7:,:], axis=0)
        # j_velocity = np.append(external_force_applied.T, j_velocity_arr[6:,:], axis=0)
        # cop_data = np.append(external_force_applied.T, cop_arr[:,:], axis=0)
        # linear_momentum_data = np.append(external_force_applied.T, linear_momentum_arr[0:3,:], axis=0)
        # angular_momentum_data = np.append(external_force_applied.T, angular_momentum_arr[0:3,:], axis=0)
        # linear_momentum_deri_data = np.append(external_force_applied.T, linear_momentum_deri_arr[0:3,:], axis=0)
        # angular_momentum_deri_data = np.append(external_force_applied.T, angular_momentum_deri_arr[0:3,:], axis=0)

        # np.savetxt(resultPath + '/torso_position.csv', torso_position, delimiter=',')
        # np.savetxt(resultPath + '/torso_orientation.csv', torso_orientation, delimiter=',')
        # np.savetxt(resultPath + '/torso_translation.csv', torso_translation, delimiter=',')
        # np.savetxt(resultPath + '/torso_rotation.csv', torso_rotation, delimiter=',')



        








        np.savetxt(resultPath + '/com.csv', com_arr.T, delimiter=',', header='x, y, z')
        # np.savetxt(resultPath + '/joint_position.csv', j_position, delimiter=',')
        # np.savetxt(resultPath + '/joint_velocity.csv', j_velocity, delimiter=',')
        np.savetxt(resultPath + '/cop.csv', cop_arr.T, delimiter=',',\
            header= 'l_foot_x, l_foot_y, l_foot_z, l_foot_cop_x, l_foot_cop_y, l_foot_f \
                     r_foot_x, r_foot_y, r_foot_z, r_foot_cop_x, r_foot_cop_y, r_foot_f')

        # np.savetxt(resultPath + '/linear_momentum_data.csv', linear_momentum_data, delimiter=',')
        # np.savetxt(resultPath + '/angular_momentum_data.csv', angular_momentum_data, delimiter=',')
        # np.savetxt(resultPath + '/linear_momentum_deri_data.csv', linear_momentum_deri_data, delimiter=',')
        # np.savetxt(resultPath + '/angular_momentum_deri_data.csv', angular_momentum_deri_data, delimiter=',')

        config = configparser.RawConfigParser()
        config.add_section('data')
        config.set('data', 'force magnitude', force_magnitude)
        config.set('data', 'force angle',  angle)
        # config.set('Data', 'force height', force_position_z)
        config.set('data', 'force application time', force_apply_time)
        config.set('data', 'fall status', robotFall)
        config.set('data', "session ID", counter)
        with open(resultPath + '/config.json', 'w') as configfile:
            config.write(configfile)

        # Print the final active force frames and the forces
        # active_contact_frames, contact_forces, contact_cop,contact_cop_force, contact_cop_ft_l, contact_cop_ft_r=bipd.get_force()

        #print("Active force_frames:", force_frames)
        #print("Corresponding forces:", forces)
        print("iteration: ", counter)

        end_time = time.time()
        print("time cost:", end_time-start_time)


    # np.savetxt(_resultPath + '/' + str_ + 'false_prediction_list.csv', false_prediction_list, delimiter=',')


if __name__ == '__main__':
    iter_ = int(sys.argv[1])



    # walking

    # # paras = [0, 1, 2, 3, 4]


   

# for i, result in enumerate(results):
# 	print('delay number is ', result.delay,'throw time is ', result.throw_time,\
# 		", group number is ", result.group, \
# 			", throw distance: ", result.distance, ', final com: ', result.comz)

# with open(savepath+'task_eval.pkl', 'wb') as f:
# 	pickle.dump(results,f)



    N = 10

    N_angles = 16
    range_angles= [-180, 180]
    sigma_angles = abs(range_angles[1]-range_angles[0])/N_angles/2
    N_magnitudes = 10
    range_magnitudes = [250, 450]
    sigma_magnitudes = (range_magnitudes[1]-range_magnitudes[0])/N_magnitudes/2
    N_time = 8
    range_time = [218, 435]
    sigma_time = (range_time[1]-range_time[0])/N_time/2

    N_random = 10



    force_direction_angles = np.linspace(range_angles[0], range_angles[1], N_angles)
    force_magnitude = np.linspace(range_magnitudes[0], range_magnitudes[1], N_magnitudes)
    force_apply_time = np.linspace(218, 435, N_time)


    parameters = []


    for angle in force_direction_angles:

        for magnitude in force_magnitude:

            for apply_time in force_apply_time:

                for r in range(N_random):

                    angle_r = np.random.normal(angle, sigma_angles, 1)[0]

                    magnitude_r = np.random.normal(magnitude, sigma_magnitudes, 1)[0]

                    apply_time_r = np.random.normal(apply_time, sigma_time, 1)[0]

                    parameters.append([angle_r, magnitude_r, apply_time_r])




    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


    N = N_angles*N_magnitudes*N_time
    split_N = int(N/14)+1

    print('total packages:', split_N)


    splited_list = list(split(parameters, split_N))

    print(len(splited_list[0]), len(parameters))




    # iter=0



    iter_parameters = []
    for (i, parameter) in enumerate(splited_list[iter_]):

        iter_parameters.append((14*iter_+i, parameter))
        # walking_data_collect(bipd, i, parameter)

    print(iter_parameters)

    # return




    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(walking_data_collect, iter_parameters[:])




    