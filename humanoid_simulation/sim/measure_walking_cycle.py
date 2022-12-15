
import os
from datetime import datetime
import numpy as np
from numpy.linalg import norm, inv, pinv, eig, svd
import time
import matplotlib
# matplotlib.use('Agg')
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


def walking_data_collect(iter):
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
    force_direction_angles = 0
    force_magnitude = np.linspace(0, 0, num = 1)
    force_position_z = 0
    force_apply_time = np.linspace(0, 0, num = 1)
    # force_apply_time = 500
    iter_name = 'Force_magnitude_vs_Force_apply_time'
    # force_apply_time = 400 + iter * (600/9)

    # Setup pybullet for the quadruped and a wrapper to pinocchio.
    bipd = Biped(doFallSimulation = False, rendering = 0)
    bipd.rendering = 0
    bipd.useTorqueCtrl = False

    # Get the current state and modify the joints to have the legs
    # bend inwards.
    q, dq = bipd.get_state()
    false_prediction_list = []

    k = 0
    l = 0
    _m = 0
    n = 0

    # endregion
    for _j in range(force_magnitude.size * force_apply_time.size):
        j = _j + iter*1000
        if j == (1+iter)*1000:
            break
        gc.collect()
        # k = j%(force_magnitude.size * force_position_z.size * force_apply_time.size)%force_direction_angles.size # k is counter for force_direction_angles
        # if j>0 and j%force_direction_angles.size == 0:
        #     l = (l+1)%force_magnitude.size # l is counter for force_magnitude
        #     if j%(force_magnitude.size * force_direction_angles.size) == 0:
        #         _m = (_m +1)%force_apply_time.size # m is counter for force_position_z
        #         if j%(force_position_z.size * force_magnitude.size * force_direction_angles.size) == 0:
        #             n += 1

        k = j%(force_magnitude.size)%force_apply_time.size # k is counter for force_direction_angles
        if j>0 and j%force_apply_time.size == 0:
            l += 1 # l is counter for force_magnitude


        force_direction_angles = 180/180*np.pi

        force_direction = np.array([ np.cos(force_direction_angles), np.sin(force_direction_angles) ])
        
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
        if j>0:
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
            "urdf", "biped.urdf") 
            )
        meshPath = str(bipd.pack_path)
        dataPath = str(os.path.abspath(os.path.join(
        bipd.pack_path, '../humanoid_control/data')))
        _resultPath = str(os.path.abspath(os.path.join(
        bipd.pack_path, '../humanoid_simulation/data')))
        str_=datetime.now().strftime("%Y_%m_%d")
        resultPath = _resultPath + "/" + str_ + "/" + iter_name + '/' + str(j + 1)

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
        
        model = simulator.model
        data = simulator.data
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
        torso_v_arr = np.zeros([3, horizon_length])
        torso_a_arr = np.zeros([4, horizon_length])
        com_arr = np.zeros([3, horizon_length])

        # add the resultant cop force
        cop_arr = np.zeros([4, horizon_length])

        torso_orientation_arr = np.zeros([3,horizon_length])
        j_position_arr = np.zeros([19,horizon_length])
        j_velocity_arr = np.zeros([18,horizon_length])
        external_force_applied = np.zeros((horizon_length,1))
        linear_momentum_arr = np.zeros([3,horizon_length])
        angular_momentum_arr = np.zeros([3,horizon_length])
        linear_momentum_deri_arr = np.zeros([3,horizon_length])
        angular_momentum_deri_arr = np.zeros([3,horizon_length])

        
        dq0 = np.zeros(18)
        fall_predicted = False
        fall_start_time = 0
        fall_predict_lead = 0

        lfootID = 19
        rfootID = 33

        start = False

        # two feet touch the ground

        last_foot_state = [19, 33]

        right_foot_height = []

        for i in range(horizon_length):
            # Get the current state (position and velocity)
            q, dq = bipd.get_state()
            active_contact_frames, contact_forces, contact_cop,contact_cop_force, contact_cop_ft_l, contact_cop_ft_r = bipd.get_force()
            active_contact_frames_link, contact_forces_link = bipd.get_force_link()



            # update progress
            # if i % 10 == 0:
            #     sys.stdout.write("\r")
            #     sys.stdout.write(', '.join([ str(x) for x in active_contact_frames]))
            #     sys.stdout.flush()

            # print(bipd.get_foot_pos(q))

            right_foot_height.append(bipd.get_foot_pos(q)["leftfoot"][2])


            if rfootID not in last_foot_state and rfootID in active_contact_frames:

                print("right foot touches on the ground: %d"%i)

                print(last_foot_state)
                print(active_contact_frames)


            last_foot_state = active_contact_frames.copy()

            cop_pt1=[contact_cop[0], contact_cop[1], 0]
            cop_pt2=[contact_cop[0], contact_cop[1], contact_cop_force*0.005]
            color = [0, 1, 0]
            p.addUserDebugLine(cop_pt1,
                     cop_pt2,
                     color,1,0.1)
            # Alternative, if you want to use properties from the pinocchio robot
            # like the jacobian or similar, you can also get the state and update
            # the pinocchio internals with one call:
        
            q, dq = bipd.get_state_update_pinocchio()
            # Get the current center of mass
            com = se3.centerOfMass(bipd.pinocchio_robot.model, bipd.pinocchio_robot.data, q)           
            ddq = (vec2list(dq)-dq0)/0.008
            dq0 = np.array(vec2list(dq))
            # Get current linear momentum and angular momentum around CoM
            se3.computeCentroidalMomentum(bipd.pinocchio_robot.model,bipd.pinocchio_robot.data,q,dq)
            se3.computeCentroidalMomentumTimeVariation(bipd.pinocchio_robot.model,bipd.pinocchio_robot.data,q,dq,ddq)
           

            cop_arr[:3, i] = contact_cop
            cop_arr[3, i] = contact_cop_force
            torso_v_arr[:, i] = dq[0:3].transpose()
            if i > 0:
                torso_a_arr[:3, i] = (torso_v_arr[:, i] - torso_v_arr[:, i-1])/0.008
                torso_a_arr[3, i] = np.sqrt(torso_a_arr[0, i]**2 + torso_a_arr[1, i]**2 + torso_a_arr[2, i]**2)

            # com orientation is actually torso orientation
            torso_orientation_arr[:,i] = p.getEulerFromQuaternion(q[3:7])
            j_position_arr[:,i] = q.T
            j_velocity_arr[:,i] = dq.T
            com_arr[:, i] = vec2list(com)
            linear_momentum_arr[:,i] = vec2list(bipd.pinocchio_robot.data.hg.linear)
            angular_momentum_arr[:,i] = vec2list(bipd.pinocchio_robot.data.hg.angular)
            linear_momentum_deri_arr[:,i] = vec2list(bipd.pinocchio_robot.data.dhg.linear)
            angular_momentum_deri_arr[:,i] = vec2list(bipd.pinocchio_robot.data.dhg.angular)
            
            # Send the commands to the robot.
            if bipd.doFallSimulation == False:
                if i < horizon_length_data:
                    q_des = joint_traj[:, i]
                else:
                    q_des = joint_traj[:, horizon_length_data-1]
            bipd.send_joint_command(q_des)

        
            if isApplyForce == True and i > force_apply_time[l] and i < force_apply_time[l]+2:
                # Apply external force
                force = [-force_magnitude[k] * force_direction[0], -force_magnitude[k] * force_direction[1], 0]
                com = vec2list(com)
                forcePos = [0, 0, force_position_z]  # define point in front of torso           
                debug_line_force_from_ptr=((np.asarray(forcePos)-np.asarray(force))*0.003).tolist()
                debug_line_force_from_ptr[2]= forcePos[2]-force[2]
                #print(debug_line_force_from_ptr,forcePos,force)
                # print('current force magnitude:'+str(force_magnitude[k]))
                p.applyExternalForce(objectUniqueId=bipd.robotId, linkIndex=0, forceObj=force, posObj=forcePos, flags=p.LINK_FRAME)
                p.addUserDebugLine(debug_line_force_from_ptr,forcePos,[1,0,0],lifeTime=0.5,parentObjectUniqueId=bipd.robotId,parentLinkIndex=0)
                p.addUserDebugText(str(force_magnitude[k]),debug_line_force_from_ptr,[1,0,0],lifeTime=0.5,parentObjectUniqueId=bipd.robotId,parentLinkIndex=0)
                external_force_applied[i] = True
            else:
                external_force_applied[i] = False

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

        if fall_predicted == True and fall_start_time > 0:
            fall_predict_lead = fall_start_time - fall_predicted_time


        # Show simulation result.
        # region
        if isShowVelocityAccelerationResult == True:
            fig = plt.figure(figsize=(12, 30))
            plt.subplot(7, 1, 1)
            plt.plot(np.arange(horizon_length), torso_v_arr[0, :])
            plt.subplot(7, 1, 2)
            plt.plot(np.arange(horizon_length), torso_v_arr[1, :])
            plt.subplot(7, 1, 3)
            plt.plot(np.arange(horizon_length), torso_v_arr[2, :])
            plt.subplot(7, 1, 4)
            plt.plot(np.arange(horizon_length), torso_a_arr[0, :])
            plt.subplot(7, 1, 5)
            plt.plot(np.arange(horizon_length), torso_a_arr[1, :])
            plt.subplot(7, 1, 6)
            plt.plot(np.arange(horizon_length), torso_a_arr[2, :])
            plt.subplot(7, 1, 7)
            plt.plot(np.arange(horizon_length), torso_a_arr[3, :])
            plt.savefig(os.path.join(resultPath, 'velocity_acceleration.png'))
            fig.clear()
            plt.close(fig)

        if isShowComResult == True:
            fig2=plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(np.arange(horizon_length), com_traj_arr[0, :])
            plt.plot(np.arange(horizon_length), com_arr[0, :])
            plt.subplot(3, 1, 2)
            plt.plot(np.arange(horizon_length), com_traj_arr[1, :])
            plt.plot(np.arange(horizon_length), com_arr[1, :])
            plt.subplot(3, 1, 3)
            plt.plot(np.arange(horizon_length), com_traj_arr[2, :])
            plt.plot(np.arange(horizon_length), com_arr[2, :])
            plt.savefig(os.path.join(resultPath, 'com_check_time.png'))
            #fig3=plt.figure()
            #ax = fig3.add_subplot()
            fig3,ax = plt.subplots(1)
            # print(foot_print)
            # print(foot_length)
            # print(foot_width)
            for i in range(np.size(foot_print, 1)):
                anchor_x=foot_print[0, i] - foot_length/2.
                anchor_y=foot_print[1, i] - foot_width/2.
                rectangle=plt.Rectangle((anchor_x, anchor_y), foot_length, foot_width, linewidth=1,edgecolor='b',facecolor='none')
                ax.add_patch(rectangle)
            plt.plot(com_arr[0, :], com_arr[1, :])
            plt.plot(com_traj_arr[0, :], com_traj_arr[1, :])
            plt.savefig(os.path.join(resultPath, 'com_check_xy.png'))
            fig2.clear()
            plt.close(fig2)
            fig3.clear()
            plt.close(fig3)

        if isShowCopResult == True:
            fig4=plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(np.arange(horizon_length), cop_traj_arr[0, :])
            plt.plot(np.arange(horizon_length), cop_arr[0, :])
            plt.subplot(3, 1, 2)
            plt.plot(np.arange(horizon_length), cop_traj_arr[1, :])
            plt.plot(np.arange(horizon_length), cop_arr[1, :])
            plt.subplot(3, 1, 3)
            plt.plot(np.arange(horizon_length), cop_traj_arr[2, :])
            plt.plot(np.arange(horizon_length), cop_arr[2, :])
            plt.savefig(os.path.join(resultPath, 'cop_check_time.png'))
            #fig5=plt.figure()
            #ax = fig5.add_subplot()
            fig5,ax = plt.subplots(1)
            for i in range(np.size(foot_print, 1)):
                anchor_x=foot_print[0, i] - foot_length/2.
                anchor_y=foot_print[1, i] - foot_width/2.
                rectangle=plt.Rectangle((anchor_x, anchor_y), foot_length, foot_width, linewidth=1,edgecolor='b',facecolor='none')
                ax.add_patch(rectangle)
            plt.plot(cop_arr[0, :], cop_arr[1, :])
            plt.plot(cop_traj_arr[0, :], cop_traj_arr[1, :])
            plt.savefig(os.path.join(resultPath, 'cop_check_xy.png'))
            fig4.clear()
            plt.close(fig4)
            fig5.clear()
            plt.close(fig5)
        # endregion

        if isRecordValues == True:
            torso_position = np.append(external_force_applied.T, j_position_arr[0:3,:], axis=0)
            torso_orientation = np.append(external_force_applied.T, torso_orientation_arr, axis=0)
            torso_translation = np.append(external_force_applied.T, j_velocity_arr[0:3,:], axis=0)
            torso_rotation = np.append(external_force_applied.T, j_velocity_arr[3:6,:], axis=0)
            com_position = np.append(external_force_applied.T,com_arr, axis=0)
            j_position = np.append(external_force_applied.T, j_position_arr[7:,:], axis=0)
            j_velocity = np.append(external_force_applied.T, j_velocity_arr[6:,:], axis=0)
            cop_data = np.append(external_force_applied.T, cop_arr[:,:], axis=0)
            linear_momentum_data = np.append(external_force_applied.T, linear_momentum_arr[0:3,:], axis=0)
            angular_momentum_data = np.append(external_force_applied.T, angular_momentum_arr[0:3,:], axis=0)
            linear_momentum_deri_data = np.append(external_force_applied.T, linear_momentum_deri_arr[0:3,:], axis=0)
            angular_momentum_deri_data = np.append(external_force_applied.T, angular_momentum_deri_arr[0:3,:], axis=0)

            np.savetxt(resultPath + '/torso_position.csv', torso_position, delimiter=',')
            np.savetxt(resultPath + '/torso_orientation.csv', torso_orientation, delimiter=',')
            np.savetxt(resultPath + '/torso_translation.csv', torso_translation, delimiter=',')
            np.savetxt(resultPath + '/torso_rotation.csv', torso_rotation, delimiter=',')
            np.savetxt(resultPath + '/com_position.csv', com_position, delimiter=',')
            np.savetxt(resultPath + '/joint_position.csv', j_position, delimiter=',')
            np.savetxt(resultPath + '/joint_velocity.csv', j_velocity, delimiter=',')
            np.savetxt(resultPath + '/cop_data.csv', cop_data, delimiter=',')
            np.savetxt(resultPath + '/linear_momentum_data.csv', linear_momentum_data, delimiter=',')
            np.savetxt(resultPath + '/angular_momentum_data.csv', angular_momentum_data, delimiter=',')
            np.savetxt(resultPath + '/linear_momentum_deri_data.csv', linear_momentum_deri_data, delimiter=',')
            np.savetxt(resultPath + '/angular_momentum_deri_data.csv', angular_momentum_deri_data, delimiter=',')

            config = configparser.RawConfigParser()
            config.add_section('Data')
            config.set('Data', 'force magnitude', force_magnitude[k])
            config.set('Data', 'force direction', [force_direction[0], force_direction[1]] )
            config.set('Data', 'force height', force_position_z)
            config.set('Data', 'force application time', force_apply_time)
            config.set('Data', 'fall', robotFall)
            if doPrediction == True:
                config.set('Data', 'false predict', false_predict)
                config.set('Data', 'gain balance', gain_balance)
                config.set('Data', 'lead time', fall_predict_lead)
            with open(resultPath + '/dataInfo.json', 'w') as configfile:
                config.write(configfile)

        # # Print the final active force frames and the forces
        # active_contact_frames, contact_forces, contact_cop,contact_cop_force, contact_cop_ft_l, contact_cop_ft_r=bipd.get_force()

        print("iteration: ", j+1)

        end_time = time.time()
        print("time cost:", end_time-start_time)


        plt.plot(right_foot_height)
        plt.show()


    np.savetxt(_resultPath + '/' + str_ + 'false_prediction_list.csv', false_prediction_list, delimiter=',')


if __name__ == '__main__':
    # iter = int(sys.argv[1])


    import os
    import inspect
    import concurrent.futures
    import numpy as np
    import pickle

    # walking

    # # paras = [0, 1, 2, 3, 4]


    # with concurrent.futures.ProcessPoolExecutor() as executor:	
    #     results = list(executor.map(walking_data_collect, paras[:]), total=len(paras))

# for i, result in enumerate(results):
# 	print('delay number is ', result.delay,'throw time is ', result.throw_time,\
# 		", group number is ", result.group, \
# 			", throw distance: ", result.distance, ', final com: ', result.comz)

# with open(savepath+'task_eval.pkl', 'wb') as f:
# 	pickle.dump(results,f)

    iter=0

    walking_data_collect(iter)