import os
from datetime import datetime
import numpy as np
from numpy.linalg import norm, inv, pinv, eig, svd
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
# from lstm import Predictor
# import torch
import sys


class Biped(PinBulletWrapper):
    def __init__(self, physicsClient=None, doFallSimulation = False, rendering = 0):
        if physicsClient is None:
            self.physicsClient = p.connect(p.GUI)

            # Setup for fall simulation, while setup walking, do turn this off!
            self.doFallSimulation = doFallSimulation
            p.setGravity(0, 0, -9.81)
            # p.setGravity(0,0,0)
            p.setPhysicsEngineParameter(fixedTimeStep=8.0/1000.0, numSubSteps=1)

            # Load the plain.

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            plain_urdf = (SoloConfig.packPath + "/urdf/plane_with_restitution.urdf")
            self.planeId = p.loadURDF(plain_urdf)

            # Load the robot
            if self.doFallSimulation == True:
                robotStartPos = [0., 0, 0.0014875+0.15]
            else:
                robotStartPos = [0., 0, 0.0014875]
            robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

            # Use real time simulation
            self.useRealTimeSim = 0

            # Use rendering tool for real time video output
            self.rendering = rendering
            self.urdf_path_pybullet = SoloConfig.urdf_path_pybullet
            self.pack_path = SoloConfig.packPath
            self.urdf_path = SoloConfig.urdf_path
            self.robotId = p.loadURDF(self.urdf_path_pybullet, robotStartPos,
                robotStartOrientation, flags=p.URDF_USE_INERTIA_FROM_FILE,
                useFixedBase=False)
            p.getBasePositionAndOrientation(self.robotId)

            # Create the robot wrapper in pinocchio.
            package_dirs = [os.path.dirname(
                os.path.dirname(self.urdf_path)) + '/urdf']
            self.pin_robot = SoloConfig.buildRobotWrapper()

            # Query all the joints.
            num_joints = p.getNumJoints(self.robotId)

            for ji in range(num_joints):
                p.changeDynamics(self.robotId, ji, linearDamping=0.04,
                    angularDamping=0.04, restitution=0., lateralFriction=1.0)

            self.base_link_name = "base_link"
            self.joint_names = ['j_leg_l_hip_y', 'j_leg_l_hip_r', 'j_leg_l_hip_p', 'j_leg_l_knee', 'j_leg_l_ankle_p',
            'j_leg_l_ankle_r', 'j_leg_r_hip_y', 'j_leg_r_hip_r', 'j_leg_r_hip_p', 'j_leg_r_knee', 'j_leg_r_ankle_p',
            'j_leg_r_ankle_r']
            controlled_joints = ['j_leg_l_hip_y', 'j_leg_l_hip_r', 'j_leg_l_hip_p', 'j_leg_l_knee', 'j_leg_l_ankle_p',
            'j_leg_l_ankle_r', 'j_leg_r_hip_y', 'j_leg_r_hip_r', 'j_leg_r_hip_p', 'j_leg_r_knee', 'j_leg_r_ankle_p',
            'j_leg_r_ankle_r']

            # Creates the wrapper by calling the super.__init__.
            super(Biped, self).__init__(self.robotId, self.pin_robot,
                controlled_joints,
                ['j_leg_l_foot', 'j_leg_r_foot']
            )

            # Adjust view and close unrelevant window

            # region
            resultPath = str(os.path.abspath(os.path.join(self.pack_path, '../humanoid_simulation/data')))

            if self.useRealTimeSim == 1:
                p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, os.path.join(resultPath, 'biped_log.mp4'))
                p.setRealTimeSimulation(self.useRealTimeSim)

            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.resetDebugVisualizerCamera(cameraDistance=0.03,
            cameraYaw=135,
            cameraPitch=0,
            cameraTargetPosition=[0.5, 0.5, 0.35])
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            self.camTargetPos = [0.7, 0.7, 0.35]
            self.cameraUp = [0, 0, 1]
            self.cameraPos = [0.5, 0.5, 0.35]
            self.yaw = 135.0
            self.pitch = 0.0
            self.roll = 0
            self.upAxisIndex = 1
            self.camDistance = 0.1
            self.pixelWidth = 1366
            self.pixelHeight = 1366
            self.nearPlane = 0.1
            self.farPlane = 100.0
            self.fov = 60
            self.aspect = float(self.pixelWidth) / self.pixelHeight
            self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(self.camTargetPos, self.camDistance, self.yaw, self.pitch,
                                                            self.roll, self.upAxisIndex)
            self.projectionMatrix = p.computeProjectionMatrixFOV(
                self.fov, self.aspect, self.nearPlane, self.farPlane)
            self.size = (self.pixelWidth, self.pixelHeight)
            # rendering parameter
            if self.rendering == 1:
                self.camTargetPos = [0.5, 0.5, 0.35]
                self.cameraUp = [0, 0, 1]
                self.cameraPos = [1, 1, 1]
                self.yaw = 135.0
                self.pitch = 0.0
                self.roll = 0
                self.upAxisIndex = 2
                self.camDistance = 0.03
                self.pixelWidth = 1366
                self.pixelHeight = 1366
                self.nearPlane = 0.1
                self.farPlane = 100.0
                self.fov = 60
                self.aspect = float(self.pixelWidth) / self.pixelHeight
                self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(self.camTargetPos, self.camDistance, self.yaw, self.pitch,
                                                                self.roll, self.upAxisIndex)
                self.projectionMatrix = p.computeProjectionMatrixFOV(
                    self.fov, self.aspect, self.nearPlane, self.farPlane)
                self.size = (self.pixelWidth, self.pixelHeight)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(os.path.join(resultPath, 'biped_realtime.mp4'), fourcc, 100.0, self.size)
            # endregion

def animate():
    # Add x and y to lists
    xs.append(com[0])
    # ys.append(com[1])
    # zs.append(com[2])
    ts.append(i*0.008)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    # ys = ys[-20:]
    # zs = ys[-20:]
    ts = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ts)
    # ax.plot(ys, ts)
    # ax.plot(zs, ts)

    # Format plot
    plt.ylabel('Center of Mass (m)')
    plt.xlabel('Time (s)')

if __name__ == "__main__":
    vec2list = lambda m: np.array(m.T).reshape(-1).tolist()
    np.set_printoptions(precision=2, suppress=True)

    # Setup simulation length
    horizon_length = 700

    # Setup simulation output
    isShowVelocityAccelerationResult = False
    isShowComResult = False
    isShowCopResult = False
    isRecordValues = False
    doPrediction = False
    
    # Setup external force
    isApplyForce = True
    force_direction_angles = -np.pi/2
    force_magnitude = 300
    force_position_z = 0    
    force_apply_time = 300
    # force_apply_time = 500
    iter_name = 'force_data'
    # force_apply_time = 400 + iter * (600/9)

    
    # Setup pybullet for the quadruped and a wrapper to pinocchio.
    bipd = Biped(doFallSimulation = False, rendering = 0)
    bipd.rendering = 0
    bipd.useTorqueCtrl = False



    # Get the current state and modify the joints to have the legs
    # bend inwards.
    q, dq = bipd.get_state()

    # verify joint positive direction
    # region
    '''
    q[7] = 20./180.* 3.14
    q[8] = -20./180.* 3.14
    # bend knee to avoid singularity
    q[9] = 15./180* 3.14  #leg_left_hip_pitch
    q[10] = -30./180.* 3.14 #leg_left_knee
    q[11] = -15./180.*3.14 #leg_left_ankle_pitch
    q[12] = 20./180.* 3.14

    q[13] = -20./180.* 3.14
    q[14] = -20./180.* 3.14
    q[15] = 15./180* 3.14 #leg_right_hip_pitch
    q[16] = -30./180.* 3.14 #leg_right_knee
    q[17] = -15./180.* 3.14 #leg_right_ankle_pitch
    q[18] = -20./180.* 3.14
    '''

    false_prediction_list = []
    k = 0
    l = 0
    _m = 0
    n = 0
    j = 0

    # endregion

    gc.collect()
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
    resultPath = _resultPath + "/" + iter_name + '/' + str(j + 1)
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
    cop_arr = np.zeros([3, horizon_length])
    torso_orientation_arr = np.zeros([3,horizon_length])
    j_position_arr = np.zeros([19,horizon_length])
    j_velocity_arr = np.zeros([18,horizon_length])
    external_force_applied = np.zeros((horizon_length,1))
    linear_momentum_arr = np.zeros([3,horizon_length])
    angular_momentum_arr = np.zeros([3,horizon_length])
    linear_momentum_deri_arr = np.zeros([3,horizon_length])
    angular_momentum_deri_arr = np.zeros([3,horizon_length])

    

    # # Load LSTM model
    # if doPrediction:
    #     predictor = Predictor(17, 16, 17)
    #     predictor.load_state_dict(torch.load('/home/jack/repos/forecast/offlineTrain/result/resultlstm712_31_21_36.pt'))
    #     predictor_x = []
    #     predictor_input = np.zeros([37+3+3+1,50])
    #     predictor_column = []

    # Run the simulator for 2000 steps = 2 seconds.
    # previous dq
    dq0 = np.zeros(18)
    fall_predicted = False
    fall_start_time = 0
    fall_predict_lead = 0
    def captureIMG(i):
        print(bipd.viewMatrix)
        print(bipd.projectionMatrix)
        img_arr = p.getCameraImage(bipd.pixelWidth,
                                    bipd.pixelHeight,
                                    bipd.viewMatrix,
                                    bipd.projectionMatrix,
                                    shadow=1,
                                    lightDirection=[1, 1, 1],
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)
        proj_opengl = np.uint8(np.reshape(img_arr[2], (bipd.pixelHeight, bipd.pixelWidth, 4)))
        # frame = cv2.resize(proj_opengl,(bipd.pixelWidth,bipd.pixelHeight))
        frame = cv2.cvtColor(proj_opengl, cv2.COLOR_RGB2BGR)
        filename = "frame"+str(i)+".jpg"
        cv2.imwrite(os.path.join(resultPath, filename), frame)


    # flot init
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    xs = []
    ys = []
    zs = []
    ts = []

    for i in range(horizon_length):
        # Get the current state (position and velocity)
        q, dq = bipd.get_state()
        active_contact_frames, contact_forces, contact_cop,contact_cop_force, contact_cop_ft_l, contact_cop_ft_r = bipd.get_force()
        active_contact_frames_link, contact_forces_link = bipd.get_force_link()
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


        # plot update
        # ani = animation.FuncAnimation(fig, animate)
        # plt.show()

        torso_pos = p.getLinkState(bipd.robotId, bipd.bullet_endeff_ids[0])
        torso_displacement = torso_pos[0][0]-stat[0][0]
        p.resetDebugVisualizerCamera(cameraDistance=0.03,
        cameraYaw=135,
        cameraPitch=0,
        cameraTargetPosition=[0.5+torso_displacement, 0.5, 0.35])
        bipd.viewMatrix = p.computeViewMatrixFromYawPitchRoll([0.5+torso_displacement, 0.5, 0.35], bipd.camDistance, bipd.yaw, bipd.pitch, bipd.roll, bipd.upAxisIndex)
    
        #print('Simulator: Linear momentum:', vec2list(bipd.pinocchio_robot.data.hg.linear))
        #print('Simulator: Angular momentum w.r.t CoM:', vec2list(bipd.pinocchio_robot.data.hg.angular))
        # Get the current acceleraton 
        # region
        cop_arr[:, i] = contact_cop
        torso_v_arr[:, i] = dq[0:3].transpose()
        if i > 0:
            torso_a_arr[:3, i] = (torso_v_arr[:, i] - torso_v_arr[:, i-1])/0.008
            torso_a_arr[3, i] = np.sqrt(torso_a_arr[0, i]**2 + torso_a_arr[1, i]**2 + torso_a_arr[2, i]**2)
        # endregion
        # com orientation is actually torso orientation
        torso_orientation_arr[:,i] = p.getEulerFromQuaternion(q[3:7])
        j_position_arr[:,i] = q.T
        j_velocity_arr[:,i] = dq.T
        com_arr[:, i] = vec2list(com)
        linear_momentum_arr[:,i] = vec2list(bipd.pinocchio_robot.data.hg.linear)
        angular_momentum_arr[:,i] = vec2list(bipd.pinocchio_robot.data.hg.angular)
        linear_momentum_deri_arr[:,i] = vec2list(bipd.pinocchio_robot.data.dhg.linear)
        angular_momentum_deri_arr[:,i] = vec2list(bipd.pinocchio_robot.data.dhg.angular)
        
        # if doPrediction:
        #     predictor_row = []
        #     if i < 50:
        #         k = i
        #     else:
        #         k = 49
        #         for n in range(48):
        #             predictor_input[:,n]=predictor_input[:,n+1]
        #     predictor_input[0:3,k] = torso_orientation_arr[:,i]
        #     predictor_input[6:8,k] = cop_arr[0:2,i]
        #     predictor_input[3:6,k] = j_position_arr[0:3,i]
        #     predictor_input[8:11,k] = j_velocity_arr[3:6,i]
        #     predictor_input[11:14,k] = j_velocity_arr[0:3,i]
        #     predictor_input[14:26,k] = j_position_arr[7:,i]
        #     predictor_input[26:38,k] = j_velocity_arr[6:,i]
        #     predictor_input[38:41,k] = linear_momentum_arr[:,i]
        #     predictor_input[41:44,k] = angular_momentum_arr[:,i]
        #     for n in range(3):
        #         predictor_row.append(float(com_arr[n,i]))
        #     '''
        #     for n in range(3):
        #         predictor_row.append(float(torso_orientation_arr[n,i]))
        #     for n in range(3):
        #         predictor_row.append(float(j_position_arr[n,i]))
        #     '''

        #     '''
        #     for n in range(3):
        #         predictor_row.append(float(j_velocity_arr[n+3,i]))
        #     for n in range(3):
        #         predictor_row.append(float(j_velocity_arr[n,i]))
        #     for n in range(12):
        #         predictor_row.append(float(j_position_arr[7+n,i]))
        #     for n in range(12):
        #         predictor_row.append(float(j_velocity_arr[6+n,i]))
        #     '''
        #     for n in range(3):
        #         predictor_row.append(float(linear_momentum_arr[n,i]))
        #     for n in range(3):
        #         predictor_row.append(float(angular_momentum_arr[n,i]))
        #     for n in range(3):
        #         predictor_row.append(float(linear_momentum_deri_arr[n,i]))
        #     for n in range(3):
        #         predictor_row.append(float(angular_momentum_deri_arr[n,i]))
        #     for n in range(2):
        #         predictor_row.append(float(cop_arr[n,i]))

        #     if i < 75:
        #         predictor_column.append(predictor_row)
        #         # print('prediction is not starting')
        #     else:
        #         predictor_column.pop(0)
        #         predictor_column.append(predictor_row)
        #         predictor_x.append(predictor_column)
        #         predictor_x = torch.FloatTensor(predictor_x)
        #         #print(predictor_x.size())
        #         predictor.eval()
        #         output = predictor(predictor_x)
        #         key = output.data.numpy()[0,0]
        #         print(key)
        #         predictor_x = []
        #         if fall_predicted == False and key > 0.5:
        #             fall_predicted = True 
        #             fall_predicted_time = i

        # print('Simulator: Linear momentum time derivative:', vec2list(bipd.pinocchio_robot.data.dhg.linear))
        # print('Simulator: Angular momentum time derivative w.r.t CoM:', vec2list(bipd.pinocchio_robot.data.dhg.angular))


        #if i % 100 == 0:
            #print('Simulator: Forces from EndEffectors:', active_contact_frames, contact_forces)
            #print('Simulator: Forces from Links:', active_contact_frames_link, contact_forces_link)

        # Compute the command torques at the joints. The torque
        # vector only takes the actuated joints (excluding the base)
        # tau = 1000. * (q_des - q[7:]) - 0.1 * dq[6:]

        # Send the commands to the robot.
        
        if bipd.doFallSimulation == False:
            if i < horizon_length_data:
                q_des = joint_traj[:, i]
            else:
                q_des = joint_traj[:, horizon_length_data-1]
        
        bipd.send_joint_command(q_des)

    
        if isApplyForce == True and i > force_apply_time and i < force_apply_time+2:
            # Apply external force
            force = [-force_magnitude * force_direction[0], -force_magnitude * force_direction[1], 0]
            com = vec2list(com)
            forcePos = [0, 0, force_position_z]  # define point in front of torso           
            debug_line_force_from_ptr=((np.asarray(forcePos)-np.asarray(force))*0.003).tolist()
            debug_line_force_from_ptr[2]= forcePos[2]-force[2]
            #print(debug_line_force_from_ptr,forcePos,force)
            print('current force magnitude:'+str(force_magnitude))
            p.applyExternalForce(objectUniqueId=bipd.robotId, linkIndex=0, forceObj=force, posObj=forcePos, flags=p.LINK_FRAME)
            p.addUserDebugLine(debug_line_force_from_ptr,forcePos,[1,0,0],lifeTime=0.5,parentObjectUniqueId=bipd.robotId,parentLinkIndex=0)
            p.addUserDebugText(str(force_magnitude),debug_line_force_from_ptr,[1,0,0],lifeTime=0.5,parentObjectUniqueId=bipd.robotId,parentLinkIndex=0)
            external_force_applied[i] = True
            captureIMG(i)
        elif isApplyForce == True and i > force_apply_time+25 and i < force_apply_time+25+2:
            captureIMG(i)
        elif isApplyForce == True and i > force_apply_time+50 and i < force_apply_time+50+2:
            captureIMG(i)
        elif isApplyForce == True and i > force_apply_time+75 and i < force_apply_time+75+2:
            captureIMG(i)
        elif isApplyForce == True and i > force_apply_time+600 and i < force_apply_time+600+2:
            # Apply external force
            force_magnitude = 400
            
            force = [-force_magnitude * force_direction[0], -force_magnitude * force_direction[1], 0]
            com = vec2list(com)
            forcePos = [0, 0, force_position_z]  # define point in front of torso           
            debug_line_force_from_ptr=((np.asarray(forcePos)-np.asarray(force))*0.003).tolist()
            debug_line_force_from_ptr[2]= forcePos[2]-force[2]
            #print(debug_line_force_from_ptr,forcePos,force)
            print('current force magnitude:'+str(force_magnitude))
            p.applyExternalForce(objectUniqueId=bipd.robotId, linkIndex=0, forceObj=force, posObj=forcePos, flags=p.LINK_FRAME)
            p.addUserDebugLine(debug_line_force_from_ptr,forcePos,[1,0,0],lifeTime=0.5,parentObjectUniqueId=bipd.robotId,parentLinkIndex=0)
            p.addUserDebugText(str(force_magnitude),debug_line_force_from_ptr,[1,0,0],lifeTime=0.5,parentObjectUniqueId=bipd.robotId,parentLinkIndex=0)
            external_force_applied[i] = True
            captureIMG(i)
        elif isApplyForce == True and i > force_apply_time+600+25 and i < force_apply_time+600+25+2:
            captureIMG(i)
        elif isApplyForce == True and i > force_apply_time+600+50 and i < force_apply_time+600+50+2:
            captureIMG(i)
        elif isApplyForce == True and i > force_apply_time+600+75 and i < force_apply_time+600+75+2:
            captureIMG(i)
        elif isApplyForce == True and i > force_apply_time+900 and i < force_apply_time+900+2:
            # Apply external force
            force_magnitude=500
            force = [-force_magnitude * force_direction[0], -force_magnitude * force_direction[1], 0]
            com = vec2list(com)
            forcePos = [0, 0, force_position_z]  # define point in front of torso           
            debug_line_force_from_ptr=((np.asarray(forcePos)-np.asarray(force))*0.003).tolist()
            debug_line_force_from_ptr[2]= forcePos[2]-force[2]
            #print(debug_line_force_from_ptr,forcePos,force)
            print('current force magnitude:'+str(force_magnitude))
            p.applyExternalForce(objectUniqueId=bipd.robotId, linkIndex=0, forceObj=force, posObj=forcePos, flags=p.LINK_FRAME)
            p.addUserDebugLine(debug_line_force_from_ptr,forcePos,[1,0,0],lifeTime=0.5,parentObjectUniqueId=bipd.robotId,parentLinkIndex=0)
            p.addUserDebugText(str(force_magnitude),debug_line_force_from_ptr,[1,0,0],lifeTime=0.5,parentObjectUniqueId=bipd.robotId,parentLinkIndex=0)
            external_force_applied[i] = True
            captureIMG(i)
        elif isApplyForce == True and i > force_apply_time+900+25 and i < force_apply_time+900+25+2:
            captureIMG(i)
        elif isApplyForce == True and i > force_apply_time+900+50 and i < force_apply_time+900+50+2:
            captureIMG(i)
        elif isApplyForce == True and i > force_apply_time+900+100 and i < force_apply_time+900+100+2:
            captureIMG(i)
        else:
            external_force_applied[i] = False

        # Step the simulator and sleep.
        if (bipd.useRealTimeSim == 0):
            p.stepSimulation()

        # Rendering
        # region
        if bipd.rendering == 1:
            img_arr = p.getCameraImage(bipd.pixelWidth,
                                        bipd.pixelHeight,
                                        bipd.viewMatrix,
                                        bipd.projectionMatrix,
                                        shadow=1,
                                        lightDirection=[1, 1, 1],
                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
            proj_opengl = np.uint8(np.reshape(img_arr[2], (bipd.pixelHeight, bipd.pixelWidth, 4)))
            # frame = cv2.resize(proj_opengl,(bipd.pixelWidth,bipd.pixelHeight))
            frame = cv2.cvtColor(proj_opengl, cv2.COLOR_RGB2BGR)
            bipd.out.write(frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
                break
        # endregion
        time.sleep(0.008)

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

    # Rendering
    # region
    '''
    if bipd.rendering == 1:
        bipd.out.release()
        cv2.destroyAllWindows()
    '''
    # endregion

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
        # fig.clear()
        # plt.close(fig)

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
        # fig2.clear()
        # plt.close(fig2)
        # fig3.clear()
        # plt.close(fig3)

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
        # fig4.clear()
        # plt.close(fig4)
        # fig5.clear()
        # plt.close(fig5)
    # endregion

    if isRecordValues == True:
        torso_position = np.append(external_force_applied.T, j_position_arr[0:3,:], axis=0)
        torso_orientation = np.append(external_force_applied.T, torso_orientation_arr, axis=0)
        torso_translation = np.append(external_force_applied.T, j_velocity_arr[0:3,:], axis=0)
        torso_rotation = np.append(external_force_applied.T, j_velocity_arr[3:6,:], axis=0)
        com_position = np.append(external_force_applied.T,com_arr, axis=0)
        j_position = np.append(external_force_applied.T, j_position_arr[7:,:], axis=0)
        j_velocity = np.append(external_force_applied.T, j_velocity_arr[6:,:], axis=0)
        cop_data = np.append(external_force_applied.T, cop_arr[0:2,:], axis=0)
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
        config.set('Data', 'force magnitude', force_magnitude)
        config.set('Data', 'force direction', [force_direction[0], force_direction[1]] )
        config.set('Data', 'force height', force_position_z)
        config.set('Data', 'fall', robotFall)
        if doPrediction == True:
            config.set('Data', 'false predict', false_predict)
            config.set('Data', 'gain balance', gain_balance)
            config.set('Data', 'lead time', fall_predict_lead)
        with open(resultPath + '/dataInfo.json', 'w') as configfile:
            config.write(configfile)

    # Print the final active force frames and the forces
    active_contact_frames, contact_forces, contact_cop,contact_cop_force, contact_cop_ft_l, contact_cop_ft_r=bipd.get_force()

    #print("Active force_frames:", force_frames)
    #print("Corresponding forces:", forces)
    print("iteration: ", j+1)
    j+=1

    np.savetxt(_resultPath + '/' + str_ + 'false_prediction_list.csv', false_prediction_list, delimiter=',')
