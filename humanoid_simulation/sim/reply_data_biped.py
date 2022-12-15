import os
import numpy as np
from numpy.linalg import norm, inv, pinv, eig, svd
import time
import matplotlib.pyplot as plt
from scipy.optimize import fmin_slsqp

import pybullet as p
import pybullet_data
import pinocchio as se3
from pinocchio.utils import *
import cv2
import json
import configparser

from py_pinocchio_bullet.wrapper import PinBulletWrapper
from robot_properties_solo.config import SoloConfig

from os.path import join, dirname

import py_lqr.model as lhm
import py_lqr.planner as pl
import py_lqr.trajectory_generation as tg


class Biped(PinBulletWrapper):
    def __init__(self, physicsClient=None, doFallSimulation = False, rendering = 0):
        if physicsClient is None:
            self.physicsClient = p.connect(p.GUI)

            # Setup for fall simulation, while setup walking, do turn this off!
            self.doFallSimulation = doFallSimulation
            p.setGravity(0, 0, -9.81)
            # p.setGravity(0,0,0)
            p.setPhysicsEngineParameter(
                fixedTimeStep=8.0/1000.0, numSubSteps=1)

            # Load the plain.

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            plain_urdf = (SoloConfig.packPath +
                      "/urdf/plane_with_restitution.urdf")
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
                ['j_leg_l_ankle_r', 'j_leg_r_ankle_r']
            )

            # Adjust view and close unrelevant window

            # region
            resultPath = str(os.path.abspath(os.path.join(self.pack_path, '../humanoid_simulation/result')))
            if self.useRealTimeSim == 1:
                p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, os.path.join(resultPath, 'biped_log.mp4'))
                p.setRealTimeSimulation(self.useRealTimeSim)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.resetDebugVisualizerCamera(cameraDistance=0.1,
                cameraYaw=160,
                cameraPitch=0,
                cameraTargetPosition=[0.9, 0.9, 0.45])

            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

            # rendering parameter
            if self.rendering == 1:
                
                self.camTargetPos = [0.9, 0.9, 0.45]
                self.cameraUp = [0, 0, 1]
                self.cameraPos = [1, 1, 1]
                self.yaw = 160.0
                self.pitch = 0.0
                self.roll = 0
                self.upAxisIndex = 2
                self.camDistance = 0.1
                self.pixelWidth = 1366
                self.pixelHeight = 768
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

if __name__ == "__main__":
    vec2list = lambda m: np.array(m.T).reshape(-1).tolist()
    np.set_printoptions(precision=2, suppress=True)

    # Setup simulation length
    horizon_length = 700

    # Setup simulation output
    isShowVelocityAccelerationResult = True
    isShowComResult = True
    isShowCopResult = True
    
    # Setup external force
    isApplyForce = True

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
    # endregion

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
    print('Simulator: Initial Feet Height:', pos)

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
    resultPath = str(os.path.abspath(os.path.join(
        bipd.pack_path, '../humanoid_simulation/data')))

    dire='2019_12_09/'
    num = 884
    json_dire= resultPath + '/' + dire + str(num) + '/dataInfo.json'
    config = configparser.RawConfigParser()
    config.read(json_dire)
    config.sections()
    force_magnitude = config.getfloat('Data', config.options('Data')[0])
    force_direction = json.loads( config.get('Data', config.options('Data')[1]) )
    force_height = config.getfloat('Data', config.options('Data')[2])


    robot = lhm.loadHumanoidModel(
        isDisplay=False, urdfPath=urdfPath, meshPath=meshPath)
    foot_width = robot.foot_width
    foot_length = robot.foot_length
    planner = pl.stepPlanner(robot)
    foot_print = planner.foot_print
    traj = tg.trajectoryGenerate(
        robot, planner, isDisplay=False, data_path=dataPath)
    simulator = tg.simulator(robot, traj, isRecompute=False,
                            isOnlineCompute=_isOnlineCompute)
    if _isOnlineCompute == False:
        joint_traj = simulator.joint_traj
        com_traj = simulator.com_traj
        cop_traj = simulator.cop_traj
        
    model = simulator.model
    data = simulator.data
    horizon_length = simulator.horizon_length + 100
    horizon_length_data = simulator.horizon_length
    print(horizon_length_data)
    torso_v_arr = np.zeros([3, horizon_length])
    torso_a_arr = np.zeros([4, horizon_length])
    com_arr = np.zeros([3, horizon_length])
    cop_arr = np.zeros([3, horizon_length])
    joint_traj_arr = np.zeros([np.size(joint_traj, 0),horizon_length])
    joint_traj_arr[:,0:np.size(joint_traj,1)] = joint_traj
    com_traj_arr = np.zeros([np.size(com_traj, 0),horizon_length])
    com_traj_arr[:,0:np.size(com_traj,1)] = com_traj
    cop_traj_arr = np.zeros([np.size(cop_traj, 0),horizon_length])
    cop_traj_arr[:,0:np.size(cop_traj,1)] = cop_traj
    # endregion

    # Run the simulator for 2000 steps = 2 seconds.

    for i in range(horizon_length):
        
        # Get the current state (position and velocity)
        q, dq = bipd.get_state()
        active_contact_frames, contact_forces, contact_cop = bipd.get_force()
        active_contact_frames_link, contact_forces_link = bipd.get_force_link()
        cop_arr[:, i] = contact_cop

        # Get the current acceleraton 
        # region
        torso_v_arr[:, i] = dq[0:3].transpose()

        if i > 0:
            torso_a_arr[:3, i] = (torso_v_arr[:, i] -
                                  torso_v_arr[:, i-1])/0.008
            torso_a_arr[3, i] = np.sqrt(
                torso_a_arr[0, i]**2 + torso_a_arr[1, i]**2 + torso_a_arr[2, i]**2)
        # endregion

        # Alternative, if you want to use properties from the pinocchio robot
        # like the jacobian or similar, you can also get the state and update
        # the pinocchio internals with one call:
        #
        q, dq = bipd.get_state_update_pinocchio()

        # Get the current center of mass
        com = se3.centerOfMass(bipd.pinocchio_robot.model,
                               bipd.pinocchio_robot.data, q)
        com_arr[:, i] = vec2list(com)

        if i % 100 == 0:
            print('Simulator: Forces from EndEffectors:', active_contact_frames, contact_forces)
            print('Simulator: Forces from Links:', active_contact_frames_link, contact_forces_link)

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

        if isApplyForce == True:
            # Apply external force
            force = [-force_magnitude * force_direction[0], -force_magnitude*force_direction[1] , 0]
            forcePos = [com[0], com[1], com[2]+force_height]  # define point in front of torso
            if i > 500 and i < 521:
                p.applyExternalForce(objectUniqueId=bipd.robotId, linkIndex=-1,forceObj=force, posObj=forcePos, flags=p.LINK_FRAME)
                # p.applyExternalForce(objectUniqueId=bipd.robotId, linkIndex=-1,forceObj=force, posObj=forcePos, flags=p.WORLD_FRAME)

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
            proj_opengl = np.uint8(np.reshape(
                img_arr[2], (bipd.pixelHeight, bipd.pixelWidth, 4)))

            # frame = cv2.resize(proj_opengl,(bipd.pixelWidth,bipd.pixelHeight))
            frame = cv2.cvtColor(proj_opengl, cv2.COLOR_RGB2BGR)

            bipd.out.write(frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
                break
        # endregion
        time.sleep(0.008)

    # Rendering
    # region
    if bipd.rendering == 1:
        bipd.out.release()
        cv2.destroyAllWindows()
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
        fig3,ax = plt.subplots(1)
        print(foot_print)
        print(foot_length)
        print(foot_width)
        for i in range(np.size(foot_print, 1)):
            anchor_x=foot_print[0, i] - foot_length/2.
            anchor_y=foot_print[1, i] - foot_width/2.
            rectangle=plt.Rectangle((anchor_x, anchor_y),
                                    foot_length, foot_width, linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rectangle)
        plt.plot(com_arr[0, :], com_arr[1, :])
        plt.plot(com_traj_arr[0, :], com_traj_arr[1, :])
        plt.savefig(os.path.join(resultPath, 'com_check_xy.png'))

    if isShowCopResult == True:
        fig4=plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(horizon_length), cop_traj_arr[0, :])
        plt.plot(np.arange(horizon_length), cop_arr[0, :])
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(horizon_length), cop_traj_arr[1, :])
        plt.plot(np.arange(horizon_length), cop_arr[1, :])
        # generate boundary of feet
        plt.plot([0, horizon_length], [0.10125, 0.10125], 'k-', lw=2)
        plt.plot([0, horizon_length], [0.02125, 0.02125], 'k-', lw=2)
        plt.plot([0, horizon_length], [-0.10125, -0.10125], 'k-', lw=2)
        plt.plot([0, horizon_length], [-0.02125, -0.02125], 'k-', lw=2)

        plt.subplot(3, 1, 3)
        plt.plot(np.arange(horizon_length), cop_traj_arr[2, :])
        plt.plot(np.arange(horizon_length), cop_arr[2, :])
        plt.savefig(os.path.join(resultPath, 'cop_check_time.png'))
        fig5,ax = plt.subplots(1)
        for i in range(np.size(foot_print, 1)):
            anchor_x=foot_print[0, i] - foot_length/2.
            anchor_y=foot_print[1, i] - foot_width/2.
            rectangle=plt.Rectangle((anchor_x, anchor_y),
                                    foot_length, foot_width, linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rectangle)
        plt.plot(cop_arr[0, :], cop_arr[1, :])
        plt.plot(cop_traj_arr[0, :], cop_traj_arr[1, :])
        plt.savefig(os.path.join(resultPath, 'cop_check_xy.png'))
    # endregion

    # Print the final active force frames and the forces
    force_frames, forces, contact_cop=bipd.get_force()

    print("Active force_frames:", force_frames)
    print("Corresponding forces:", forces)
