import os
from datetime import datetime
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
import configparser

from py_pinocchio_bullet.wrapper import PinBulletWrapper
from robot_properties_solo.config import BipedConfig

from os.path import join, dirname

import py_lqr.model as lhm
import py_lqr.planner as pl
import py_lqr.trajectory_generation as tg

import gc

from lstm import Predictor
import torch

se3.switchToNumpyMatrix()

class Biped(PinBulletWrapper):
    def __init__(self, physicsClient=None):
        
        if physicsClient is None:
            self.physicsClient = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.setPhysicsEngineParameter(
                fixedTimeStep=8.0/1000.0, numSubSteps=1)

            # Load the scene
            objs = p.loadSDF(BipedConfig.packPath + "/botlab/botlab.sdf", globalScaling=1.0)
            zero=[0.8,-0.1,1.0]
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
            print("converting y to z axis")
            for o in objs:
                pos,orn = p.getBasePositionAndOrientation(o)
                y2x = p.getQuaternionFromEuler([3.14/2.,0,3.14/2])
                newpos,neworn = p.multiplyTransforms(zero,y2x,pos,orn)
                p.resetBasePositionAndOrientation(o,newpos,neworn)

            # Load the robot
            robotStartPos = [0., 0, 0.0014875]
            robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

            self.urdf_path_pybullet = BipedConfig.urdf_path_pybullet
            self.pack_path = BipedConfig.packPath
            self.urdf_path = BipedConfig.urdf_path
            self.robotId = p.loadURDF(self.urdf_path_pybullet, robotStartPos,
                                      robotStartOrientation, flags=p.URDF_USE_INERTIA_FROM_FILE,
                                      useFixedBase=False, globalScaling=1.0)
            p.getBasePositionAndOrientation(self.robotId)

            # Create the robot wrapper in pinocchio.
            package_dirs = [os.path.dirname(
                os.path.dirname(self.urdf_path)) + '/urdf']
            self.pin_robot = BipedConfig.buildRobotWrapper()

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
                                        ['j_leg_r_foot', 'j_leg_l_foot']
                                        )

            # Adjust view and close unrelevant window
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.resetDebugVisualizerCamera(cameraDistance=0.8,
                                         cameraYaw=120,
                                         cameraPitch=-15,
                                         cameraTargetPosition=[1.0, 0.0, 0.65])
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


if __name__ == "__main__":
    def vec2list(m): return np.array(m.T).reshape(-1).tolist()
    np.set_printoptions(precision=2, suppress=True)

    # Setup simulation output
    isShowVelocityAccelerationResult = True
    isShowComResult = True
    isShowCopResult = True
    isRecordValues = True

    # Setup external force
    isApplyForce = True
    force_direction_angles = np.linspace(-np.pi/2, np.pi/2, num=10)
    force_magnitude = np.linspace(300, 500, num=10)
    force_position_z = np.linspace(-0.1, 0.1, num=10)

    # Setup pybullet for the quadruped and a wrapper to pinocchio.
    bipd = Biped()
    bipd.useTorqueCtrl = False

    q, dq = bipd.get_state()

    k = 0
    l = 0
    _m = 0
    
    # Camera projection matrix
    fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)  

    for j in range(force_position_z.size * force_magnitude.size * force_direction_angles.size):
        gc.collect()
        # k is counter for force_direction_angles
        k = j % (force_magnitude.size *
                 force_position_z.size) % force_direction_angles.size
        if j > 0 and j % force_direction_angles.size == 0:
            # l is counter for force_magnitude
            l = (l+1) % force_magnitude.size
            if j % (force_magnitude.size * force_direction_angles.size) == 0:
                _m += 1  # m is counter for force_position_z

        force_direction = np.array(
            [np.cos(force_direction_angles[k]), np.sin(force_direction_angles[k])])
        q[7] = q[13] = 0
        q[8] = q[14] = 0
        q[9] = q[15] = 25./180*3.14
        q[10] = q[16] = -50./180*3.14
        q[11] = q[17] = -25./180*3.14
        q[12] = q[18] = 0

        # Reset environment
        if j > 0:
            q[0] = 0
            q[1] = 0
            q[2] = 0.0014875
            q[3] = 0
            q[4] = 0
            q[5] = 0
            q[6] = 1
            dq = np.zeros((18, 1))
            bipd.reset_state(q, dq)
            p.resetBaseVelocity(bipd.robotId, [0, 0, 0], [0, 0, 0])

        # Take the initial joint states as desired state.
        q_des = q[7:].copy()

        # Update the simulation state to the new initial configuration.
        bipd.reset_state(q, dq)

        # Call calculated trajectory
        _isOnlineCompute = False
        urdfPath = str(
            join(bipd.pack_path,
                 "urdf", "biped_new.urdf")
        )
        meshPath = str(bipd.pack_path)
        dataPath = str(os.path.abspath(os.path.join(
            bipd.pack_path, '../humanoid_control/data')))
        resultPath = str(os.path.abspath(os.path.join(
            bipd.pack_path, '../humanoid_simulation/data')))
        str_ = datetime.now().strftime("%Y_%m_%d")
        resultPath = resultPath + "/" + str_ + "/" + str(j+1)
        resultPath_pic = resultPath + "/pic"
        print(resultPath)
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)
            os.makedirs(resultPath_pic)
        else:
            import shutil
            print('WARNING: DIRECTORY EXISTS AND REMOVE IT IN 5 SECONDS...! Press Ctrl+C to Stop')
            time.sleep(5)
            shutil.rmtree(resultPath)
            os.makedirs(resultPath)
            os.makedirs(resultPath_pic)

        robot = lhm.loadHumanoidModel(
            isDisplay=False, urdfPath=urdfPath, meshPath=meshPath)
        foot_width = robot.foot_width
        foot_length = robot.foot_length
        planner = pl.stepPlanner(robot)
        foot_print = planner.foot_print
        traj = tg.trajectoryGenerate(
            robot, planner, isDisplay=False, data_path=dataPath)
        simulator = tg.simulator(
            robot, traj, isRecompute=False, isOnlineCompute=_isOnlineCompute)
        if _isOnlineCompute == False:
            joint_traj = simulator.joint_traj
            com_traj = simulator.com_traj
            cop_traj = simulator.cop_traj

        model = simulator.model
        data = simulator.data
        horizon_length = simulator.horizon_length + 300
        horizon_length_data = simulator.horizon_length

        # planned trajectory
        joint_traj_arr = np.zeros([np.size(joint_traj, 0), horizon_length])
        joint_traj_arr[:, 0:np.size(joint_traj, 1)] = joint_traj
        com_traj_arr = np.zeros([np.size(com_traj, 0), horizon_length])
        com_traj_arr[:, 0:np.size(com_traj, 1)] = com_traj
        cop_traj_arr = np.zeros([np.size(cop_traj, 0), horizon_length])
        cop_traj_arr[:, 0:np.size(cop_traj, 1)] = cop_traj

        # simulated result trajectory
        torso_v_arr = np.zeros([3, horizon_length])
        torso_a_arr = np.zeros([4, horizon_length])
        com_arr = np.zeros([3, horizon_length])
        cop_arr = np.zeros([3, horizon_length])
        torso_orientation_arr = np.zeros([3, horizon_length])
        j_position_arr = np.zeros([19, horizon_length])
        j_velocity_arr = np.zeros([18, horizon_length])
        external_force_applied = np.zeros((horizon_length, 1))
        linear_momentum_arr = np.zeros([3, horizon_length])
        angular_momentum_arr = np.zeros([3, horizon_length])
        linear_momentum_deri_arr = np.zeros([3, horizon_length])
        angular_momentum_deri_arr = np.zeros([3, horizon_length])
        dq0 = np.zeros(18)

        for i in range(horizon_length):
            # Get the current state (position and velocity)
            q, dq = bipd.get_state()
            active_contact_frames, contact_forces, contact_cop,contact_cop_force, contact_cop_ft_l, contact_cop_ft_r = bipd.get_force()
            active_contact_frames_link, contact_forces_link = bipd.get_force_link()

            q, dq = bipd.get_state_update_pinocchio()
            # Get the current center of mass
            com = se3.centerOfMass(
                bipd.pinocchio_robot.model, bipd.pinocchio_robot.data, q)
            ddq = (vec2list(dq)-dq0)/0.008
            dq0 = np.array(vec2list(dq))
            # Get current linear momentum and angular momentum around CoM
            se3.computeCentroidalMomentum(
                bipd.pinocchio_robot.model, bipd.pinocchio_robot.data, q, dq)
            se3.computeCentroidalMomentumTimeVariation(
                bipd.pinocchio_robot.model, bipd.pinocchio_robot.data, q, dq, ddq)

            # Get the current acceleraton
            cop_arr[:, i] = contact_cop
            torso_v_arr[:, i] = dq[0:3].transpose()

            # We update image every 32ms, Image updating frequency is 30Hz
            if i % 4 ==0:
                # Center of mass position and orientation (of link-7)
                com_p, com_o, _, _, _, _ = p.getLinkState(bipd.robotId, 2, computeForwardKinematics=True)
                rot_matrix = p.getMatrixFromQuaternion(com_o)
                rot_matrix = np.array(rot_matrix).reshape(3, 3)
                # Initial vectors
                init_camera_vector = (1, 0, 0) # x-axis
                init_up_vector = (0, 0, 1) # z-axis
                # Rotated vectors
                camera_vector = rot_matrix.dot(init_camera_vector)
                up_vector = rot_matrix.dot(init_up_vector)
                view_matrix = p.computeViewMatrix(com_p + 1 * camera_vector, com_p + 10 * camera_vector, up_vector)

                # NOTE: This function consumes about 220ms in this loop
                img = p.getCameraImage(512, 360, view_matrix, projection_matrix)
                cv2.imwrite(resultPath_pic + "/" + str(i//4) +'.jpg', img[2])
                print('record camera no. %d'%(i//4))
            if i > 0:
                torso_a_arr[:3, i] = (
                    torso_v_arr[:, i] - torso_v_arr[:, i-1])/0.008
                torso_a_arr[3, i] = np.sqrt(
                    torso_a_arr[0, i]**2 + torso_a_arr[1, i]**2 + torso_a_arr[2, i]**2)

            # com orientation is actually torso orientation
            torso_orientation_arr[:, i] = p.getEulerFromQuaternion(q[3:7])
            j_position_arr[:, i] = q.T
            j_velocity_arr[:, i] = dq.T
            com_arr[:, i] = vec2list(com)
            linear_momentum_arr[:, i] = vec2list(
                bipd.pinocchio_robot.data.hg.linear)
            angular_momentum_arr[:, i] = vec2list(
                bipd.pinocchio_robot.data.hg.angular)
            linear_momentum_deri_arr[:, i] = vec2list(
                bipd.pinocchio_robot.data.dhg.linear)
            angular_momentum_deri_arr[:, i] = vec2list(
                bipd.pinocchio_robot.data.dhg.angular)

            # Send the commands to the robot.
            if i < horizon_length_data:
                q_des = joint_traj[:, i]
            else:
                q_des = joint_traj[:, horizon_length_data-1]
            bipd.send_joint_command(q_des)

            if isApplyForce == True and i > 500 and i < 502:
                # Apply external force
                force = [-10. * force_direction[0], -force_magnitude[l] * force_direction[1], 0]
                # force = [-100., 0, 0]
                # define point in front of torso
                forcePos = [com[0], com[1], com[2] + force_position_z[_m]]
                p.applyExternalForce(objectUniqueId=bipd.robotId, linkIndex=-1,
                                     forceObj=force, posObj=forcePos, flags=p.WORLD_FRAME)
                external_force_applied[i] = True
            else:
                external_force_applied[i] = False

            p.stepSimulation()
        
        if com_arr[2, -1] < 0.2:
            robotFall = True
        else:
            robotFall = False

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
            fig2 = plt.figure()
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
            fig3, ax = plt.subplots(1)
            print(foot_print)
            print(foot_length)
            print(foot_width)
            for i in range(np.size(foot_print, 1)):
                anchor_x = foot_print[0, i] - foot_length/2.
                anchor_y = foot_print[1, i] - foot_width/2.
                rectangle = plt.Rectangle(
                    (anchor_x, anchor_y), foot_length, foot_width, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rectangle)
            plt.plot(com_arr[0, :], com_arr[1, :])
            plt.plot(com_traj_arr[0, :], com_traj_arr[1, :])
            plt.savefig(os.path.join(resultPath, 'com_check_xy.png'))
            fig2.clear()
            plt.close(fig2)
            fig3.clear()
            plt.close(fig3)

        if isShowCopResult == True:
            fig4 = plt.figure()
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
            fig5, ax = plt.subplots(1)
            for i in range(np.size(foot_print, 1)):
                anchor_x = foot_print[0, i] - foot_length/2.
                anchor_y = foot_print[1, i] - foot_width/2.
                rectangle = plt.Rectangle(
                    (anchor_x, anchor_y), foot_length, foot_width, linewidth=1, edgecolor='b', facecolor='none')
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
            torso_position = np.append(
                external_force_applied.T, j_position_arr[0:3, :], axis=0)
            torso_orientation = np.append(
                external_force_applied.T, torso_orientation_arr, axis=0)
            torso_translation = np.append(
                external_force_applied.T, j_velocity_arr[0:3, :], axis=0)
            torso_rotation = np.append(
                external_force_applied.T, j_velocity_arr[3:6, :], axis=0)
            com_position = np.append(external_force_applied.T, com_arr, axis=0)
            j_position = np.append(
                external_force_applied.T, j_position_arr[7:, :], axis=0)
            j_velocity = np.append(
                external_force_applied.T, j_velocity_arr[6:, :], axis=0)
            cop_data = np.append(external_force_applied.T,
                                 cop_arr[0:2, :], axis=0)
            linear_momentum_data = np.append(
                external_force_applied.T, linear_momentum_arr[0:3, :], axis=0)
            angular_momentum_data = np.append(
                external_force_applied.T, angular_momentum_arr[0:3, :], axis=0)
            linear_momentum_deri_data = np.append(
                external_force_applied.T, linear_momentum_deri_arr[0:3, :], axis=0)
            angular_momentum_deri_data = np.append(
                external_force_applied.T, angular_momentum_deri_arr[0:3, :], axis=0)

            np.savetxt(resultPath + '/torso_position.csv',
                       torso_position, delimiter=',')
            np.savetxt(resultPath + '/torso_orientation.csv',
                       torso_orientation, delimiter=',')
            np.savetxt(resultPath + '/torso_translation.csv',
                       torso_translation, delimiter=',')
            np.savetxt(resultPath + '/torso_rotation.csv',
                       torso_rotation, delimiter=',')
            np.savetxt(resultPath + '/com_position.csv',
                       com_position, delimiter=',')
            np.savetxt(resultPath + '/joint_position.csv',
                       j_position, delimiter=',')
            np.savetxt(resultPath + '/joint_velocity.csv',
                       j_velocity, delimiter=',')
            np.savetxt(resultPath + '/cop_data.csv', cop_data, delimiter=',')
            np.savetxt(resultPath + '/linear_momentum_data.csv',
                       linear_momentum_data, delimiter=',')
            np.savetxt(resultPath + '/angular_momentum_data.csv',
                       angular_momentum_data, delimiter=',')
            np.savetxt(resultPath + '/linear_momentum_deri_data.csv',
                       linear_momentum_deri_data, delimiter=',')
            np.savetxt(resultPath + '/angular_momentum_deri_data.csv',
                       angular_momentum_deri_data, delimiter=',')

            config = configparser.RawConfigParser()
            config.add_section('Data')
            config.set('Data', 'force magnitude', force_magnitude[l])
            config.set('Data', 'force direction', [
                       force_direction[0], force_direction[1]])
            config.set('Data', 'force height', force_position_z[_m])
            config.set('Data', 'fall', robotFall)
            with open(resultPath + '/dataInfo.json', 'w') as configfile:
                config.write(configfile)        
            
        print("iteration: ", j+1)
