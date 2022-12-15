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
from robot_properties_solo.config import SoloConfig

from os.path import join, dirname

import py_lqr.model as lhm
import py_lqr.planner as pl
import py_lqr.trajectory_generation as tg


class Biped(PinBulletWrapper):
    def __init__(self, physicsClient=None, doFallSimulation = False, rendering = 0):
        if physicsClient is None:
            self.physicsClient = p.connect(p.DIRECT)

            # Setup for fall simulation, while setup walking, do turn this off!
            self.doFallSimulation = doFallSimulation
            p.setGravity(0, 0, -9.81)
            # p.setGravity(0,0,0)
            p.setPhysicsEngineParameter(fixedTimeStep=5.0/1000.0, numSubSteps=1)

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
                    angularDamping=0.04, restitution=0., lateralFriction=1)

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
            resultPath = str(os.path.abspath(os.path.join(self.pack_path, '../humanoid_simulation/data')))

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
    isRecordValues = True
    
    # Setup external force
    isApplyForce = True
    force_direction_angles = np.linspace(-np.pi/2, np.pi/2, num = 10)
    force_magnitude = np.linspace(0, 200, num = 10)
    force_position_z = np.linspace(-0.1, 0.1, num = 10)
    
    # Setup pybullet for the quadruped and a wrapper to pinocchio.
    bipd = Biped(doFallSimulation = False, rendering = 0)
    bipd.rendering = 0
    bipd.useTorqueCtrl = False

    # Get the current state and modify the joints to have the legs
    # bend inwards.
    q, dq = bipd.get_state()

    k = 0
    l = 0
    m = 0

    # endregion
    for j in range(force_position_z.size * force_magnitude.size * force_direction_angles.size):
        k = j%(force_magnitude.size * force_position_z.size)%force_direction_angles.size # k is counter for force_direction_angles
        if j>0 and j%force_direction_angles.size == 0:
            l = (l+1)%force_magnitude.size # l is counter for force_magnitude
            if j%(force_magnitude.size * force_direction_angles.size) == 0:
                m += 1 # m is counter for force_position_z

        force_direction = np.array([ np.cos(force_direction_angles[k]), np.sin(force_direction_angles[k]) ])
        
        # Setup initial condition
        q[7] = q[13] = 0
        q[8] = q[14] = 0
        q[9] = q[15] = 25./180*3.14
        q[10] = q[16] = -50./180*3.14
        q[11] = q[17] = -25./180*3.14
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

        v_h = 0.
        v_v = 0.
        dq[0] = v_h
        dq[1] = v_v
    
        # Take the initial joint states as desired state.
        q_des = q[7:]

        # Update the simulation state to the new initial configuration.
        bipd.reset_state(q, dq)

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
        str_=datetime.now().strftime("%Y_%m_%d")
        resultPath = resultPath + "/" + str_ + "/" + str(j+1)
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
        torso_orientation = np.zeros([3,horizon_length])
        external_force_applied = np.zeros((horizon_length,1))

        # endregion
        # Run the simulator for 2000 steps = 2 seconds.

        for i in range(horizon_length):
        # Get the current state (position and velocity)
            q, dq = bipd.get_state()
            active_contact_frames, contact_forces, contact_cop = bipd.get_force()
            active_contact_frames_link, contact_forces_link = bipd.get_force_link()
            cop_arr[:, i] = contact_cop

            # Alternative, if you want to use properties from the pinocchio robot
            # like the jacobian or similar, you can also get the state and update
            # the pinocchio internals with one call:
        
            q, dq = bipd.get_state_update_pinocchio()

            torso_orientation[:,i] = p.getEulerFromQuaternion(q[3:7])

            # Get the current center of mass
            com = se3.centerOfMass(bipd.pinocchio_robot.model, bipd.pinocchio_robot.data, q)
            com_arr[:, i] = vec2list(com)

            # Send the commands to the robot.
            bipd.send_joint_command(q_des)

            if isApplyForce == True and i > 500 and i < 521:
                # Apply external force
                force = [-force_magnitude[l] * force_direction[0], -force_magnitude[l] * force_direction[1], 0]
                forcePos = [com[0], com[1], com[2] + force_position_z[m]]  # define point in front of torso           
                p.applyExternalForce(objectUniqueId=bipd.robotId, linkIndex=-1, forceObj=force, posObj=forcePos, flags=p.WORLD_FRAME)
                external_force_applied[i] = True
            else:
                external_force_applied[i] = False

            # Step the simulator and sleep.
            if (bipd.useRealTimeSim == 0):
                p.stepSimulation()

            # time.sleep(0.008)

        if com_arr[2,-1] < 0.2:
            robotFall = True
        else:
            robotFall = False

        if isRecordValues == True:
            com_position = np.append(external_force_applied.T, com_arr, axis=0)
            np.savetxt(resultPath + '/com_position.csv', com_position, delimiter=',')

            # torso_position = np.append(external_force_applied.T, j_position[0:3,:], axis=0)
            # np.savetxt(resultPath + '/torso_position.csv', torso_position, delimiter=',')

            torso_orientation = np.append(external_force_applied.T, torso_orientation, axis=0)
            np.savetxt(resultPath + '/torso_orientation.csv', torso_orientation, delimiter=',')

            # torso_translation = np.append(external_force_applied.T, j_velocity[1:3,:], axis=0)
            # np.savetxt(resultPath + '/torso_translation.csv', torso_translation, delimiter=',')

            # torso_rotation = np.append(external_force_applied.T, j_velocity[3:6,:], axis=0)
            # np.savetxt(resultPath + '/torso_rotation.csv', torso_rotation, delimiter=',')

            # j_position = np.append(external_force_applied.T, j_position[7:,:], axis=0)
            # np.savetxt(resultPath + '/joint_position.csv', j_position, delimiter=',')

            # j_velocity = np.append(external_force_applied.T, j_velocity[6:,:], axis=0)
            # np.savetxt(resultPath + '/joint_velocity.csv', j_velocity, delimiter=',')
            
            cop_data = np.append(external_force_applied.T, cop_arr[:-1,:], axis=0)
            np.savetxt(resultPath + '/cop_data.csv', cop_data, delimiter=',')

            config = configparser.RawConfigParser()
            config.add_section('Data')
            config.set('Data', 'force magnitude', force_magnitude[l])
            config.set('Data', 'force direction', [force_direction[0], force_direction[1]] )
            config.set('Data', 'force height', force_position_z[m])
            config.set('Data', 'fall', robotFall)
            with open(resultPath + '/dataInfo.json', 'w') as configfile:
                config.write(configfile)

        print("Iteration count: ", j+1)