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

# from lstm import Predictor
# import torch
import sys

class Biped(PinBulletWrapper):
    def __init__(self, physicsClient=None, doFallSimulation = False, rendering = 0):
        if physicsClient is None:
            self.physicsClient = p.connect(p.GUI)
            # self.physicsClient = p.connect(p.GUI)


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