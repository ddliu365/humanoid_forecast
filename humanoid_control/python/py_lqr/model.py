import numpy as np
import os
import pinocchio as pin
from pinocchio.utils import *


def m2a(m): return np.array(m.flat)


def a2m(a): return np.matrix(a).T


class loadHumanoidModel():
    def __init__(self, urdfPath="urdf/biped.urdf", meshPath = "meshes/",isDisplay=False):

        # Model path
        mesh_dir = meshPath
        urdf_model_path = urdfPath

        # Import model
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_model_path, [mesh_dir], pin.JointModelFreeFlyer())  # floating base
        self.model = self.robot.model
        self.data = self.model.createData()

        #################################################################
        ###################INTERNAL PARAMETER############################
        #################################################################

        self.isDisplay = isDisplay

        self.step_width = 0.
        self.com = np.array([0., 0., 0.])

        self.idRfoot = 0.
        self.idLfoot = 0.
        self.idTorso = 0.
        self.q = self.robot.q0
        self.h_r_foot = 0.
        self.f_r_foot = 0.
        self.foot_length = 0.118
        self.foot_width = 0.08
        self.g = 9.81

        #################################################################
        ###################EXTERNAL PARAMETER############################
        #################################################################

        ###################PLANNER#######################################
        self.step_num = 6
        self.step_length = 0.118
        self.step_height = 0.02  # online:0.32
        self.step_height_r = 0.02  # online:0.32

        # Step offset in y direction
        # Positive means outward; Negative means inward
        self.step_offset_y = 0.

        # Step offset in x direciton
        # Positive means forward; Negative means backward
        self.step_offset_x = 0.

        # contains the bound calculated from foot
        self.cop_safe = 0.057

        # CoP offset in y direction
        # Positive means outward; Negative means inward
        self.cop_offset_y = -0.01

        # CoP offset in x direction
        # Positive means forward; Negative means backward
        self.cop_offset_x = self.foot_length/2 - self.cop_safe

        # self.com_offset_traj_y = 0.01
        # compensate com is not exactly at 0 in y direction
        # compensate cop traj for com
        self.cop_offset_for_com_y = 0.
        # compensate com is at the left part
        self.cop_offset_left = 0.

        # compensate for com in height in real robot in LQR
        self.com_0ffset_z = 0.

        ###################TRAJECTORY############################
        self.delta_t = 0.008
        self.num_iter_ss_init = 75  # online:300; offline
        # For a single support step: T_single = num_iter_ss * delta_t
        # number of iterations for a single support step
        self.num_iter_ss = 75  # online:290; offline:

        # For last step minor change
        self.num_iter_ss_end = 75  # online:180; offline:

        # number of iterations for initial double support step
        self.num_iter_ds_init = 150  # online:300; offline:

        # For a double support step: T_double = num_iter_ds * delta_t
        # number of iterations for a double support step

        self.num_iter_ds = 25  # online:250; offline:
        self.num_iter_ds_end = 150 # online:200; offline:

        # When putting foot down, keep it's original pose before next action
        self.num_step_delay = 3  # online:10;offline:

        self.horizon_length = (self.step_num-2)*self.num_iter_ss + \
            (self.step_num-1)*self.num_iter_ds + \
            self.num_step_delay*self.step_num +\
            self.num_iter_ds_init +\
            self.num_iter_ds_end +\
            self.num_iter_ss_end +\
            self.num_iter_ss_init

        # offset z in com for orientation task
        self.com_z_offset = 0.

        ###################PID##################################
        self.enablePID = False
        self.enableIMU = True

        self.p_roll = 0.
        self.i_roll = 0.
        self.d_roll = 0.

        self.p_pitch = 0.
        self.i_pitch = 0.
        self.d_pitch = 0.

        ###################LQR##################################
        self.Q = np.array([[1e7, 0],
                           [0, 1e7]
                           ])  # q0 = 150 which can work # online: 0.55, 0.4

        self.R = np.array([[1., 0],
                           [0, 1.]])

        # Initialization
        #self.cal_step_width()
        #self.cal_com(self.q)
        self.initDisplay()
        self.cal_step_width()
        self.cal_com(self.q)

    def initDisplay(self):
        _isDisplay = self.isDisplay

        # convert matrix and array for pinocchio

        if _isDisplay == True:
            self.robot.initViewer(loadModel=True)
        ############################################################################
        ###################TEST Joint Positive Direction############################
        ############################################################################
        '''
        self.q[7,0] = 20./180.* 3.14
        self.q[8,0] = -20./180.* 3.14
        # bend knee to avoid singularity
        self.q[9,0] = 15./180* 3.14  #leg_left_hip_pitch
        self.q[10,0] = -30./180.* 3.14 #leg_left_knee
        self.q[11,0] = -15./180.*3.14 #leg_left_ankle_pitch
        self.q[12,0] = 20./180.* 3.14

        self.q[13,0] = -20./180.* 3.14
        self.q[14,0] = -20./180.* 3.14
        self.q[15,0] = 15./180* 3.14 #leg_right_hip_pitch
        self.q[16,0] = -30./180.* 3.14 #leg_right_knee
        self.q[17,0] = -15./180.* 3.14 #leg_right_ankle_pitch
        self.q[18,0] = -20./180.* 3.14
        '''
        ############################################################################
        ###################TEST Joint Positive Direction############################
        ############################################################################

        # Initial Joint Position
        
        self.q[7] = 0./180. * 3.14       # leg_left_hip_yaw
        self.q[8] = 0./180. * 3.14       # leg_left_hip_roll
        self.q[9] = 25./180 * 3.14       # leg_left_hip_pitch
        self.q[10] = -50./180. * 3.14    # leg_left_knee
        self.q[11] = -25./180.*3.14      # leg_left_ankle_pitch
        self.q[12] = 0./180. * 3.14      # leg_left_ankle_roll

        self.q[13] = 0./180. * 3.14      # leg_right_hip_yaw
        self.q[14] = 0./180. * 3.14      # leg_right_hip_roll
        self.q[15] = 25./180 * 3.14      # leg_right_hip_pitch
        self.q[16] = -50./180. * 3.14    # leg_right_knee
        self.q[17] = -25./180. * 3.14    # leg_right_ankle_pitch
        self.q[18] = 0./180. * 3.14      # leg_right_ankle_roll
        
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        if _isDisplay == True:
            self.robot.display(self.q)
            self.robot.viewer.gui.addFloor('hpp-gui/floor')
            self.print_info()

    def cal_step_width(self):
        self.idRfoot = self.model.getFrameId('leg_r_foot')
        self.idLfoot = self.model.getFrameId('leg_l_foot')
        self.idTorso = self.model.getFrameId('torso')
        disRfoot = m2a(self.data.oMf[self.idRfoot].translation.T)
        disLfoot = m2a(self.data.oMf[self.idLfoot].translation.T)
        self.step_width = abs(disLfoot[1]-disRfoot[1])

    def cal_com(self, q):

        # World coordinate of CoM
        _com = m2a(pin.centerOfMass(self.model, self.data, q))

        # CoM Z
        h_r_foot = m2a(self.data.oMf[self.idRfoot].translation.T)[2]
        self.com[2] = _com[2]

        # CoM X
        f_r_foot = m2a(self.data.oMf[self.idRfoot].translation.T)[0]
        self.com[0] = _com[0] - f_r_foot

        # CoM Y
        l_r_foot = m2a(self.data.oMf[self.idRfoot].translation.T)[1]
        l_l_foot = m2a(self.data.oMf[self.idLfoot].translation.T)[1]
        self.com[1] = _com[1] - (l_r_foot + l_l_foot)/2

        ##################Test##################

        print("Planner: World initial pose: Z of CoM:", _com[2])
        print("Planner: World initial pose: Z of right foot:", h_r_foot) # almost zero
        print("Planner: Using CoM height:", self.com[2])

        print("Planner: World initial pose: X of CoM:", _com[0])
        print("Planner: World initial pose: Z of right foot:", f_r_foot) # almost zero
        print("Planner: Using CoM forward:", self.com[0])

        print("Planner: World initial pose: Y of CoM:", _com[1])
        print("Planner: World initial pose: Y of center of two feet:", (l_r_foot+l_l_foot)/2) # almost zero
        print("Planner: Using CoM left_right:", self.com[1])
        print("Planner: Using step_width:", self.step_width)

        ##################Test##################

    def print_info(self):
        for i, j in enumerate(self.model.names):
            print(i, j)
