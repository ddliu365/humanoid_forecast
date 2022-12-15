import time
import sys 
py_version = sys.version_info[0]
import IPython
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import importlib
from mpl_toolkits import mplot3d
from numpy.linalg import eig, inv, matrix_rank, norm, pinv, svd
from pinocchio.utils import *
import math
import os
if py_version == 2:
    import lqr as lqr
    import generator as generator
    import PID
else: 
    from .lqr import lqr as lqr
    from .generator import generator as generator
    from .PID import PID

def m2a(m): return np.array(m.flat)
def a2m(a): return np.matrix(a).T


class trajectoryGenerate():
    def __init__(self, robot, planner,isDisplay = False, data_path = '/home/hoon/Desktop/humanoid/bipedsim'):

        self.isDisplayResult = isDisplay

        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.foot_length = robot.foot_length
        self.foot_width = robot.foot_width
        self.com_height = robot.com[2]
        self.step_height = robot.step_height
        self.step_height_r = robot.step_height_r
        self.step_num = robot.step_num
        self.delta_t = robot.delta_t
        self.num_iter_ss_init = robot.num_iter_ss_init
        self.num_iter_ss = robot.num_iter_ss
        self.num_iter_ss_end = robot.num_iter_ss_end
        self.num_iter_ds = robot.num_iter_ds
        self.num_iter_ds_init = robot.num_iter_ds_init
        self.num_iter_ds_end = robot.num_iter_ds_end
        self.horizon_length = robot.horizon_length
        self.planner = planner
        self.cop_mat = planner.cop_mat
        self.num_footprint = planner.num_footprint
        self.step_mat = planner.step_mat
        self.Traj_left_foot = np.zeros([3, self.horizon_length+1])
        self.Traj_right_foot = np.zeros([3, self.horizon_length+1])
        self.cop_offset_x = robot.cop_offset_x
        self.cop_offset_y = robot.cop_offset_y
        self.num_step_delay = robot.num_step_delay
        self.cop_bar = np.zeros([3, self.horizon_length + 1])
        self.com_z_offset = robot.com_z_offset
        self.path_ = data_path
        self.generate_traj()

        # output
        # center of pressure trajectory
        
        self.v_com = np.zeros([3, self.horizon_length])
        self.com = np.zeros([3, self.horizon_length])
        # define feet trajectories & feet velocity trajectories


        self.Traj_vel_left_foot = np.zeros([6, self.horizon_length])
        self.Traj_vel_right_foot = np.zeros([6, self.horizon_length])
             
        self.generate_traj_com()
        self.display_trajectories()

        # trajectory for task
        # construct com task matrix
        self.w1 = self.v_com
        # construct feet task matrix
        self.w2 = self.Traj_vel_right_foot.copy()
        self.w3 = self.Traj_vel_left_foot.copy()

    def generate_traj(self):
        ##
        #step_l_s: single support step length
        step_l_s_init = self.num_iter_ss_init
        step_l_s = self.num_iter_ss
        step_l_s_end = self.num_iter_ss_end
        step_l_d = self.num_iter_ds
        step_l_d_init = self.num_iter_ds_init
        step_l_d_end = self.num_iter_ds_end
        step_delay = self.num_step_delay
        com_z_offset = self.com_z_offset

        cop_init = self.cop_mat[:,0]

        l_foot_init = self.step_mat[:,1]
        r_foot_init = self.step_mat[:,0]
        com_z_init = np.array([0,0,self.com_height])

        if py_version == 2:
            cop_generator = generator.generator(cop_init)
            l_foot_generator = generator.generator(l_foot_init,mid_point_z = self.step_height)
            r_foot_generator = generator.generator(r_foot_init,mid_point_z = self.step_height_r)
            com_z_generator =  generator.generator(com_z_init,mid_point_z = com_z_offset)
        else:
            cop_generator = generator(cop_init)
            l_foot_generator = generator(l_foot_init,mid_point_z = self.step_height)
            r_foot_generator = generator(r_foot_init,mid_point_z = self.step_height_r)
            com_z_generator =  generator(com_z_init,mid_point_z = com_z_offset)

        cop_generator.add(point = self.cop_mat[:,1], length= step_l_d_init)
        l_foot_generator.add(length = step_l_d_init)
        r_foot_generator.add(length = step_l_d_init)
        com_z_generator.add(z = -com_z_offset,length = step_l_d_init)


        cop_generator.add(length = step_delay)
        l_foot_generator.add(length = step_delay)
        r_foot_generator.add(length = step_delay)
        com_z_generator.add(length = step_delay)

        cop_generator.add(point = self.offset_point(self.cop_mat[:,1],x = self.cop_offset_x),length= step_l_s_init)
        l_foot_generator.add(length = step_l_s_init)
        r_foot_generator.add(point = self.step_mat[:,2],isCurve = True, length = step_l_s_init,ratio = 0.5)
        com_z_generator.add(length = step_l_s_init)

        cop_generator.add(length = step_delay)
        l_foot_generator.add(length = step_delay)
        r_foot_generator.add(length = step_delay)
        com_z_generator.add(length = step_delay)

        cop_generator.add(point = self.offset_point(self.cop_mat[:,2],x = -self.cop_offset_x),length = step_l_d)
        l_foot_generator.add(length = step_l_d)
        r_foot_generator.add(length = step_l_d)
        com_z_generator.add(isCurve = True, length = step_l_d,ratio =0.5)

        if self.step_num >2:
            for i in range(1, self.step_num-1):
                cop_generator.add(point = self.offset_point(self.cop_mat[:,i+1],x = self.cop_offset_x),length = step_l_s)
                cop_generator.add(length = step_delay)
                cop_generator.add(point = self.offset_point(self.cop_mat[:,i+2],x = -self.cop_offset_x), length = step_l_d)

                com_z_generator.add(length = step_l_s)
                com_z_generator.add(length = step_delay)
                com_z_generator.add(isCurve = True, length = step_l_d, ratio = 0.5)
        # generate feet trajectories
        id_l = 1
        id_r = 2
        flag = 0
        # define other trajectories
        if self.step_num > 2:
            for i in range(1, self.step_num-1):
                if flag == 0:
                    # single support phase
                    l_foot_generator.add(point = self.step_mat[:, id_l+2], isCurve = True, length = step_l_s,ratio = 0.5)
                    r_foot_generator.add(length = step_l_s)

                    l_foot_generator.add(length = step_delay)
                    r_foot_generator.add(length = step_delay)

                    # double support phase
                    l_foot_generator.add(length = step_l_d)
                    r_foot_generator.add(length = step_l_d)
                    id_l = id_l + 2
                    flag = 1
                else:

                    # single support phase
                    l_foot_generator.add(length = step_l_s)
                    r_foot_generator.add(point = self.step_mat[:, id_r+2], isCurve = True, length = step_l_s, ratio = 0.5)                 
                    l_foot_generator.add(length = step_delay)
                    r_foot_generator.add(length = step_delay)

                    # double support phase
                    l_foot_generator.add(length = step_l_d)
                    r_foot_generator.add(length = step_l_d)
                    id_r = id_r + 2
                    flag = 0  
                


        cop_generator.add(point = self.cop_mat[:,self.num_footprint-2], length = step_l_s_end)
        cop_generator.add(point = self.cop_mat[:,self.num_footprint-1], length = step_l_d_end)

        com_z_generator.add(length = step_l_s_end)
        com_z_generator.add(length = step_l_d_end)
        if self.step_num % 2 == 0:
            l_foot_generator.add(point = self.step_mat[:, self.num_footprint-1], isCurve = True, length = step_l_s_end, ratio = 0.5)
            r_foot_generator.add(length = step_l_s_end)

            #l_foot_generator.add(length = step_delay)
            #r_foot_generator.add(length = step_delay)

            l_foot_generator.add(length = step_l_d_end)
            r_foot_generator.add(length = step_l_d_end)

        else:
            l_foot_generator.add(length = step_l_s_end)
            r_foot_generator.add(point = self.step_mat[:, self.num_footprint-1], isCurve = True, length = step_l_s_end, ratio = 0.5)

            #l_foot_generator.add(length = step_delay)
            #r_foot_generator.add(length = step_dealy)

            l_foot_generator.add(length = step_l_d_end)
            r_foot_generator.add(length = step_l_d_end)

        self.cop_bar = cop_generator.traj
        cop_data_path = os.path.join(self.path_, 'cop.npy')
        np.save(cop_data_path,self.cop_bar)
        self.Traj_left_foot = l_foot_generator.traj
        self.Traj_right_foot = r_foot_generator.traj
        self.com_z_traj = com_z_generator.traj
        print('Planner: any Z of CoM:',self.com_z_traj[2,301])
        self.horizon_length = np.size(self.cop_bar,axis = 1)

        '''
        fig = plt.figure(figsize=(12, 6.5))
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(cop_generator.traj[0,:],cop_generator.traj[1,:])
        plt.title('trajectory')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()
        '''
        
    def offset_point(self, x_init, x = 0, y= 0, z = 0):
        ##
        # x_init: current coordinate
        # x: offset in x direction 
        # y: offset in y direction
        # z: offset in z direction
        ##
        x_end = x_init.copy()
        x_end[0] = x_end[0] + x
        x_end[1] = x_end[1] + y
        x_end[2] = x_end[2] + z

        return x_end
    
    def generate_traj_com(self):

        if py_version == 2:
            _lqr = lqr.lqr(self.robot, self.cop_bar, self.isDisplayResult)
        else:
            _lqr = lqr(self.robot, self.cop_bar, self.isDisplayResult)

        X = _lqr.X

        # construct CoM velocity trajectories
        self.v_com[0, :] = X[1, :]
        self.v_com[1, :] = X[4, :]
        self.com[0, :] = X[0, :]
        self.com[1, :] = X[3, :]
        #self.com[2, :] = self.com_height
        self.com[2,:] = self.com_z_traj[2,:]
        com_data_path = os.path.join(self.path_, 'com.npy')
        np.save(com_data_path,self.com)
    
    def display_trajectories(self):

        if self.isDisplayResult == True:
            # display trajectories
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            # orange: right foot trajectory
            xline = self.Traj_right_foot[0, :]
            yline = self.Traj_right_foot[1, :]
            zline = self.Traj_right_foot[2, :]
            ax.plot3D(xline, yline, zline, 'orange')

            # blue: left foot trajectory
            xline = self.Traj_left_foot[0, :]
            yline = self.Traj_left_foot[1, :]
            zline = self.Traj_left_foot[2, :]
            ax.plot3D(xline, yline, zline, 'blue')

            # red: center of pressure trajectory
            xline = self.cop_bar[0, :]
            yline = self.cop_bar[1, :]
            zline = self.cop_bar[2, :]
            ax.plot3D(xline, yline, zline, 'red')

            # green: center of mass trajectory
            xline = self.com[0, :]
            yline = self.com[1, :]
            zline = self.com[2, :]
            ax.plot3D(xline, yline, zline, 'green')
            plt.show()


class simulator():
    def __init__(self, robot, traj_gene, isRecompute=True, isOnlineCompute = False):

        # isOnlineCompute = False: compute trajectory offline 
        # isOnlineComputer = True: compute trajectory online
        self.w1 = traj_gene.w1
        self.w2 = traj_gene.w2
        self.w3 = traj_gene.w3
        self.model = robot.model
        self.data = robot.data
        self.robot = robot.robot
        self.horizon_length = traj_gene.horizon_length
        self.q = robot.q
        self.q_des = robot.q.copy()
        self.delta_t = traj_gene.delta_t
        self.path_ = traj_gene.path_
        self.idRfoot = robot.idRfoot
        self.idLfoot = robot.idLfoot
        self.idTorso = robot.idTorso
        self.isDisplay = robot.isDisplay
        self.Traj_left_foot = traj_gene.Traj_left_foot
        self.Traj_right_foot = traj_gene.Traj_right_foot
        self.Traj_com = traj_gene.com
        self.isRecompute = isRecompute
        self.cop_bar = traj_gene.cop_bar
        self.enablePID = robot.enablePID
        self.enableIMU = robot.enableIMU
        self.p_pitch = robot.p_pitch
        self.p_roll = robot.p_roll
        self.i_pitch = robot.i_pitch
        self.i_roll = robot.i_roll
        self.d_pitch = robot.d_pitch
        self.d_roll = robot.d_roll

        self.desired_torso_pitch = 0.
        self.desired_torso_yaw = 0.
        self.desired_torso_roll = 0.

        self.com_tradeoff_height = 0.

        # joint trajectory: first 7 elements are torso position
        self.joint_traj = np.empty([self.model.nq-7, self.horizon_length+1])
        self.com_traj = np.empty([3, self.horizon_length +1])
        self.cop_traj = np.empty([3, self.horizon_length +1])
        # track ik iteration
        self.num_iter_ik = np.zeros([1, self.horizon_length + 1])
        # track ik error
        self.err_traj = np.zeros([3,self.horizon_length + 1])
        

        # state_traj: contains states for robot model
        #self.state_traj = np.empty([self.model.nq, self.horizon_length+1])

        # initialization
        #pin.forwardKinematics(self.model, self.data, self.q)
        #pin.computeJointJacobians(self.model, self.data, self.q)
        #pin.updateFramePlacements(self.model, self.data)

        # four tasks to construct
        # task1: follow the center of mass
        # task2: follow the left foot trajectory
        # task3: follow the right foot trajectory
        # task4: follow the orientation task

        # initialize com jacobian
        #self.J_com = pin.jacobianCenterOfMass(self.model, self.data, self.q)

        # initialize feet jacobian
        #self.J_foot_right = pin.getFrameJacobian(
        #    self.model, self.data, robot.idRfoot, pin.ReferenceFrame.WORLD)
        #self.J_foot_left = pin.getFrameJacobian(
        #    self.model, self.data, robot.idLfoot, pin.ReferenceFrame.WORLD)

        # initialize torso jacobian with orientation 
        #self.J_torso_rot = pin.getFrameJacobian(self.model, self.data, robot.idTorso, pin.ReferenceFrame.WORLD)[3:6,:]
        
        # initialize joint trajectory
        self.joint_traj[:, 0] = m2a(self.q)[7:]
        #self.state_traj[:, 0] = m2a(self.q)
        #self.foot_l_traj = np.zeros([3, self.horizon_length])
        #self.foot_r_traj = np.zeros([3, self.horizon_length])
        #self.com_sim_traj = np.zeros([3, self.horizon_length])
        #self.h_r_foot = robot.h_r_foot
        #self.f_r_foot = robot.f_r_foot

        # error_pitch_traj: track error infomation in torso(pitch direction)
        # error_roll_traj: track error information in torso(roll direction)
        self.error_pitch_traj = np.zeros([1,self.horizon_length])
        self.error_roll_traj = np.zeros([1,self.horizon_length])
        joint_traj_path = os.path.join(self.path_, 'joint_traj.npy')
        com_traj_path = os.path.join(self.path_, 'com.npy')
        cop_traj_path = os.path.join(self.path_, 'cop.npy')
        value_data_path = os.path.join(self.path_, 'values.csv')
        if isOnlineCompute == False:
            if self.isRecompute == True:

                self.cal_traj()
                
                np.save(joint_traj_path,self.joint_traj)
                np.savetxt(value_data_path, self.joint_traj, fmt="%f", delimiter=",")
            else:

                self.joint_traj = np.load(joint_traj_path)
                self.com_traj = np.load(com_traj_path)
                self.cop_traj = np.load(cop_traj_path)
                print("Load joint trajectory, com trajectory... Success!")

        '''
        
        if self.isRecompute == True:
            #self.cal_traj()
            #print(np.size(self.joint_traj,axis = 1))
            #print(self.horizon_length)
            #np.save('joint_traj.npy',self.joint_traj)
            #np.save('num_iter_ik.npy',self.num_iter_ik)
        else:
            #self.joint_traj = np.load('/Users/dongdong/proj/joint_traj.npy')
            self.joint_traj = np.load('/home/jack/catkin_ws/src/humanoid/beginner_tutorials/scripts/humanoid_control/joint_traj.npy')
            print("loading joint_traj done!")
            if self.isDisplay == True:
                for i in range(self.horizon_length):
                    self.q[7:] = a2m(self.joint_traj[:,i])
                    self.robot.display(self.q)
                    time.sleep(self.delta_t)
        '''

        '''
        if self.isRecompute == True:
            self.walk()
            np.save('joint_traj.npy', self.joint_traj)
            np.save('state_traj.npy', self.state_traj)
            if self.isDisplay == True: 
                self.showLeftFootTrajectories()
                self.showRightFootTrajectories()
                self.showComTrajectories()
                self.showCumulativeResult()
                self.showJointTraj()
                self.showCopTrajectories()
                # self.compareJointTraj()
        else:
            self.joint_traj = np.load('motor_traj.npy')
            self.joint_traj = self.joint_traj[:, 550:]
            state_joint = np.load('state_traj.npy')

            for i in range(self.horizon_length):
                self.q[7:] = a2m(self.joint_traj[:, i])
                if self.isDisplay == True:
                    # time.sleep(self.delta_t)
                    self.robot.display(self.q)
            # self.compareJointTraj()
            self.showTorqueTraj()
        '''
    # calculate null space projector N
    def getProjector(self, J):

        I = np.eye(len(J[0]))

        N = I - np.linalg.pinv(J).dot(J)

        return N
    
    # this function is for simulatio
    def update(self,body_pose,i):

        # body_pose: orientation information in quaternion form
        # i: counter: 0~ horizon_length

        self.joint_traj[:,i] = m2a(self.q)[7:]

        

        # save error
        if i == self.horizon_length -1:
            error_data_pitch_path = os.path.join(self.path_, "error_pitch.npy")
            error_data_roll_path = os.path.join(self.path_, "error_roll.npy")
            np.save(error_data_pitch_path, self.error_pitch_traj)
            np.save(error_data_roll_path, self.error_roll_traj)
            #np.save("/Users/dongdong/proj/err_traj.npy",self.err_traj)

        # Update Target
        #refTorso = np.matrix([ 0., 0., 0.]).T
        refTorso = eye(3)
        #refTorso = pin.SE3(eye(3), refTorso_pos)
        refCom = self.Traj_com[:, i]
        refCom[2] = refCom[2] - self.com_tradeoff_height
        # refLeftFoot = ?
        # refRightFoot = ?                 

        # Update refLeftFoot and refRightFoot
        if self.enablePID == True:

            # get body pose from external sensors:
            if self.enableIMU == False:

                # enableIMU = false: orientation of torso from model
                T_torso_ = self.data.oMf[self.idTorso]
            else:

                # enableIMU = true: orientation of torso from IMU
                #_q0,_q1,_q2,_q3 = self.toQuaternion(0,body_pitch,body_roll)
                XYZQUAT_torso = np.matrix([ 0,
                                            0,
                                            0,
                                            body_pose[0],
                                            body_pose[1],
                                            body_pose[2],
                                            body_pose[3]]).T 
                T_torso_ = pin.XYZQUATToSe3(XYZQUAT_torso)
                refTorso_pos = np.matrix([ 0., 0., 0.]).T
                refTorso_pid = pin.SE3(eye(3), refTorso_pos)
                tmp_torso = T_torso_.inverse()*refTorso_pid
                Vs_torso = m2a(pin.log(tmp_torso).vector)

                error_roll = Vs_torso[3]
                error_pitch = Vs_torso[4]

                self.error_pitch_traj[:,i] = error_roll
                self.error_roll_traj[:,i] = error_pitch

                print("error with roll", error_roll)
                print("error with pitch", error_pitch)

                p_roll = self.p_roll
                i_roll = self.i_roll
                d_roll = self.d_roll

                p_pitch = self.p_pitch
                i_pitch = self.i_pitch
                d_pitch = self.d_pitch

                pid_roll = PID.PID(p_roll, i_roll, d_roll)
                pid_pitch = PID.PID(p_pitch, i_pitch, d_pitch)

                pid_roll.SetPoint=0.0
                pid_pitch.SetPoint=0.0
                pid_roll.setSampleTime(self.delta_t)
                pid_pitch.setSampleTime(self.delta_t)

                pid_roll.update(error_roll)
                pid_pitch.update(error_pitch)

                # if torso pitch in positive direction, then 
                # the feet should pitch in positive direction to guarantee upright torso
                output_roll = -pid_roll.output
                output_pitch = -pid_pitch.output
                q0,q1,q2,q3 = self.toQuaternion(0,output_pitch,output_roll) # toQuaternion(yaw,pitch,roll)
                XYZQUAT_left = np.matrix([ self.Traj_left_foot[0, i],
                                                self.Traj_left_foot[1, i],
                                                self.Traj_left_foot[2, i],
                                                q0,
                                                q1,
                                                q2,
                                                q3]).T 
                XYZQUAT_right = np.matrix([ self.Traj_right_foot[0, i],
                                    self.Traj_right_foot[1, i],
                                    self.Traj_right_foot[2, i],
                                    q0,
                                    q1,
                                    q2,
                                    q3]).T 
                refLeftFoot = pin.XYZQUATToSe3(XYZQUAT_left)
                refRightFoot = pin.XYZQUATToSe3(XYZQUAT_right)

        # enablePID = false: no revise orientation of feet
        else:
            
            reflfoot = a2m(self.Traj_left_foot[:, i])
            refrfoot = a2m(self.Traj_right_foot[:, i])

            refLeftFoot = pin.SE3(eye(3), reflfoot)
            refRightFoot = pin.SE3(eye(3), refrfoot)

        #T_torso = np.zeros(3)

        #T_torso[0:2] = refCom[0:2]
        #T_torso[2] = m2a(self.data.oMf[self.idTorso].translation)[2]
        # print(T_torso)
        # print(m2a(self.data.oMf[self.idTorso].translation))
        #tmp_torso =a2m(T_torso)
        #refTorso = eye(3)
        #refTorso = pin.SE3(eye(3),a2m(T_torso))
        

        eomg = 1e-4
        ev = 1e-5
        ecom = 0.5e-5
        etorso = 1e-4
        j = 0
        maxiterations = 2000

        pin.forwardKinematics(self.model, self.data, self.q)
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        T_left_foot = self.data.oMf[self.idLfoot]
        T_right_foot = self.data.oMf[self.idRfoot]
        com = m2a(pin.centerOfMass(self.model, self.data, self.q))

        # print("",self.h_r_foot)
        # print(self.f_r_foot)

        tmp_left = T_left_foot.inverse()*refLeftFoot
        Vs_left = m2a(pin.log(tmp_left).vector)

        tmp_right = T_right_foot.inverse()*refRightFoot
        Vs_right = m2a(pin.log(tmp_right).vector)

        tmp_com = refCom - com
        residue_com = sum(tmp_com**2)

        T_torso_ = self.data.oMf[self.idTorso].rotation

        #tmp_torso = a2m(np.linalg.inv(T_torso_).dot(refTorso))
        #tmp_torso = T_torso_.inverse()*refTorso
        
        #Vs_torso = m2a(pin.log3(tmp_torso))
        Vs_torso = m2a(pin.log3(np.linalg.inv(T_torso_).dot(refTorso)))

        err_left_trans = np.linalg.norm([Vs_left[0], Vs_left[1], Vs_left[2]])
        err_left_orien = np.linalg.norm([Vs_left[3], Vs_left[4], Vs_left[5]])
        err_right_trans = np.linalg.norm([Vs_right[0], Vs_right[1], Vs_right[2]])
        err_right_orien = np.linalg.norm([Vs_right[3], Vs_right[4], Vs_right[5]])


        err = err_left_trans > eomg \
            or err_left_orien > ev or \
                err_right_trans > eomg \
            or  err_right_orien > ev or \
            residue_com > ecom  
        '''
        err = np.linalg.norm([Vs_left[0], Vs_left[1], Vs_left[2]]) > eomg or \
            np.linalg.norm([Vs_left[3], Vs_left[4], Vs_left[5]]) > ev or \
            np.linalg.norm([Vs_right[0], Vs_right[1], Vs_right[2]]) > eomg or\
            np.linalg.norm([Vs_right[3], Vs_right[4], Vs_right[5]]) > ev or \
            residue_com > ecom 
            #or\
            #np.linalg.norm([Vs_torso[0], Vs_torso[1], Vs_torso[2]]) > etorso
        '''
        epsilon = 0.1

        q_prev = self.q.copy()
        counter = 0

        while err and j < maxiterations:
            self.num_iter_ik[0,i] = j
            self.err_traj[0,i] = err_left_trans
            self.err_traj[1,i] = err_right_trans
            self.err_traj[2,i] = residue_com
            if j > 900:
                print("current step",)

                counter = counter + 1
                if counter == 50:
                    
                    refCom[2] = refCom[2] - self.com_tradeoff_height
                    self.com_tradeoff_height = self.com_tradeoff_height + 1e-3
                    tmp_com = refCom - com
                    counter = 0
                
            if j > 1800:
                # Once inverse kinematics failed
                # return last pose to prevent random movement
                self.q = q_prev
                print("Inverse Kinematics cannot be solved! Use previous value!")
                break

            # construct J matrix
            J = np.empty([15, self.model.nv])
            J_2 = pin.jacobianCenterOfMass(
                self.model, self.data, self.q)
            J[3:9, :] = pin.getFrameJacobian(
                self.model, self.data, self.idRfoot, pin.ReferenceFrame.WORLD)
            J[9:15, :] = pin.getFrameJacobian(
                self.model, self.data, self.idLfoot, pin.ReferenceFrame.WORLD)
            
            # com task added to null space
            # get torso Jacobian for optimization of torso pose
            #J[15:18, :] = pin.getFrameJacobian(
            #       self.model, self.data, self.idTorso, pin.ReferenceFrame.WORLD)[3:6,:]

            J[0:3, :] = pin.getFrameJacobian(self.model, self.data, self.idTorso, pin.ReferenceFrame.WORLD)[3:6,:]

            #J_2 = pin.getFrameJacobian(
            #    self.model, self.data, self.idTorso, pin.ReferenceFrame.WORLD)[3:6, :]

            '''
            Vs = np.zeros([15,1])
            Vs[0:3,0] = tmp_com 
            Vs[3:9,0] = Vs_right
            Vs[9:15,0] = Vs_left
            '''
            Vs = np.zeros([15, 1])
            
            Vs[3:9, 0] = Vs_right
            Vs[9:15, 0] = Vs_left
            #Vs[15:16, 0] = Vs_torso

            Vs_2 = np.zeros([3,1])
            Vs_2[0:3, 0] = tmp_com
            Vs[0:3,0] = Vs_torso
            # calculate delta q
            dq0 = pin.difference(self.model, self.q, self.q_des)
            #dq0 = np.linalg.pinv(J_2).dot(Vs_2)
            #dq = np.linalg.pinv(J).dot(Vs) + self.getProjector(J).dot(dq0)
            tmp_item = np.linalg.pinv(J_2.dot(self.getProjector(J)))
            tmp_item_2 = Vs_2 - J_2.dot(np.linalg.pinv(J)).dot(Vs)
            dq = np.linalg.pinv(J).dot(Vs) + self.getProjector(J).dot(tmp_item.dot(tmp_item_2))
            dq = dq * epsilon
            self.q = pin.integrate(self.model, self.q, dq)

            j = j + 1
            print('Planner: Working on inverse kinematics:', j)

            pin.forwardKinematics(self.model, self.data, self.q)
            pin.computeJointJacobians(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)

            T_left_foot = self.data.oMf[self.idLfoot]
            T_right_foot = self.data.oMf[self.idRfoot]
            com = m2a(pin.centerOfMass(self.model, self.data, self.q))
            #T_torso_ = self.data.oMf[self.idTorso]

            tmp_left = T_left_foot.inverse()*refLeftFoot
            Vs_left = m2a(pin.log(tmp_left).vector)

            tmp_right = T_right_foot.inverse()*refRightFoot
            Vs_right = m2a(pin.log(tmp_right).vector)

            tmp_com = refCom - com
            residue_com = sum(tmp_com**2)

            #tmp_torso = np T_torso_.inverse()*refTorso
            #Vs_torso = m2a(np.array([0, 0, 0]))

            #T_torso_ = self.data.oMf[self.idTorso].rotation

            #tmp_torso = a2m(np.linalg.inv(T_torso_).dot(refTorso))
            #tmp_torso = T_torso_.inverse()*refTorso
            
            #Vs_torso = m2a(pin.log3(tmp_torso))[0]
            err_left_trans = np.linalg.norm([Vs_left[0], Vs_left[1], Vs_left[2]])
            err_left_orien = np.linalg.norm([Vs_left[3], Vs_left[4], Vs_left[5]])
            err_right_trans = np.linalg.norm([Vs_right[0], Vs_right[1], Vs_right[2]])
            err_right_orien = np.linalg.norm([Vs_right[3], Vs_right[4], Vs_right[5]])


            err = err_left_trans > eomg \
                or err_left_orien > ev or \
                 err_right_trans > eomg \
                or  err_right_orien > ev or \
                residue_com > ecom    

    #calculate trajectory offline 
    def cal_traj(self):
        for i in range(self.horizon_length):
            # no pid 
            torso_pose = np.array([0,0,0,1])
            self.update(torso_pose,i)
            
    '''
    def walk(self):
        for i in range(self.horizon_length):
            # print(i)

            #Active balance module: maintain a upright torso position, by reducing variations of torso angles
            refTorso_pos = np.matrix([ 0., 0., 0.]).T 
            refTorso = pin.SE3(eye(3), refTorso_pos)

            T_torso_ = self.data.oMf[self.idTorso]
            tmp_torso = T_torso_.inverse()*refTorso
            Vs_torso = m2a(pin.log(tmp_torso).vector)

            error_roll = Vs_torso[3]
            error_pitch = Vs_torso[4]


            p_roll = 0.
            i_roll = 0.
            d_roll = 0.

            p_pitch = 0.
            i_pitch = 0.
            d_pitch = 0.

            pid_roll = PID.PID(p_roll, i_roll, d_roll)
            pid_pitch = PID.PID(p_pitch, i_pitch, d_pitch)

            pid_roll.SetPoint=0.0
            pid_pitch.SetPoint=0.0
            pid_roll.setSampleTime(0.008)
            pid_pitch.setSampleTime(0.008)

            pid_roll.update(error_roll)
            pid_pitch.update(error_pitch)

            # if torso pitch in positive direction, then 
            # the feet should pitch in positive direction to guarantee upright torso
            output_roll = -pid_roll.output
            output_pitch = -pid_pitch.output

            print(output_roll)
            print(output_pitch)
    '''
    '''
            if i ==2:
                return
    '''
    '''
            q0,q1,q2,q3 = self.toQuaternion(0,output_pitch,output_roll) # toQuaternion(yaw,pitch,roll)

            print("quaternion:",q0)
            print("quaternion:",q1)
            print("quaternion:",q2)
            print("quaternion:",q3)
            
            XYZQUAT_left = np.matrix([ self.Traj_left_foot[0, i],
                                            self.Traj_left_foot[1, i],
                                            self.Traj_left_foot[2, i],
                                            q0,
                                            q1,
                                            q2,
                                            q3]).T 
            XYZQUAT_right = np.matrix([ self.Traj_right_foot[0, i],
                                self.Traj_right_foot[1, i],
                                self.Traj_right_foot[2, i],
                                q0,
                                q1,
                                q2,
                                q3]).T 
            refLeftFoot = pin.XYZQUATToSe3(XYZQUAT_left)
            refRightFoot = pin.XYZQUATToSe3(XYZQUAT_right)

            # for debugging
            reflfoot = a2m(self.Traj_left_foot[:, i])
            refrfoot = a2m(self.Traj_right_foot[:, i])
            _refLeftFoot = pin.SE3(eye(3), reflfoot)       # Target position
            _refRightFoot = pin.SE3(eye(3), refrfoot)

            _left = pin.se3ToXYZQUAT(_refLeftFoot)
            _right = pin.se3ToXYZQUAT(_refRightFoot)


            print("converted xyzquat:",_left)
            print("converted xyzquat:",_right)
            print("current:",refLeftFoot)
            print("current:",refRightFoot)

            print("before:",_refLeftFoot)
            print("before:",_refRightFoot)

    '''
    '''
            reflfoot = a2m(self.Traj_left_foot[:, i])
            refrfoot = a2m(self.Traj_right_foot[:, i])
            # desired position
            refLeftFoot = pin.SE3(eye(3), reflfoot)       # Target position
            refRightFoot = pin.SE3(eye(3), refrfoot)
    '''
    '''
            refCom = self.Traj_com[:, i]                  # Target position

            #T_torso = np.zeros(3)

            #T_torso[0:2] = refCom[0:2]
            #T_torso[2] = m2a(self.data.oMf[self.idTorso].translation)[2]
            # print(T_torso)
            # print(m2a(self.data.oMf[self.idTorso].translation))
            #tmp_torso =a2m(T_torso)
            #refTorso = eye(3)
            #refTorso = pin.SE3(eye(3),a2m(T_torso))
            

            eomg = 1e-5
            ev = 1e-5
            ecom = 1e-6
            j = 0
            maxiterations = 200

            pin.forwardKinematics(self.model, self.data, self.q)
            pin.computeJointJacobians(self.model, self.data, self.q)
            pin.updateFramePlacements(self.model, self.data)

            T_left_foot = self.data.oMf[self.idLfoot]
            T_right_foot = self.data.oMf[self.idRfoot]
            com = m2a(pin.centerOfMass(self.model, self.data, self.q))

            # print("",self.h_r_foot)
            # print(self.f_r_foot)

            tmp_left = T_left_foot.inverse()*refLeftFoot
            Vs_left = m2a(pin.log(tmp_left).vector)

            tmp_right = T_right_foot.inverse()*refRightFoot
            Vs_right = m2a(pin.log(tmp_right).vector)

            tmp_com = refCom - com
            residue_com = sum(tmp_com**2)

            #T_torso_ = self.data.oMf[self.idTorso].rotation

            #tmp_torso = a2m(np.linalg.inv(T_torso_).dot(refTorso))
            #tmp_torso = T_torso_.inverse()*refTorso
            
            #Vs_torso = m2a(pin.log3(tmp_torso))[0]
    '''

    '''
            print("current left foot position:",T_left_foot)
            print("current right foot position:",T_right_foot)

            print("reference left foot position:",refLeftFoot)
            print("reference right foot position:",refRightFoot)

            print("left foot error:",Vs_left)
            print("right foot error:",Vs_right)
            
            print("reference com:",refCom)
            print("current com:",com)
            print("com error:",residue_com)
    '''
    '''
            err = np.linalg.norm([Vs_left[0], Vs_left[1], Vs_left[2]]) > eomg or \
                np.linalg.norm([Vs_left[3], Vs_left[4], Vs_left[5]]) > ev or \
                np.linalg.norm([Vs_right[0], Vs_right[1], Vs_right[2]]) > eomg or\
                np.linalg.norm([Vs_right[3], Vs_right[4], Vs_right[5]]) > ev or \
                residue_com > ecom 
                #or \
                #np.linalg.norm([Vs_torso[0], Vs_torso[1], Vs_torso[2]]) > eomg
            epsilon = 0.1
            # print(err)
            while err and j < maxiterations:

                # construct J matrix
                J = np.empty([15, self.model.nv])
                J[0:3, :] = pin.jacobianCenterOfMass(
                    self.model, self.data, self.q)
                J[3:9, :] = pin.getFrameJacobian(
                    self.model, self.data, self.idRfoot, pin.ReferenceFrame.WORLD)
                J[9:15, :] = pin.getFrameJacobian(
                    self.model, self.data, self.idLfoot, pin.ReferenceFrame.WORLD)
                # get torso Jacobian for optimization of torso pose
                #J[15:16, :] = pin.getFrameJacobian(
                 #   self.model, self.data, self.idTorso, pin.ReferenceFrame.WORLD)[3,:]

                #J_2 = pin.getFrameJacobian(self.model, self.data, self.idTorso, pin.ReferenceFrame.WORLD)[3:6,:]

                #print(J_2.shape)
                #J_2 = pin.getFrameJacobian(
                #    self.model, self.data, self.idTorso, pin.ReferenceFrame.WORLD)[3:6, :]
    '''
    '''
                Vs = np.zeros([15,1])
                Vs[0:3,0] = tmp_com 
                Vs[3:9,0] = Vs_right
                Vs[9:15,0] = Vs_left
    '''
    '''
                Vs = np.zeros([15, 1])
                Vs[0:3, 0] = tmp_com
                Vs[3:9, 0] = Vs_right
                Vs[9:15, 0] = Vs_left
                #Vs[15:16, 0] = Vs_torso

                #Vs_2 = np.zeros([3,1])
                #Vs_2[0:3,0] = Vs_torso
                # calculate delta q
                dq0 = pin.difference(self.model, self.q, self.q_des)
                #dq0 = np.linalg.pinv(J_2).dot(Vs_2)
                dq = np.linalg.pinv(J).dot(Vs) + self.getProjector(J).dot(dq0)
                #tmp_item = np.linalg.pinv(J_2.dot(self.getProjector(J)))
                #tmp_item_2 = Vs_2 - J_2.dot(np.linalg.pinv(J)).dot(Vs)
                #dq = np.linalg.pinv(J).dot(Vs) + self.getProjector(J).dot(tmp_item.dot(tmp_item_2))
                dq = dq * epsilon
                self.q = pin.integrate(self.model, self.q, dq)

                j = j + 1
                print(j)

                pin.forwardKinematics(self.model, self.data, self.q)
                pin.computeJointJacobians(self.model, self.data, self.q)
                pin.updateFramePlacements(self.model, self.data)

                T_left_foot = self.data.oMf[self.idLfoot]
                T_right_foot = self.data.oMf[self.idRfoot]
                com = m2a(pin.centerOfMass(self.model, self.data, self.q))
                #T_torso_ = self.data.oMf[self.idTorso]

                tmp_left = T_left_foot.inverse()*refLeftFoot
                Vs_left = m2a(pin.log(tmp_left).vector)

                tmp_right = T_right_foot.inverse()*refRightFoot
                Vs_right = m2a(pin.log(tmp_right).vector)

                tmp_com = refCom - com
                residue_com = sum(tmp_com**2)

                #tmp_torso = np T_torso_.inverse()*refTorso
                #Vs_torso = m2a(np.array([0, 0, 0]))

                #T_torso_ = self.data.oMf[self.idTorso].rotation

                #tmp_torso = a2m(np.linalg.inv(T_torso_).dot(refTorso))
                #tmp_torso = T_torso_.inverse()*refTorso
                
                #Vs_torso = m2a(pin.log3(tmp_torso))[0]

                err = np.linalg.norm([Vs_left[0], Vs_left[1], Vs_left[2]]) > eomg \
                    or np.linalg.norm([Vs_left[3], Vs_left[4], Vs_left[5]]) > ev or \
                    np.linalg.norm([Vs_right[0], Vs_right[1], Vs_right[2]]) > eomg \
                    or np.linalg.norm([Vs_right[3], Vs_right[4], Vs_right[5]]) > ev or \
                    residue_com > ecom 
                    #or \
                #np.linalg.norm([Vs_torso[0], Vs_torso[1], Vs_torso[2]]) > eomg 
                #or \
                #np.linalg.norm([Vs_torso[3], Vs_torso[4], Vs_torso[5]]) > ev

            # monitoring actual trajectories and com trajectories
            # first left and then right
            self.joint_traj[:, i+1] = m2a(self.q)[7:]
            self.foot_l_traj[:, i] = m2a(
                self.data.oMf[self.idLfoot].translation.T)
            self.foot_r_traj[:, i] = m2a(
                self.data.oMf[self.idRfoot].translation.T)
            self.com_sim_traj[2, i] = m2a(
                pin.centerOfMass(self.model, self.data, self.q))[2]
            self.com_sim_traj[1, i] = m2a(
                pin.centerOfMass(self.model, self.data, self.q))[1]
            self.com_sim_traj[0, i] = m2a(
                pin.centerOfMass(self.model, self.data, self.q))[0]

            # update animation
            if self.isDisplay == True:
                self.robot.display(self.q)
                # time.sleep(self.delta_t)
    '''

    def toQuaternion(self,yaw, pitch, roll):# yaw (Z), pitch (Y), roll (X)
        # Abbreviations for the various angular functions
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)


        q0 = cy * cp * sr - sy * sp * cr
        q1 = sy * cp * sr + cy * sp * cr
        q2 = sy * cp * cr - cy * sp * sr
        q3 = cy * cp * cr + sy * sp * sr
        return q0,q1,q2,q3
    def quat2eul(self,quat):
        q_0 = quat[0]
        # vector part:
        q_1 = quat[1]
        q_2 = quat[2]
        q_3 = quat[3]
        eul_y = math.atan2(2*(q_1*q_2 + q_0*q_3), q_0*q_0 + q_1*q_1 - q_2*q_2 - q_3*q_3)

        eul_p = math.asin(-2*(q_1*q_3 - q_0*q_2))                                     

        eul_r = math.atan2(2*(q_2*q_3 + q_0*q_1), q_0*q_0 - q_1*q_1 - q_2*q_2 + q_3*q_3)

        return eul_y, eul_p, eul_r
    def showLeftFootTrajectories(self):
        fig = plt.figure(figsize=(12, 6.5))
        plt.subplot(3, 1, 1)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length), self.foot_l_traj[0, :])
        plt.title('Left foot trajectory: x')
        plt.ylabel('x')
        plt.xlabel('t')
        plt.subplot(3, 1, 2)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length), self.foot_l_traj[1, :])
        plt.title('Left foot trajectory: y')
        plt.ylabel('y')
        plt.xlabel('t')
        plt.subplot(3, 1, 3)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length), self.foot_l_traj[2, :])
        plt.title('Left foot trajectory: z')
        plt.ylabel('z')
        plt.xlabel('t')
        plt.show()

    def showRightFootTrajectories(self):
        fig = plt.figure(figsize=(12, 6.5))
        plt.subplot(3, 1, 1)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length), self.foot_r_traj[0, :])
        plt.title('Right foot trajectory: x')
        plt.ylabel('x')
        plt.xlabel('t')
        plt.subplot(3, 1, 2)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length), self.foot_r_traj[1, :])
        plt.title('Right foot trajectory: y')
        plt.ylabel('y')
        plt.xlabel('t')
        plt.subplot(3, 1, 3)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length), self.foot_r_traj[2, :])
        plt.title('Right foot trajectory: z')
        plt.ylabel('z')
        plt.xlabel('t')
        plt.show()

    def showComTrajectories(self):
        fig = plt.figure(figsize=(12, 6.5))
        plt.subplot(3, 1, 1)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length), self.com_sim_traj[0, :])
        plt.plot(np.arange(self.horizon_length), self.Traj_com[0, :])
        plt.title('Com traj: x')
        plt.ylabel('x')
        plt.xlabel('t')
        plt.subplot(3, 1, 2)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length), self.com_sim_traj[1, :])
        plt.plot(np.arange(self.horizon_length), self.Traj_com[1, :])
        plt.title('Com traj: y')
        plt.ylabel('y')
        plt.xlabel('t')
        plt.subplot(3, 1, 3)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length), self.com_sim_traj[2, :])
        plt.plot(np.arange(self.horizon_length), self.Traj_com[2, :])
        plt.title('Com traj: z')
        plt.ylabel('z')
        plt.xlabel('t')
        plt.show()

    def showCumulativeResult(self):
        init_traj = np.zeros([3, self.horizon_length+1])
        for i in range(self.horizon_length):
            init_traj[:, i+1] = init_traj[:, i] + self.w3[0:3, i]*self.delta_t
        fig = plt.figure(figsize=(12, 6.5))
        plt.subplot(3, 1, 1)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length),
                 init_traj[0, 0:self.horizon_length])
        plt.title('Com traj: x')
        plt.ylabel('x')
        plt.xlabel('t')
        plt.subplot(3, 1, 2)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length),
                 init_traj[1, 0:self.horizon_length])
        plt.title('Com traj: y')
        plt.ylabel('y')
        plt.xlabel('t')
        plt.subplot(3, 1, 3)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length),
                 init_traj[2, 0:self.horizon_length])
        plt.title('Com traj: z')
        plt.ylabel('z')
        plt.xlabel('t')
        plt.show()

    def showJointTraj(self):
        fig = plt.figure(figsize=(12, 6.5))
        plt.subplot(6, 2, 1)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.joint_traj[0, :])
        plt.title('Joint Trajectory: Hip_yaw')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 2)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.joint_traj[1, :])
        plt.title('Joint Trajectory: Hip_roll')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 3)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.joint_traj[2, :])
        plt.title('Joint Trajectory: Hip_pitch')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 4)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.joint_traj[3, :])
        plt.title('Joint Trajectory: Knee')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 5)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.joint_traj[4, :])
        plt.title('Joint Trajectory: Ankle_pitch')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 6)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.joint_traj[5, :])
        plt.title('Joint Trajectory: Ankle_roll')
        plt.ylabel('y')
        plt.xlabel('x')

        plt.subplot(6, 2, 7)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.joint_traj[6, :])
        plt.title('Joint Trajectory: Hip_yaw')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 8)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.joint_traj[7, :])
        plt.title('Joint Trajectory: Hip_roll')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 9)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.joint_traj[8, :])
        plt.title('Joint Trajectory: Hip_pitch')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 10)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.joint_traj[9, :])
        plt.title('Joint Trajectory: Knee')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 11)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.joint_traj[10, :])
        plt.title('Joint Trajectory: Ankle_pitch')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 12)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.joint_traj[11, :])
        plt.title('Joint Trajectory: Ankle_roll')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()

    def showCopTrajectories(self):
        fig = plt.figure(figsize=(12, 6.5))
        plt.subplot(3, 1, 1)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.cop_bar[0, :])
        plt.title('Cop traj: x')
        plt.ylabel('x')
        plt.xlabel('t')
        plt.subplot(3, 1, 2)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.cop_bar[1, :])
        plt.title('Cop traj: y')
        plt.ylabel('y')
        plt.xlabel('t')
        plt.subplot(3, 1, 3)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(np.arange(self.horizon_length+1), self.cop_bar[2, :])
        plt.title('Cop traj: z')
        plt.ylabel('z')
        plt.xlabel('t')
        plt.show()

    def compareJointTraj(self):
        plt.subplot(6, 2, 1)
        motor_joint_before = np.load('motor_traj.npy')
        #data_length = traj_joint_before.shape[1]
        traj_t = np.arange(self.horizon_length+1)
        traj_joint_before = motor_joint_before[:, 550:]
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(traj_t, self.joint_traj[0, :])
        plt.plot(traj_t, traj_joint_before[0, :])
        plt.title('Joint Trajectory: Hip_yaw')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 2)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(traj_t, self.joint_traj[1, :])
        plt.plot(traj_t, traj_joint_before[1, :])
        plt.title('Joint Trajectory: Hip_roll')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 3)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(traj_t, self.joint_traj[2, :])
        plt.plot(traj_t, traj_joint_before[2, :])
        plt.title('Joint Trajectory: Hip_pitch')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 4)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(traj_t, self.joint_traj[3, :])
        plt.plot(traj_t, traj_joint_before[3, :])
        plt.title('Joint Trajectory: Knee')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 5)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(traj_t, self.joint_traj[4, :])
        plt.plot(traj_t, traj_joint_before[4, :])
        plt.title('Joint Trajectory: Ankle_pitch')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 6)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(traj_t, self.joint_traj[5, :])
        plt.plot(traj_t, traj_joint_before[5, :])
        plt.title('Joint Trajectory: Ankle_roll')
        plt.ylabel('y')
        plt.xlabel('x')

        plt.subplot(6, 2, 7)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(traj_t, self.joint_traj[6, :])
        plt.plot(traj_t, traj_joint_before[6, :])
        plt.title('Joint Trajectory: Hip_yaw')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 8)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(traj_t, self.joint_traj[7, :])
        plt.plot(traj_t, traj_joint_before[7, :])
        plt.title('Joint Trajectory: Hip_roll')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 9)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(traj_t, self.joint_traj[8, :])
        plt.plot(traj_t, traj_joint_before[8, :])
        plt.title('Joint Trajectory: Hip_pitch')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 10)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(traj_t, self.joint_traj[9, :])
        plt.plot(traj_t, traj_joint_before[9, :])
        plt.title('Joint Trajectory: Knee')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 11)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(traj_t, self.joint_traj[10, :])
        plt.plot(traj_t, traj_joint_before[10, :])
        plt.title('Joint Trajectory: Ankle_pitch')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 12)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(traj_t, self.joint_traj[11, :])
        plt.plot(traj_t, traj_joint_before[11, :])
        plt.title('Joint Trajectory: Ankle_roll')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()

    def showTorqueTraj(self):
        plt.subplot(6, 2, 1)
        torque_joint_before = np.load('torque_traj.npy')
        #data_length = traj_joint_before.shape[1]
        traj_t = np.arange(self.horizon_length+1)
        traj_joint_before = torque_joint_before[:, 550:]
        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.plot(traj_t,self.joint_traj[0,:])
        plt.plot(traj_t, traj_joint_before[0, :])
        plt.title('Joint Trajectory: Hip_yaw')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 2)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.plot(traj_t,self.joint_traj[1,:])
        plt.plot(traj_t, traj_joint_before[1, :])
        plt.title('Joint Trajectory: Hip_roll')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 3)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.plot(traj_t,self.joint_traj[2,:])
        plt.plot(traj_t, traj_joint_before[2, :])
        plt.title('Joint Trajectory: Hip_pitch')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 4)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.plot(traj_t,self.joint_traj[3,:])
        plt.plot(traj_t, traj_joint_before[3, :])
        plt.title('Joint Trajectory: Knee')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 5)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.plot(traj_t,self.joint_traj[4,:])
        plt.plot(traj_t, traj_joint_before[4, :])
        plt.title('Joint Trajectory: Ankle_pitch')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 6)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.plot(traj_t,self.joint_traj[5,:])
        plt.plot(traj_t, traj_joint_before[5, :])
        plt.title('Joint Trajectory: Ankle_roll')
        plt.ylabel('y')
        plt.xlabel('x')

        plt.subplot(6, 2, 7)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.plot(traj_t,self.joint_traj[6,:])
        plt.plot(traj_t, traj_joint_before[6, :])
        plt.title('Joint Trajectory: Hip_yaw')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 8)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.plot(traj_t,self.joint_traj[7,:])
        plt.plot(traj_t, traj_joint_before[7, :])
        plt.title('Joint Trajectory: Hip_roll')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 9)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.plot(traj_t,self.joint_traj[8,:])
        plt.plot(traj_t, traj_joint_before[8, :])
        plt.title('Joint Trajectory: Hip_pitch')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 10)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.plot(traj_t,self.joint_traj[9,:])
        plt.plot(traj_t, traj_joint_before[9, :])
        plt.title('Joint Trajectory: Knee')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 11)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.plot(traj_t,self.joint_traj[10,:])
        plt.plot(traj_t, traj_joint_before[10, :])
        plt.title('Joint Trajectory: Ankle_pitch')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(6, 2, 12)
        plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.plot(traj_t,self.joint_traj[11,:])
        plt.plot(traj_t, traj_joint_before[11, :])
        plt.title('Joint Trajectory: Ankle_roll')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()
