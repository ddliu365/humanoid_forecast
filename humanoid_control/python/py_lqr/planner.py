import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from mpl_toolkits import mplot3d


class stepPlanner():
    def __init__(self, robot):  # original value =0.12

        # external parameter
        self.step_length = robot.step_length
        self.step_width = robot.step_width
        self.step_height = robot.step_height
        self.step_num = robot.step_num
        self.cop_bound_right_left = robot.cop_offset_y      # -0.01 can work

        # Step_width_add: adjust the middle footstep wider
        self.step_width_add = robot.step_offset_y
        self.com_forward = robot.com[0]
        self.com_leftward = robot.com[1]
        self.cop_offset_for_com_y = robot.cop_offset_for_com_y

        if self.step_num < 2:
            print("step number should be at least 2!")
            quit()

        # number of footprints
        self.num_footprint = self.step_num + 2

        # only the start and ending position are different for cop matrix and step matrix

        # CoP matrix: anchor point for generating cop trajectory
        self.cop_mat = np.zeros([3, self.num_footprint])

        # Step matrix: for foot trajectory generation
        self.step_mat = np.zeros([3, self.num_footprint])
        self.foot_print = np.zeros([3, self.num_footprint])

        self.generate_cop_matrix()
        self.generate_step_matrix()
        self.generate_footprint()

    # cop planning
    def generate_cop_matrix(self):

        # First and last step should be original step

        # NOTICE:step number definition: one action of walking counts 1 step
        if self.step_num < 2:
            print("please input step number more than 1!")
            return

        # initialize first two and last two anchor point
        self.cop_mat[0:2, 0] = np.array([self.com_forward, self.com_leftward])
        self.cop_mat[0:2, 1] = np.array(
            [0, self.step_width/2. + self.cop_bound_right_left])
        self.cop_mat[0:2, self.num_footprint - 2] =\
            np.array([(self.step_num/2.-0.5)*self.step_length, (-1) **
                      (self.num_footprint - 1)*(self.step_width/2. + self.step_width_add + self.cop_bound_right_left)])
        self.cop_mat[0:2, self.num_footprint - 1] =\
            np.array([(self.step_num/2.-0.5) *
                      self.step_length + self.com_forward + 0.01, self.com_leftward])

        for i in range(2, self.num_footprint-2):
            _x = (i-1)*self.step_length/2.
            _y = (-1)**(i+1)*(self.step_width/2. +
                              self.cop_bound_right_left + self.step_width_add)
            self.cop_mat[0:2, i] = np.array([_x, _y])

    # foot step planning
    # this function may be depreciated in the future
    def generate_step_matrix(self):
        if self.step_num < 2:
            print("please input step number more than 1!")
            return
        # generate foot matrix: (x,y,z): contain the center of foot, only different from the inital point and finishing point above
        self.step_mat[0:2, 0] = np.array([self.com_forward, 0])
        self.step_mat[0:2, 1] = np.array([0, self.step_width/2.])
        self.step_mat[0:2, self.num_footprint - 2] =\
            np.array([(self.step_num/2.-0.5)*self.step_length, (-1) **
                      (self.num_footprint - 1)*(self.step_width/2. + self.step_width_add)])
        self.step_mat[0:2, self.num_footprint - 1] =\
            np.array([(self.step_num/2.-0.5) *
                      self.step_length + self.com_forward, 0])

        for i in range(2, self.num_footprint-2):
            _x = (i-1)*self.step_length/2.
            _y = (-1)**(i+1)*(self.step_width/2. + self.step_width_add)
            self.step_mat[0:2, i] = np.array([_x, _y])

        self.step_mat[0:2, 0] = np.array([0, -1*self.step_width/2])
        self.step_mat[0:2, self.num_footprint-1] = np.array(
            [(self.step_num/2.-0.5)*self.step_length, (-1) **
             (self.num_footprint-1+1)*(self.step_width/2. + self.step_width_add)])
    def generate_footprint(self):
        if self.step_num < 2:
            print("please input step number more than 1!")
            return
        # generate foot matrix: (x,y,z): contain the center of foot, only different from the inital point and finishing point above
        self.foot_print[0:2, 0] = np.array([0, -self.step_width/2.])
        self.foot_print[0:2, 1] = np.array([0, self.step_width/2.])
        self.foot_print[0:2, self.num_footprint - 2] =\
            np.array([(self.step_num/2.-0.5)*self.step_length, (-1) **
                      (self.num_footprint - 1)*(self.step_width/2. + self.step_width_add)])
        self.foot_print[0:2, self.num_footprint - 1] =\
            np.array([(self.step_num/2.-0.5) *
                      self.step_length + self.com_forward, 0])

        for i in range(2, self.num_footprint-2):
            _x = (i-1)*self.step_length/2.
            _y = (-1)**(i+1)*(self.step_width/2. + self.step_width_add)
            self.foot_print[0:2, i] = np.array([_x, _y])

        self.foot_print[0:2, 0] = np.array([0, -1*self.step_width/2])
        self.foot_print[0:2, self.num_footprint-1] = np.array(
            [(self.step_num/2.-0.5)*self.step_length, (-1) **
             (self.num_footprint-1+1)*(self.step_width/2. + self.step_width_add)])
        
    def showFoot(self):
        # display footprint
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        xline = self.step_mat[0, :]
        yline = self.step_mat[1, :]
        x = np.zeros([2, self.num_footprint])
        ax.plot3D(xline, yline, x[0, :], 'orange')

    def showStep(self):
        # display step print
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        xline = self.cop_mat[0, :]
        yline = self.cop_mat[1, :]
        x = np.zeros([2, self.num_footprint])
        ax.plot3D(xline, yline, x[0, :], 'red')
