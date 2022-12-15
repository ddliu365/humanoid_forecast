import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
import IPython
import pinocchio as pin
from pinocchio.utils import *
from numpy.linalg import norm, inv, pinv, svd, eig
from numpy.linalg import matrix_rank


class lqr():
    def __init__(self, robot, cop_bar, isDisplayResult=False):

        delta_t = robot.delta_t

        self.model = robot.model
        self.data = robot.data
        self.horizon_length = robot.horizon_length
        self.com_height = robot.com[2]
        self.com_forward = robot.com[0]
        self.com_offset_z = robot.com_0ffset_z
        self.g = robot.g
        self.Q0 = robot.Q
        self.R0 = robot.R

        self.traj_cop = cop_bar
        self.x0 = np.array(
            [self.com_forward, 0., 0., 0, 0., 0.])

        self.isDisplayResult = isDisplayResult

        self.A0 = np.array([[1, delta_t, 0.5*delta_t**2, 0, 0, 0],
                            [0, 1, delta_t, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, delta_t, 0.5*delta_t**2],
                            [0, 0, 0, 0, 1, delta_t],
                            [0, 0, 0, 0, 0, 1]])

        self.B0 = np.array([[1/6*delta_t**3, 0],
                            [1/2*delta_t**2, 0],
                            [delta_t, 0],
                            [0, 1/6*delta_t**3],
                            [0, 1/2*delta_t**2],
                            [0, delta_t]])

        self.C = np.array([[1, 0, -1*(self.com_height-self.com_offset_z)/self.g, 0, 0, 0],
                           [0, 0, 0, 1, 0, -1*(self.com_height-self.com_offset_z)/self.g]])

        self.X = self.lqrSolver()

    def solve_ricatti_equations(self, A, B, Q, q, R):
        """
        This function solves the backward Riccatti equations for regulator problems of the form
        min xQx + sum(xQx + uRu) subject to xn+1 = Axn + Bun

        Arguments:
        A, B, Q, q, R: numpy arrays defining the problem
        horizon_length: length of the horizon

        Returns:
        P: list of numpy arrays containing Pn from N to 0
        K: list of numpy arrays containing Kn from N-1 to 0
        p: list of numpy arrays containing pn from N to 0
        k: list of numpy arrays containing kn from N-1 to 0
        """
        P = []  # will contain the list of Ps from N to 0
        K = []  # will contain the list of Ks from N-1 to 0

        p = []  # will contain the list of ps from N to 0
        k = []  # will contain the list of ks from N-1 to 0

        P.append(Q[self.horizon_length-1])
        p.append(q[self.horizon_length-1])

        for n in range(self.horizon_length-2, -1, -1):

            c = R[n]+B[n].T.dot(P[self.horizon_length-2-n]).dot(B[n])
            K_n = - \
                np.linalg.inv(c).dot(B[n].T.dot(
                    P[self.horizon_length-2-n]).dot(A[n]))
            P_n = Q[n] + A[n].T.dot(P[self.horizon_length-2-n]).dot(A[n]) + \
                A[n].T.dot(P[self.horizon_length-2-n]).dot(B[n]).dot(K_n)
            k_n = -np.linalg.inv(c).dot(B[n].T.dot(p[self.horizon_length-2-n]))
            p_n = q[n].T + A[n].T.dot(p[self.horizon_length-2-n]) + \
                A[n].T.dot(P[self.horizon_length-2-n]).dot(B[n]).dot(k_n)

            P.append(P_n)
            K.append(K_n)
            k.append(k_n)
            p.append(p_n)

        # reverse the order in the list
        return P[::-1], K[::-1], p[::-1], k[::-1]

    def cost(self):
        # dynamic model
        # LQR
        q = []
        Q = []

        A = []
        B = []
        R = []

        Q0 = self.Q0
        R0 = self.R0

        for i in range(self.horizon_length):
            A.append(self.A0)

        for i in range(self.horizon_length):
            B.append(self.B0)

        for i in range(self.horizon_length):  # n+1 Q_n
            Q_n = self.C.T.dot(Q0).dot(self.C)
            Q.append(Q_n)

        for i in range(self.horizon_length):  # n+1 q_n
            q_n = -self.C.T.dot(Q0).dot(self.traj_cop[0:2, i])
            q.append(q_n)

        for i in range(self.horizon_length):
            R.append(R0)
        return A, B, Q, q, R

    def calculate_system(self, A, B, x0, K, k):
        '''
        This function calculates the state xn for each step
        xn+1 = Axn + Bun
        un=Kn*xn
        Arguments:
        A, B, x0, K 

        Returns:
        X: list of numpy arrays containing states xn from 0 to N
        time: list of numpy arrays containing time step from 0 to N
        '''
        X = np.zeros([x0.size, self.horizon_length])
        u = []
        xn = x0
        time = []
        X[:, 0] = xn
        for i in range(self.horizon_length-1):
            u.append(K[i].dot(xn) + k[i])
            xn = A.dot(xn) + B.dot(K[i].dot(xn)+k[i])
            X[:, i+1] = xn
            time.append(i)
        return X, u, time

    def check_controllability(self, A, B):
        """
        This function check  the controllabilitystate for system
        c=[B AB A^2B A^3B A^4B A^5B]
        """
        c = np.concatenate([B, np.dot(A, B), np.dot(A, A).dot(B), np.dot(A, A).dot(A).dot(
            B), np.dot(A, A).dot(A).dot(A).dot(B), np.dot(A, A).dot(A).dot(A).dot(A).dot(B)], axis=1)
        R = np.linalg.matrix_rank(c)
        print('Planner: Rank of control matrix is:', R)
        if R < np.linalg.matrix_rank(A):
            print('Planner: The system is not controllable')
        else:
            print('Planner: The system is controllable')

    def lqrSolver(self):

        A, B, Q, q, R = self.cost()

        self.check_controllability(self.A0, self.B0)

        P, K, p, k = self.solve_ricatti_equations(A, B, Q, q, R)

        X, u, time = self.calculate_system(self.A0, self.B0, self.x0, K, k)

        return X
