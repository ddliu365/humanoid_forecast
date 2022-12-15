import numpy as np

########################################################
######This functon is for trajecotry generation#########
########################################################


class generator():
    def __init__(self, init=np.array([0, 0, 0]), mid_point_z=0.):

        ##
        # init: indicate starting point of trajectory
        # mid_point_z: For foot trajectories, we only consider z of waypoint
        ##

        # trajectory contains (x, y, z)
        self.traj = np.zeros([3, 1])
        self.traj[:, 0] = init
        self.mid_point_z = mid_point_z

    def add(self, x=0., y=0., z=0., point=np.array([0., 0., 0.]), length=1., isCurve=False, ratio=0.5):
        ##
        # x: delta_x compared with last position
        # y: delat_y compared with last position
        # z: delat_z compared with last position
        # length: adding time to realize the step
        # isCurve = False: using linear line to generate way points
        # isCurve = True: using curve to generate way points
        # mid_time_rato: middle point time indicator during waypoints generation
        # point: end point if it exists
        ##

        if length < 1:
            print("Trajectory Generator Error: input length is less than 1!")
            return

        # extract the last time of trajectory matrix
        _last_time = np.size(self.traj, 1)-1
        _last_pos = self.traj[:, _last_time]
        _curr_pos = _last_pos + np.array([x, y, z])

        if (point != np.array([0., 0., 0.])).any():
            _curr_pos = point

        if _last_time == 0:
            if isCurve == False:
                _tmp_traj = self.linearTraj(_last_pos, _curr_pos, length)
            else:
                _tmp_traj = self.getFootTrajectory(
                    _last_pos, _curr_pos, T=length, ratio=ratio)
            self.traj = _tmp_traj.copy()
        else:
            if isCurve == False:
                _tmp_traj = self.linearTraj(_last_pos, _curr_pos, length+1)
            else:
                _tmp_traj = self.getFootTrajectory(
                    _last_pos, _curr_pos, T=(length + 1), ratio=ratio)

            self.traj = np.concatenate(
                (self.traj.copy(), _tmp_traj[:, 1:]), axis=1)

    def linearTraj(self, x_init, x_end, T):

        # given initPoint(x,y) and endPoint(x,y)
        # output trajectory y = (x(t),y(t)) from 0 to T
        y = np.zeros([3, T])
        for i in range(0, T):
            y[0, i] = (x_end[0]-x_init[0])/(T-1)*i + x_init[0]
            y[1, i] = (x_end[1]-x_init[1])/(T-1)*i + x_init[1]
            y[2, i] = (x_end[2]-x_init[2])/(T-1)*i + x_init[2]
        return y

    def getFootTrajectory(self, x_a, x_c, T, ratio=0.5):

        ##
        # x_a: start point
        # x_b: middle point
        # x_c: end point
        # ratio: the percentage of the first part
        ##

        traj = np.zeros([3, T])

        for i in range(3):

            s_a = x_a[i]

            if i == 2:

                s_b = self.mid_point_z + x_a[i]
            else:
                s_b = ratio*x_c[i]+(1-ratio)*x_a[i]

            s_c = x_c[i]

            v_a = 0.
            v_b = 0.
            v_c = 0.

            a_a = 0.
            a_b = 0.
            a_c = 0.

            T_1 = int(T*ratio)
            T_2 = T

            if i == 2:

                a = np.array([
                    [1, 0, 0, 0, 0, 0],
                    [1, (T_1-1), (T_1-1)**2, (T_1-1) **
                     3,   (T_1-1)**4,   (T_1-1)**5],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1,   2*(T_1-1),  3*(T_1-1)**2, 4 *
                     (T_1-1)**3, 5*(T_1-1)**4],
                    [0, 0, 2, 0, 0, 0],
                    [0, 0, 2, 6*(T_1-1), 12*(T_1-1)**2, 20*(T_1-1)**3]
                ])

                b = np.array([
                    [s_a],
                    [s_b],
                    [v_a],
                    [v_b],
                    [a_a],
                    [a_b]
                ])

                x_1 = np.linalg.solve(a, b)

                def y1(t): return x_1[0] + x_1[1] * t + \
                    x_1[2]*t**2 + x_1[3]*t**3 + x_1[4]*t**4 + x_1[5]*t**5

            else:
                # curve 1: cubic time scaling
                a = np.array([
                    [1, 0, 0, 0, 0],
                    [1, (T_1-1), (T_1-1)**2, (T_1-1)**3, (T_1-1)**4],
                    [0, 1, 0, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 2, 6*(T_1-1), 12*(T_1-1)**2]
                ])

                b = np.array([
                    [s_a],
                    [s_b],
                    [v_a],
                    [a_a],
                    [a_b]
                ])

                x_1 = np.linalg.solve(a, b)

                def y1(t): return x_1[0] + x_1[1] * t + \
                    x_1[2]*t**2 + x_1[3]*t**3 + x_1[4]*t**4
                def d_y1(t): return x_1[1] + 2*x_1[2] * \
                    t + 3*x_1[3]*t**2 + 4*x_1[4]*t**3

                v_b = d_y1(T_1)[0]

            a = np.array([
                [1, (T_1-1), (T_1-1)**2, (T_1-1)**3, (T_1-1)**4, (T_1-1)**5],
                [1, (T_2-1), (T_2-1)**2, (T_2-1)**3, (T_2-1)**4, (T_2-1)**5],
                [0, 1, 2*(T_1-1), 3*(T_1-1)**2, 4*(T_1-1)**3, 5*(T_1-1)**4],
                [0, 1, 2*(T_2-1), 3*(T_2-1)**2, 4*(T_2-1)**3, 5*(T_2-1)**4],
                [0, 0, 2, 6*(T_1-1), 12*(T_1-1)**2, 20*(T_1-1)**3],
                [0, 0, 2, 6*(T_2-1), 12*(T_2-1)**2, 20*(T_2-1)**3]
            ])

            b = np.array([
                [s_b],
                [s_c],
                [v_b],
                [v_c],
                [a_b],
                [a_c]
            ])
            b_m = b
            x_2 = np.linalg.solve(a, b)

            def y2(t): return x_2[0] + x_2[1] * t + x_2[2] * \
                t**2 + x_2[3]*t**3 + x_2[4]*t**4 + x_2[5]*t**5

            for j in range(0, T_1):

                traj[i, j] = y1(j)

            for j in range(T_1, T_2):

                traj[i, j] = y2(j)

        return traj
