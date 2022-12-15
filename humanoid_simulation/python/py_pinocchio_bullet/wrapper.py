#///////////////////////////////////////////////////////////////////////////////
#// BSD 3-Clause License
#//
#// Copyright (C) 2018-2019, New York University
#// Copyright note valid unless otherwise stated in individual files.
#// All rights reserved.
#///////////////////////////////////////////////////////////////////////////////

import pybullet as p
import pinocchio as se3
import numpy as np
from time import sleep

from pinocchio.utils import zero

class PinBulletWrapper(object):
    def __init__(self, robot_id, pinocchio_robot, joint_names, endeff_names, useFixedBase=False, useTorqueCtrl = True):
        self.nq = pinocchio_robot.nq
        self.nv = pinocchio_robot.nv
        self.nj = len(joint_names)
        self.nf = len(endeff_names)
        self.robot_id = robot_id
        self.pinocchio_robot = pinocchio_robot
        self.useFixedBase = useFixedBase
        self.useTorqueCtrl = useTorqueCtrl

        self.joint_names = joint_names
        self.endeff_names = endeff_names

        bullet_joint_map = {}
        for ji in range(p.getNumJoints(robot_id)):
            bullet_joint_map[p.getJointInfo(robot_id, ji)[1].decode('UTF-8')] = ji

        self.bullet_joint_ids = np.array([bullet_joint_map[name] for name in joint_names])
        self.pinocchio_joint_ids = np.array([pinocchio_robot.model.getJointId(name) for name in joint_names])

        self.pin2bullet_joint_only_array = []

        if not self.useFixedBase:
            for i in range(2, self.nj + 2):
                self.pin2bullet_joint_only_array.append(np.where(self.pinocchio_joint_ids == i)[0][0])
        else:
            for i in range(1, self.nj + 1):
                self.pin2bullet_joint_only_array.append(np.where(self.pinocchio_joint_ids == i)[0][0])


        # Disable the velocity control on the joints as we use torque control.
        if self.useTorqueCtrl == True:
            p.setJointMotorControlArray(robot_id, self.bullet_joint_ids, p.VELOCITY_CONTROL, forces=np.zeros(self.nj))

        # In pybullet, the contact wrench is measured at a joint. In our case
        # the joint is fixed joint. Pinocchio doesn't add fixed joints into the joint
        # list. Therefore, the computation is done wrt to the frame of the fixed joint.
        self.bullet_endeff_ids = [bullet_joint_map[name] for name in endeff_names]
        self.pinocchio_endeff_ids = [pinocchio_robot.model.getFrameId(name) for name in endeff_names]
        # Enable force torque sensor on the foot
        for i in self.bullet_endeff_ids:
            p.enableJointForceTorqueSensor(robot_id, i)

    def _action(self, pos, rot):
        res = np.zeros((6, 6))
        res[:3, :3] = rot
        res[3:, 3:] = rot
        res[3:, :3] = se3.utils.skew(np.array(pos)).dot(rot)
        return res

    def get_force(self):
        """ Returns the force readings as well as the set of active contacts """
        active_contacts_frame_ids = []
        contact_forces = []

        contact_forces_foot = {"left": 0, "right": 0}


        '''
        Calcualte center of pressure on the foot
        All force for CoP calcualtion are normal
        Cop_x = sum(force_i_x * force_i)/sum(force_i)
        Cop_y = sum(force_i_y * force_i)/sum(force_i)

        contact_cop_dist_force += (force_i_x * force_i, force_i_y * force_i, 0)
        contact_cop_force += force_i
        '''
        contact_cop = np.zeros(3)
        contact_cop_ft = np.zeros(3)# obtain cop from force torque sensor
        contact_cop_dist_force_x = 0.
        contact_cop_dist_force_y = 0.
        contact_cop_force = 0.

        contact_cop_ft_l = np.zeros(3)
        contact_cop_ft_r = np.zeros(3)

        l_torque = p.getJointState(self.robot_id, self.bullet_endeff_ids[0])[2][0:3]
        l_force = p.getJointState(self.robot_id, self.bullet_endeff_ids[0])[2][3:6]
        r_torque = p.getJointState(self.robot_id, self.bullet_endeff_ids[1])[2][0:3]
        r_force = p.getJointState(self.robot_id, self.bullet_endeff_ids[1])[2][3:6]
        #print(l_force,l_torque,r_force,r_torque)
        if l_force[2] == 0:
            contact_cop_ft_l[0]=0
            contact_cop_ft_l[1]=0
        else:
            contact_cop_ft_l[0] = (-l_torque[1]-l_force[0]*0.)/l_force[2]
            contact_cop_ft_l[1] = (l_torque[0]-l_force[1]*0.)/l_force[2]        # Get the contact model using the p.getContactPoints() api.

        if r_force[2] == 0:
            contact_cop_ft_r[0]=0
            contact_cop_ft_r[1]=0
        else:
            contact_cop_ft_r[0] = (-r_torque[1]-r_force[0]*0.)/r_force[2]
            contact_cop_ft_r[1] = (r_torque[0]-r_force[1]*0.)/r_force[2]

        def sign(x):
            if x >= 0:
                return 1.
            else:
                return -1.

        cp = p.getContactPoints()
        is_a = 0
        for ci in reversed(cp):
            contact_normal = ci[7]
            normal_force = ci[9]
            lateral_friction_direction_1 = ci[11]
            lateral_friction_force_1 = ci[10]
            lateral_friction_direction_2 = ci[13]
            lateral_friction_force_2 = ci[12]

            # Calculate cop on the foot
            world_pos_force_a = ci[5]
            world_pos_force_b = ci[6]


            if ci[3] in self.bullet_endeff_ids:
                i = np.where(np.array(self.bullet_endeff_ids) == ci[3])[0][0]
                is_a = 0
                #print(i)
            elif ci[4] in self.bullet_endeff_ids:
                i = np.where(np.array(self.bullet_endeff_ids) == ci[4])[0][0]
                is_a = 1
                #print(i)
            else:
                continue

            if self.pinocchio_endeff_ids[i] in active_contacts_frame_ids:
                continue

            active_contacts_frame_ids.append(self.pinocchio_endeff_ids[i])
            force = np.zeros(6)

            force[:3] = normal_force * np.array(contact_normal) + \
                        lateral_friction_force_1 * np.array(lateral_friction_direction_1) + \
                        lateral_friction_force_2 * np.array(lateral_friction_direction_2)
            contact_cop_force = contact_cop_force + normal_force
            if is_a == 0:
                contact_cop_dist_force_x = contact_cop_dist_force_x + normal_force * world_pos_force_a[0]
                contact_cop_dist_force_y = contact_cop_dist_force_y + normal_force * world_pos_force_a[1]
            else:
                contact_cop_dist_force_x = contact_cop_dist_force_x + normal_force * world_pos_force_b[0]
                contact_cop_dist_force_y = contact_cop_dist_force_y + normal_force * world_pos_force_b[1]

            contact_forces.append(force)
        if contact_cop_force != 0:
            contact_cop[0] = contact_cop_dist_force_x/contact_cop_force
            contact_cop[1] = contact_cop_dist_force_y/contact_cop_force

        return active_contacts_frame_ids[::-1], contact_forces[::-1], contact_cop, contact_cop_force,contact_cop_ft_l,contact_cop_ft_r

    def get_force_link(self):
        """ Returns the force readings as well as the set of active contacts """
        active_contacts_frame_ids = []
        active_contacts_frame_names = []
        contact_forces = []

        # Get the contact model using the p.getContactPoints() api.

        cp = p.getContactPoints()

        for ci in reversed(cp):
            contact_normal = ci[7]
            normal_force = ci[9]
            lateral_friction_direction_1 = ci[11]
            lateral_friction_force_1 = ci[10]
            lateral_friction_direction_2 = ci[13]
            lateral_friction_force_2 = ci[12]

            if ci[3] in self.bullet_joint_ids:
                j = np.where(np.array(self.bullet_joint_ids) == ci[3])[0][0]
            elif ci[4] in self.bullet_joint_ids:
                j = np.where(np.array(self.bullet_joint_ids) == ci[4])[0][0]
            else:
                continue

            if self.pinocchio_joint_ids[j] in active_contacts_frame_ids:
                continue

            active_contacts_frame_ids.append(self.pinocchio_joint_ids[j])
            force = np.zeros(6)

            force[:3] = normal_force * np.array(contact_normal) + \
                        lateral_friction_force_1 * np.array(lateral_friction_direction_1) + \
                        lateral_friction_force_2 * np.array(lateral_friction_direction_2)

            contact_forces.append(force)

        return active_contacts_frame_ids[::-1], contact_forces[::-1]

    def get_state(self):
        # Returns a pinocchio like representation of the q, dq matrixes
        q = zero(self.nq)
        dq = zero(self.nv)

        if not self.useFixedBase:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            q[:3] = np.array(pos).reshape(3)
            q[3:7] = np.array(orn).reshape(4)

            vel, orn = p.getBaseVelocity(self.robot_id)
            dq[:3] = np.array(vel).reshape(3)
            dq[3:6] = np.array(orn).reshape(3)

            # Pinocchio assumes the base velocity to be in the body frame -> rotate.
            rot = np.matrix(p.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
            dq[0:3] = rot.T.dot(dq[0:3])
            dq[3:6] = rot.T.dot(dq[3:6])

        # Query the joint readings.
        joint_states = p.getJointStates(self.robot_id, self.bullet_joint_ids)

        if not self.useFixedBase:
            for i in range(self.nj):
                q[5 + self.pinocchio_joint_ids[i]] = joint_states[i][0]
                dq[4 + self.pinocchio_joint_ids[i]] = joint_states[i][1]
        else:
            for i in range(self.nj):
                q[self.pinocchio_joint_ids[i] - 1] = joint_states[i][0]
                dq[self.pinocchio_joint_ids[i] - 1] = joint_states[i][1]

        return q, dq

    def update_pinocchio(self, q, dq):
        """Updates the pinocchio robot.

        This includes updating:
        - kinematics
        - joint and frame jacobian
        - centroidal momentum

        Args:
          q: Pinocchio generalized position vect.
          dq: Pinocchio generalize velocity vector.
        """
        self.pinocchio_robot.computeJointJacobians(q)
        self.pinocchio_robot.framesForwardKinematics(q)
        self.pinocchio_robot.centroidalMomentum(q, dq)

    def get_state_update_pinocchio(self):
        """Get state from pybullet and update pinocchio robot internals.

        This gets the state from the pybullet simulator and forwards
        the kinematics, jacobians, centroidal moments on the pinocchio robot
        (see forward_pinocchio for details on computed quantities). """
        q, dq = self.get_state()
        self.update_pinocchio(q, dq)
        return q, dq

    def reset_state(self, q, dq):
        vec2list = lambda m: np.array(m.T).reshape(-1).tolist()

        if not self.useFixedBase:
            p.resetBasePositionAndOrientation(self.robot_id, vec2list(q[:3]), vec2list(q[3:7]))

            # Pybullet assumes the base velocity to be aligned with the world frame.
            rot = np.matrix(p.getMatrixFromQuaternion(q[3:7])).reshape((3, 3))
            p.resetBaseVelocity(self.robot_id, vec2list(rot.dot(dq[:3])), vec2list(rot.dot(dq[3:6])))

            for i, bullet_joint_id in enumerate(self.bullet_joint_ids):
                p.resetJointState(self.robot_id, bullet_joint_id,
                    q[5 + self.pinocchio_joint_ids[i]],
                    dq[4 + self.pinocchio_joint_ids[i]])
        else:
            for i, bullet_joint_id in enumerate(self.bullet_joint_ids):
                p.resetJointState(self.robot_id, bullet_joint_id,
                    q[self.pinocchio_joint_ids[i] - 1],
                    dq[self.pinocchio_joint_ids[i] - 1])


    def send_joint_command(self, tau):
        # TODO: Apply the torques on the base towards the simulator as well.
        if not self.useFixedBase:
            assert(tau.shape[0] == self.nv - 6)
        else:
            assert(tau.shape[0] == self.nv)

        zeroGains = tau.shape[0] * (0.,)
        forces = tau.shape[0] * (8.4,)
        if self.useTorqueCtrl == True:
            p.setJointMotorControlArray(self.robot_id, self.bullet_joint_ids, p.TORQUE_CONTROL,
                    forces=tau[self.pin2bullet_joint_only_array],
                    positionGains=zeroGains, velocityGains=zeroGains)
        else:
            p.setJointMotorControlArray(self.robot_id, self.bullet_joint_ids, p.POSITION_CONTROL,
                    targetPositions=tau, forces = forces)

    def get_foot_pos(self, q):

        ret = {"leftfoot": None, "rightfoot": None}

        lfootName, rfootName = "leg_l_foot", "leg_r_foot"
        lfootId, rfootId = self.pinocchio_robot.model.getFrameId(lfootName), self.pinocchio_robot.model.getFrameId(rfootName)

        se3.forwardKinematics(self.pinocchio_robot.model, self.pinocchio_robot.data, q)
        se3.updateFramePlacements(self.pinocchio_robot.model, self.pinocchio_robot.data)
        lfootPos = self.pinocchio_robot.data.oMf[lfootId].translation.copy()
        rfootPos = self.pinocchio_robot.data.oMf[rfootId].translation.copy()


        ret["leftfoot"] = lfootPos
        ret["rightfoot"] = rfootPos
        
        return ret



    def get_left_foot_cop(self):

        return 0

    
    def get_left_foot_normal_force(self):

        """ Returns the force readings as well as the set of active contacts """
        active_contacts_frame_ids = []
        contact_forces = []

        contact_forces_foot = {"left": 0, "right": 0}


        '''
        Calcualte center of pressure on the foot
        All force for CoP calcualtion are normal
        Cop_x = sum(force_i_x * force_i)/sum(force_i)
        Cop_y = sum(force_i_y * force_i)/sum(force_i)

        contact_cop_dist_force += (force_i_x * force_i, force_i_y * force_i, 0)
        contact_cop_force += force_i
        '''
        contact_cop = np.zeros(3)
        contact_cop_ft = np.zeros(3)# obtain cop from force torque sensor
        contact_cop_dist_force_x = 0.
        contact_cop_dist_force_y = 0.
        contact_cop_force = 0.

        contact_cop_ft_l = np.zeros(3)
        contact_cop_ft_r = np.zeros(3)

        l_torque = p.getJointState(self.robot_id, self.bullet_endeff_ids[0])[2][0:3]
        l_force = p.getJointState(self.robot_id, self.bullet_endeff_ids[0])[2][3:6]
        r_torque = p.getJointState(self.robot_id, self.bullet_endeff_ids[1])[2][0:3]
        r_force = p.getJointState(self.robot_id, self.bullet_endeff_ids[1])[2][3:6]
        #print(l_force,l_torque,r_force,r_torque)
        if l_force[2] == 0:
            contact_cop_ft_l[0]=0
            contact_cop_ft_l[1]=0
        else:
            contact_cop_ft_l[0] = (-l_torque[1]-l_force[0]*0.)/l_force[2]
            contact_cop_ft_l[1] = (l_torque[0]-l_force[1]*0.)/l_force[2]        # Get the contact model using the p.getContactPoints() api.

        if r_force[2] == 0:
            contact_cop_ft_r[0]=0
            contact_cop_ft_r[1]=0
        else:
            contact_cop_ft_r[0] = (-r_torque[1]-r_force[0]*0.)/r_force[2]
            contact_cop_ft_r[1] = (r_torque[0]-r_force[1]*0.)/r_force[2]

        def sign(x):
            if x >= 0:
                return 1.
            else:
                return -1.

        cp = p.getContactPoints()
        is_a = 0
        for ci in reversed(cp):
            contact_normal = ci[7]
            normal_force = ci[9]
            lateral_friction_direction_1 = ci[11]
            lateral_friction_force_1 = ci[10]
            lateral_friction_direction_2 = ci[13]
            lateral_friction_force_2 = ci[12]

            # Calculate cop on the foot
            world_pos_force_a = ci[5]
            world_pos_force_b = ci[6]


            if ci[3] in self.bullet_endeff_ids:
                i = np.where(np.array(self.bullet_endeff_ids) == ci[3])[0][0]
                is_a = 0
                #print(i)
            elif ci[4] in self.bullet_endeff_ids:
                i = np.where(np.array(self.bullet_endeff_ids) == ci[4])[0][0]
                is_a = 1
                #print(i)
            else:
                continue

            if self.pinocchio_endeff_ids[i] in active_contacts_frame_ids:
                continue

            active_contacts_frame_ids.append(self.pinocchio_endeff_ids[i])
            force = np.zeros(6)

            force[:3] = normal_force * np.array(contact_normal) + \
                        lateral_friction_force_1 * np.array(lateral_friction_direction_1) + \
                        lateral_friction_force_2 * np.array(lateral_friction_direction_2)
            contact_cop_force = contact_cop_force + normal_force
            if is_a == 0:
                contact_cop_dist_force_x = contact_cop_dist_force_x + normal_force * world_pos_force_a[0]
                contact_cop_dist_force_y = contact_cop_dist_force_y + normal_force * world_pos_force_a[1]
            else:
                contact_cop_dist_force_x = contact_cop_dist_force_x + normal_force * world_pos_force_b[0]
                contact_cop_dist_force_y = contact_cop_dist_force_y + normal_force * world_pos_force_b[1]

            contact_forces.append(force)
        if contact_cop_force != 0:
            contact_cop[0] = contact_cop_dist_force_x/contact_cop_force
            contact_cop[1] = contact_cop_dist_force_y/contact_cop_force

        return active_contacts_frame_ids[::-1], contact_forces[::-1], contact_cop, contact_cop_force,contact_cop_ft_l,contact_cop_ft_r


    