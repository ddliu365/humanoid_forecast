import py_lqr.model as model
import py_lqr.planner as pl
import py_lqr.trajectory_generation as tg
import os
import sys

# get current folder path
current_path=sys.argv[0]
pathname = os.path.dirname(sys.argv[0])

# set urdf, mesh, output data folder
urdfPath = os.path.abspath(os.path.join(current_path, '../../../humanoid_property/urdf/humanoid_pinocchio.urdf'))
meshPath=os.path.abspath(os.path.join(current_path, '../../../humanoid_property'))
dataPath=os.path.abspath(os.path.join(current_path, '../../data'))


robot = model.loadHumanoidModel(isDisplay=False,urdfPath = urdfPath,meshPath = meshPath)
planner = pl.stepPlanner(robot)
traj = tg.trajectoryGenerate(robot, planner, isDisplay=False, data_path = dataPath)
simulator = tg.simulator(robot, traj, isRecompute=True, isOnlineCompute=False)