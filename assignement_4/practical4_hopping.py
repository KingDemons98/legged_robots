# Hopping practical
from env.leg_gym_env import LegGymEnv
from leg_helpers import *
import numpy as np
import matplotlib.pyplot as plt

env = LegGymEnv(render=True, 
                on_rack=False,    # set True to debug 
                motor_control_mode='TORQUE',
                action_repeat=1,
                # record_video=True
                )

NUM_SECONDS = 5   # simulate N seconds (sim dt is 0.001)
tau = np.zeros(2) # either torques or motor angles, depending on mode

# peform one jump, or continuous jumping
SINGLE_JUMP = False

# sample Cartesian PD gains (can change or optimize)
kpCartesian = np.diag([500,300])
kdCartesian = np.diag([30,20])

# define variables and force profile
t = np.linspace(0,NUM_SECONDS,NUM_SECONDS*1000 + 1)
Fx_max = 0     # max peak force in X direction
Fz_max = 0     # max peak force in Z direction
f = 0          # frequency

if SINGLE_JUMP:
    # may want to choose different parameters
    Fx_max = 0     # max peak force in X direction
    Fz_max = 0     # max peak force in Z direction
    f = 0

# design Z force trajectory as a funtion of Fz_max, f, t
#   Hint: use a sine function (but don't forget to remove positive forces)
force_traj_z = np.zeros(len(t))

if SINGLE_JUMP:
    # remove rest of profile (just keep the first peak)
    force_traj_z = np.zeros(len(t))

# design X force trajectory as a funtion of Fx_max, f, t
force_traj_x = np.zeros(len(t))

# sample nominal foot position (can change or optimize)
nominal_foot_pos = np.array([0.0,-0.2]) 

# keep track of max z height
max_base_z = 0

# Track the profile: what kind of controller will you use? 
for i in range(NUM_SECONDS*1000):
    # Torques
    tau = np.zeros(2) 

    # Compute jacobian and foot_pos in leg frame (use GetMotorAngles() )
    J, ee_pos_legFrame = jacobian_rel(env.robot.GetMotorAngles())

    # Add Cartesian PD (and/or joint PD? Think carefully about this, and try it out.)
    tau += np.zeros(2) # [TODO]

    # Add force profile contribution
    tau += J.T @ np.array([force_traj_x[i], force_traj_z[i]])

    # Apply control, simulate
    env.step(tau)

    # Record max base position (and/or other states)
    base_pos = env.robot.GetBasePosition()
    if max_base_z < base_pos[2]:
        max_base_z = base_pos[2]

print('Peak z', max_base_z)

# [TODO] make some plots to verify your force profile and system states
