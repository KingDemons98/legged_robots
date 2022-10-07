# Jacobian practical
from env.leg_gym_env import LegGymEnv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def jacobian_abs(q, l1=0.209, l2=0.195):
    """ Jacobian based on absolute angles (like double pendulum)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    # Jacobian
    J = np.zeros((2, 2))
    J[0, 0] = l1 * np.cos(q[0])
    J[1, 0] = l1 * np.sin(q[0])
    J[0, 1] = l2 * np.cos(q[1])
    J[1, 1] = l2 * np.sin(q[1])

    # foot pos
    pos = np.zeros(2)
    pos[0] = l1 * np.sin(q[0]) + l2 * np.sin(q[1])
    pos[1] = -l1 * np.cos(q[0]) - l2 * np.cos(q[1])

    return J, pos


def jacobian_rel(q, l1=0.209, l2=0.195):
    """ Jacobian based on relative angles (like URDF)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    # Jacobian
    J = np.zeros((2, 2))
    J[0, 0] = -l1 * np.cos(q[0]) - l2 * np.cos(q[0] + q[1])
    J[1, 0] = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    J[0, 1] = -l2 * np.cos(q[0] + q[1])
    J[1, 1] = l2 * np.sin(q[0] + q[1])

    # foot pos
    pos = np.zeros(2)
    pos[0] = -l1 * np.sin(q[0]) - l2 * np.sin(q[0] + q[1])
    pos[1] = -l1 * np.cos(q[0]) - l2 * np.cos(q[0] + q[1])

    return J, pos


env = LegGymEnv(render=True,
                on_rack=True,  # set True to hang up robot in air
                motor_control_mode='TORQUE',
                action_repeat=1,
                )
dt = 1000
NUM_STEPS = 50 * dt  # simulate 5 seconds (sim dt is 0.001)

env._robot_config.INIT_MOTOR_ANGLES = np.array([-np.pi / 4, np.pi / 2])  # test different initial motor angles
obs = env.reset()  # reset environment if changing initial configuration

action = np.zeros(2)  # either torques or motor angles, depending on mode

# Test different Cartesian gains! How important are these?
kpCartesian = np.diag([500] * 2)
kdCartesian = np.diag([30] * 2)

# test different desired foot positions
des_foot_pos = np.array([0.05, -0.3])

torques = np.zeros((2, NUM_STEPS))
joint_positions = np.zeros((2, NUM_STEPS))
foot_positions = np.zeros((2, NUM_STEPS))
time = np.zeros(NUM_STEPS)

for i in range(NUM_STEPS):
    # Compute jacobian and foot_pos in leg frame (use GetMotorAngles() )
    motor_ang = env.robot.GetMotorAngles()
    J, foot_pos = jacobian_rel(q=motor_ang)

    # Get foot velocity in leg frame (use GetMotorVelocities() )
    motor_vel = env.robot.GetMotorVelocities()
    foot_vel = np.matmul(J, motor_vel)

    # Calculate torque (Cartesian PD, and/or desired force)
    tau = np.matmul(np.transpose(J), np.matmul(kpCartesian, des_foot_pos - foot_pos) + np.matmul(kdCartesian, -foot_vel))

    # add gravity compensation (Force), (get mass with env.robot.total_mass)
    g = 9.81
    force = np.zeros(2)
    force[1] = env.robot.total_mass * g
    tau = tau + np.matmul(np.transpose(J), force)
    torques[:, i] = tau
    joint_positions[:, i] = motor_ang
    foot_positions[:, i] = foot_pos
    time[i] = i/dt
    action = tau
    # apply control, simulate
    env.step(action)

# make plots of joint positions, foot positions, torques, etc.

fig, ax = plt.subplots(2, 2)
"""Torques plot"""
ax[0, 0].plot(time, torques[0], label = 'torques for x')
ax[0, 0].plot(time, torques[1], label = 'torques for y')
ax[0, 0].set_xlabel('time [s]')
ax[0, 0].set_ylabel('torques [Nm]')
ax[0, 0].set_title('Torques')
ax[0, 0].legend()
"""joint position plot"""
ax[0, 1].plot(time, joint_positions[0], label = 'joint position for x')
ax[0, 1].plot(time, joint_positions[1], label = 'joint position for y')
ax[0, 1].set_xlabel('time [s]')
ax[0, 1].set_ylabel('joint positions [m]')
ax[0, 1].set_title('Joint positions')
ax[0, 1].legend()
"""foot position plot"""
ax[1, 0].plot(time, foot_positions[0], label = 'foot position for x')
ax[1, 0].plot(time, foot_positions[1], label = 'foot position for y')
ax[1, 0].set_xlabel('time [s]')
ax[1, 0].set_ylabel('foot positions [m]')
ax[1, 0].set_title('Feet positions')
ax[1, 0].legend()

plt.show()