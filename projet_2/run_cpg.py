# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

""" Run CPG """
import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv


ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

gait = "TROT"

###### plots variable
plots_speed = True
plot_cpg = True
plot_CoT = True
compare_PD = False
test_cartesian = True
test_joint = False

#################################

simulation_time = 10
TEST_STEPS = int(simulation_time / (TIME_STEP))

print_mode = ["with Cartesian PD", "with Joint PD", "with both PD"]
color_mode = ["r", "g", "b"]
if compare_PD:
    number_of_simulations = len(print_mode)
else:
    number_of_simulations = 1

avg_vector = np.zeros([number_of_simulations, TEST_STEPS])
x_pos_vector = np.zeros([number_of_simulations, TEST_STEPS])
y_pos_vector = np.zeros([number_of_simulations, TEST_STEPS])
begin = 9000 # for the plots
begin_cot = 9500 # for the plots

cot_avg = np.zeros([number_of_simulations, TEST_STEPS-begin_cot])
act_leg_pos_vector = np.zeros((number_of_simulations, 3, TEST_STEPS-begin))
act_joint_angles_vector = np.zeros((number_of_simulations, 3, TEST_STEPS-begin))



for sim in range(number_of_simulations):
    env = QuadrupedGymEnv(render=False,             # visualize
                        on_rack=False,              # useful for debugging!
                        isRLGymInterface=False,     # not using RL
                        time_step=TIME_STEP,
                        action_repeat=1,
                        motor_control_mode="TORQUE",
                        add_noise=False,    # start in ideal conditions
                        # record_video=True
                        )
    if compare_PD:
        if sim == 0:
            test_cartesian = True
            test_joint = False
        elif sim == 1 :
            test_cartesian = False
            test_joint = True
        else:
            test_cartesian = True
            test_joint = True

        # initialize Hopf Network, supply gait
    if(gait == "TROT"):
        cpg = HopfNetwork(time_step=TIME_STEP, gait="TROT", omega_swing=8.2 * 2 * np.pi, omega_stance=6.4 * 2 * np.pi,
                          ground_penetration=0.015, alpha =50, ground_clearance=0.07, robot_height=0.26,  des_step_len = 0.064)
    elif(gait == "BOUND"):
        cpg = HopfNetwork(time_step=TIME_STEP, gait="BOUND", omega_swing=5 * 2 * np.pi, omega_stance=2 * 2 * np.pi)
    elif(gait == "WALK"):
        cpg = HopfNetwork(time_step=TIME_STEP, gait="WALK", omega_swing=5 * 2 * np.pi, omega_stance=2 * 2 * np.pi)
    elif(gait == "PACE"):
        cpg = HopfNetwork(time_step=TIME_STEP, gait="PACE", omega_swing=5 * 2 * np.pi, omega_stance=2 * 2 * np.pi)
    elif(gait == "ROTARY_GALLOP"):
        cpg = HopfNetwork(time_step=TIME_STEP, gait="ROTARY_GALLOP", omega_swing=5 * 2 * np.pi, omega_stance=2 * 2 * np.pi)
    else:
        print("error: gait not supported, using the default one")
        cpg = HopfNetwork(time_step=TIME_STEP)



    t = np.arange(TEST_STEPS)*TIME_STEP

    des_leg_pos = np.zeros((3, TEST_STEPS))
    act_leg_pos = np.zeros((3, TEST_STEPS))

    des_joint_angles = np.zeros((3, TEST_STEPS)) #peut etre pas suffisant
    act_joint_angles = np.zeros((3, TEST_STEPS))

    r = np.zeros((4, TEST_STEPS))
    rdot = np.zeros((4, TEST_STEPS))
    theta = np.zeros((4, TEST_STEPS))
    theta_dot = np.zeros((4, TEST_STEPS))

    ############## Sample Gains
    # joint PD gains
    kp=np.array([100,87,87.5])
    kd=np.array([1.98,2.02,2])
    # Cartesian PD gains
    kpCartesian = np.diag([929.2]*3)
    kdCartesian = np.diag([17.8]*3)

    speeds = np.empty([5, TEST_STEPS]) #contains time, speed, Xpos,Ypos,CoT
    mass = np.sum(env.robot.GetTotalMassFromURDF())
    G=9.81
    # for u in range(5):
    for j in range(TEST_STEPS):
        # initialize torque array to send to motors
        action = np.zeros(12)
        # get desired foot positions from CPG
        xs,zs = cpg.update()
        q = env.robot.GetMotorAngles()
        dq = env.robot.GetMotorVelocities()
        torques = env.robot.GetMotorTorques()

        speed = np.linalg.norm(env.robot.GetBaseLinearVelocity()[:-1])
        speeds[0, j] = j * TIME_STEP  # time
        speeds[1, j] = speed  # instant speed

        x, y, z = env.robot.GetBasePosition()
        speeds[2, j] = x  # Xpos
        speeds[3, j] = y  # Ypos

        power = np.sum(np.abs(np.multiply(dq, torques)))

        speeds[4, j] = power / (speed * mass * G)  # CoT

        #get values for plot
        r[:, j] = cpg.get_r()
        rdot[:, j] = cpg.get_dr()
        theta[:, j] = cpg.get_theta()
        theta_dot[:, j] = cpg.get_dtheta()

         # loop through desired foot positions and calculate torques
        for i in range(4):
            # initialize torques for legi
            tau = np.zeros(3)
            # get desired foot i pos (xi, yi, zi) in leg frame
            leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])
            # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
            leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)
            if(test_joint == True):
                # Add joint PD contribution to tau for leg i (Equation 4)
                tau += kp * (leg_q - q[3*i:3*i+3]) + kd * (0 - dq[3*i:3*i+3])

            # print(f'leg q is {leg_q}')

            #get values for plots
            if (i == 0):
                des_leg_pos[:, j] = leg_xyz
                des_joint_angles[:, j] = leg_q
                act_joint_angles[:, j] = q[0:3]

            # add Cartesian PD contribution
            if ADD_CARTESIAN_PD:
                # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
                jacob_cart, foot_pos = env.robot.ComputeJacobianAndPosition(i)

                #get values for plots
                if (i == 0):
                    act_leg_pos[:, j] = foot_pos

                # Get current foot velocity in leg frame (Equation 2)
                foot_vel = np.matmul(jacob_cart, dq[3*i:3*i+3])
                if(test_cartesian == True):
                    # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
                    tau += np.matmul(np.transpose(jacob_cart), np.matmul(kpCartesian, leg_xyz - foot_pos) + np.matmul(kdCartesian, 0 - foot_vel))

            # Set tau for legi in action vector
            action[3*i:3*i+3] = tau

        # send torques to robot and simulate TIME_STEP seconds
        env.step(action)

    ######################################################
    # PLOTS
    ######################################################

    avg = np.zeros([TEST_STEPS])
    n = 500
    for i in range(int(n/2), TEST_STEPS):
        if i < TEST_STEPS-(int(n/2)-1):
            avg[i] = np.average(speeds[1, int(i-n/2):int(i+n/2)]) #we find the moving average of the n/2 prev and next
        else:
            avg[i] = avg[i-1] #to avoid array problems we just copy the precedent value


    # plots the speed
    if plots_speed:
        # Mobile average of the speeds for smoothing
        avg = np.zeros([TEST_STEPS])
        n = 500
        m = 200
        for i in range(int(n/2), TEST_STEPS):
            if i < TEST_STEPS-(int(n/2)-1):
                avg[i] = np.average(speeds[1, int(i-n/2):int(i+n/2)]) #we find the moving average of the n/2 prev and next
            else:
                avg[i] = avg[i-1] #to avoid array problems we just copy the precedent value

        print(f"speed after convergence: {avg[-1]}[m/s]")
        # print(f"Average CoT of the last {m} timesteps{np.average(speeds[4,-m:])} ")
        # print(f"Total average CoT {np.average(speeds[4,:])} ")


        fig_vel_pos, ax = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={'width_ratios': [2, 1]})

        # plots the speed
        ax1 = ax[0]
        ax1.plot(speeds[0, :], speeds[1, :], label="instant speed")
        ax1.plot(speeds[0, :], avg, label=f"instant speed (averaged over {int(n/2)} next and prev values")
        ax1.axhline(y = avg[-1], color = 'orange', linestyle = 'dashed', linewidth = 1 )
        ax1.axvline(x = 1.75, color = 'orange', linewidth = 1)
        ax1.set(xlabel=f"time [s]\n speed after convergence: {avg[-1]:.4f}[m/s]", ylabel="speed [m/s]")
        ax1.title.set_text("Instant and mobile averaged horizontal speed")
        # ax1.set_box_aspect(0.4)
        ax1.legend()

        # plots the position
        x_scale = 15
        y_scale = 15
        ratio = x_scale/y_scale
        ax2 = ax[1]
        ax2.plot(speeds[2, :], speeds[3, :])
        ax2.axvline(x=0, color = 'green', linestyle = 'dashdot', linewidth = 0.5)
        ax2.axhline(y=0, color = 'green', linestyle = 'dashdot', linewidth = 0.5)
        # ax2.grid(color = 'green', linestyle = '--', linewidth = 0.5)
        ax2.set_xlim([-x_scale, x_scale])
        ax2.set_ylim([-y_scale, y_scale])
        ax2.set(xlabel="x displacement [m]", ylabel="y displacement [m]", label ="test")
        ax2.title.set_text("X-Y Trajectory")
        ax2.set_aspect('equal', 'box')
        # ax2.set_box_aspect(1/ratio)

        fig_vel_pos.tight_layout()
        avg_vector[sim,:] = avg
        x_pos_vector[sim, :] = speeds[2, :]
        y_pos_vector[sim, :] = speeds[3, :]


    # plots instant CoT

    if plot_CoT:
        if sim == 0:
            plt.figure()
            plt.plot(speeds[0, :], speeds[4, :])
            plt.xlabel("time [s]")
            plt.ylabel("CoT")
            plt.title("Instant CoT of robot")

            plt.figure()
            plt.plot(speeds[0, begin_cot:], speeds[4, begin_cot:])
            avg_cot = np.average(speeds[4, begin_cot:])
            plt.axhline(y=avg_cot, color = 'orange', linestyle = 'dashed', linewidth = 1)
            plt.xlabel("time [s]")
            plt.ylabel("CoT")
            plt.title(f"Instant CoT of robot over the last {(10000-begin_cot)/1000} [s]")

        cot_avg[sim, :] = speeds[4, begin_cot:]


    if plot_cpg:
        colors = np.array(["b", "g", "r", "c"])

        # fig = plt.figure("CPG")
        if sim == 0:
            fig = plt.figure(figsize =(8.5, 6))
            subfigs = fig.subfigures(2, 2, wspace=0.07)

            labels = np.array(["time [s]", "amplitudes []"])
            ax1 = subfigs[0, 0].subplots(4, sharex=True)

            subfigs[0, 0].suptitle("amplitude of oscillators (r)")
            for i, ax in enumerate(ax1):
                ax.plot(t[begin:], r[i, begin:], label = 'leg' + str(i), color = colors[i])
                ax.grid(True)
                if i == 1:
                    ax.set_ylabel(labels[1], loc="bottom")
                ax.legend()
            plt.xlabel(labels[0])
            subfigs[0, 0].subplots_adjust(left = 0.15, hspace = 0.4, bottom = 0.15, right = 0.95)

            labels = np.array(["time [s]", "angles [rad]"])
            ax2 = subfigs[0, 1].subplots(4, sharex=True)
            subfigs[0, 1].suptitle("angles of oscillators (theta)")
            for i, ax in enumerate(ax2):
                ax.plot(t[begin:], theta[i, begin:], label = 'leg' + str(i), color = colors[i])
                ax.grid(True)
                if i == 1:
                    ax.set_ylabel(labels[1], loc="bottom")
                ax.legend()
            plt.xlabel(labels[0])

            labels = np.array(["time [s]", "derivate of amplitude []"])
            ax3 = subfigs[1, 0].subplots(4, sharex=True)
            subfigs[1, 0].suptitle("derivative of amplitude (r dot)")
            for i, ax in enumerate(ax3):
                ax.plot(t[begin:], rdot[i, begin:], label = 'leg' + str(i), color = colors[i])
                ax.grid(True)
                if i == 1:
                    ax.set_ylabel(labels[1], loc="top")
                ax.legend()
            plt.xlabel(labels[0])

            labels = np.array(["time [s]", "angular velocity [rad/s]"])
            ax4 = subfigs[1, 1].subplots(4, sharex=True)
            subfigs[1, 1].suptitle("Angular velocity (theta dot)")
            for i, ax in enumerate(ax4):
                ax.plot(t[begin:], theta_dot[i, begin:], label = 'leg' + str(i), color = colors[i])
                ax.grid(True)
                if i == 1:
                    ax.set_ylabel(labels[1], loc="top")
                ax.legend()
            plt.xlabel(labels[0])

        ##### Value comparison between desired and real##################################

        fig = plt.figure(figsize =(10, 5))
        subfigs = fig.subfigures(1, 2, wspace=0.07)

        labels = np.array(["time [s]", "X [m]"])
        labels_positions = np.array(["x", "y", "z"])
        labels_joint = np.array(["hip", "thigh", "calf"])
        ax1 = subfigs[0].subplots(3, 1, sharex=True)
        subfigs[0].suptitle("foot positions")
        for i, ax in enumerate(ax1):
            ax.plot(t[begin:], des_leg_pos[i, begin:], label = "desired foot position for " + labels_positions[i])
            ax.plot(t[begin:], act_leg_pos[i, begin:], label = "actual foot position for " + labels_positions[i], color = "r")
            ax.grid(True)
            if i == 1:
                ax.set_ylabel(labels[1])
            ax.legend()
        subfigs[0].subplots_adjust(left = 0.2, bottom = 0.09, right = 0.96, top = 0.9)
        plt.xlabel(labels[0])

        labels = np.array(["time [s]", "joint angle [rad]"])
        ax2 = subfigs[1].subplots(3, sharex=True)
        subfigs[1].suptitle("joint positions")
        for i, ax in enumerate(ax2):
            ax.plot(t[begin:], des_joint_angles[i, begin:], label="desired joint position for joint " + labels_joint[i])
            ax.plot(t[begin:], act_joint_angles[i, begin:], label="actual joint position for joint " + labels_joint[i], color="r")
            ax.grid(True)
            if i == 1:
                ax.set_ylabel(labels[1])
            ax.legend()
        plt.xlabel(labels[0], loc= 'center')

        act_leg_pos_vector[sim, :, :] = act_leg_pos[:, begin:]
        act_joint_angles_vector[sim, :, :] = act_joint_angles[:, begin:]

####################################################################################################################
if compare_PD:

    #### speed comparison
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={'width_ratios': [2, 1]})
    for sim in range(number_of_simulations):
        ax[0].plot(avg_vector[sim, :], label ="average speed " + print_mode[sim])
        ax[1].plot(x_pos_vector[sim, :], y_pos_vector[sim, :], label ="trajectory " + print_mode[sim])
    ax[0].set(xlabel="time [s]", ylabel="speed [m/s]")
    ax[0].legend()
    x_scale = 15
    y_scale = 15
    ratio = x_scale/y_scale
    ax[1].axvline(x=0, color = 'green', linestyle = 'dashdot', linewidth = 0.5)
    ax[1].axhline(y=0, color = 'green', linestyle = 'dashdot', linewidth = 0.5)
    ax[1].set_xlim([-x_scale, x_scale])
    ax[1].set_ylim([-y_scale, y_scale])
    ax[1].set(xlabel="x displacement [m]", ylabel="y displacement [m]", label ="test")
    ax[1].title.set_text("X-Y Trajectory")
    ax[1].set_aspect('equal', 'box')
    ax[1].legend()
    fig.tight_layout()

    #### cot comparison
    fig_cot = plt.figure()
    for sim in range(number_of_simulations):
        plt.plot(speeds[0, begin_cot:], cot_avg[sim, :], label="CoT " + print_mode[sim])
    plt.xlabel("time [s]")
    plt.ylabel("CoT")
    plt.title(f"Instant CoT of robot over the last {(10000-begin_cot)/1000} [s]")
    plt.legend()

    #### foot pos comparison
    handles = []
    plot_labels = []
    fig = plt.figure(figsize=(10, 5))
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    labels = np.array(["time [s]", "X [m]"])
    labels_positions = np.array(["x", "y", "z"])
    labels_joint = np.array(["hip", "thigh", "calf"])
    ax1 = subfigs[0].subplots(3, 1, sharex=True)
    for i, ax in enumerate(ax1):
        ax.set_title("actual foot positions for " + labels_positions[i])
        for sim in range(number_of_simulations):
            ax.plot(t[begin:], act_leg_pos_vector[sim, i, :], label= " " + print_mode[sim],
                    color=color_mode[sim])
        ax.grid(True)
        if i == 1:
            ax.set_ylabel(labels[1])
            handles, plot_labels = ax.get_legend_handles_labels()
            # print(handles)
            # print(plot_labels)
        # ax.legend()
    subfigs[0].subplots_adjust(left=0.2, bottom=0.09, right=0.96, top=0.9)
    plt.xlabel(labels[0])

    labels = np.array(["time [s]", "joint angle [rad]"])
    ax2 = subfigs[1].subplots(3, sharex=True)
    for i, ax in enumerate(ax2):
        ax.set_title("actual joint position for the " + labels_joint[i])
        for sim in range(number_of_simulations):
            ax.plot(t[begin:], act_joint_angles_vector[sim, i, :], label= " " + print_mode[sim],
                    color=color_mode[sim])
        ax.grid(True)
        if i == 1:
            ax.set_ylabel(labels[1])
        # ax.legend()

    subfigs[1].legend(handles, plot_labels, loc='lower right')

    # subfigs[0].legend(handles, plot_labels, loc='center right')
    # fig.axes[-1].legend( loc='lower right') #print_mode[sim] for sim in range(number_of_simulations)
    plt.xlabel(labels[0], loc='center')


############################################################
# end of plot
############################################################

plt.show()

