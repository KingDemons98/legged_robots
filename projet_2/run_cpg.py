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

from optimize import Optimize
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
sideSign = np.array([-1, 1, -1, 1])  # get correct hip sign (body right is negative)


#######################
# parameters to run the code
####
# Optimization. Set the starting  value from which you want to optimize. Don't try to optimize parameters that have a
# limit.  Specify if you want to optimize for the slowest speed or the fastest. This method only looks at the average
# speed, and not the standard deviation. Because of this, it is better to use for slower speed optimization.
# then replace the parameter in the code by opt.new_val


optimization = True

start_val = 50
opti_slow = True  # slow if you want to optimize for slower speed, anything else for faster
nb_run_opt = 20


gait = "TROT"

###### plots variable
begin = 8000  # starting value for most plot that don't start at 0
begin_cot = 7500  # starting value for the zoomed plot, and the torque for high speed

plots_speed = False
plot_cpg = False
plot_CoT = False
plot_torque = False
compare_PD = False
test_cartesian = True
test_joint = True
save_plots = False  # this will save all the plots in the Pictures directory


# DON'T SET UP THIS WITH compare_PD

if test_cartesian & test_joint:
    pd_no_compare = 2
elif test_joint:
    pd_no_compare = 1
else: 
    pd_no_compare = 0
#################################

simulation_time = 10
TEST_STEPS = int(simulation_time / (TIME_STEP))

print_mode = ["with Cartesian PD", "with Joint PD", "with both PD"]
color_mode = ["#ff7f0e", "#2ca02c", "#1f77b4"]
if compare_PD:
    number_of_simulations = len(print_mode)
elif optimization:
    number_of_simulations = 5*nb_run_opt
    opt = Optimize(start_val, opti_slow, nb_run_opt)
else:
    number_of_simulations = 1

avg_vector = np.zeros([number_of_simulations, TEST_STEPS])
x_pos_vector = np.zeros([number_of_simulations, TEST_STEPS])
y_pos_vector = np.zeros([number_of_simulations, TEST_STEPS])

cot_compare = np.zeros([number_of_simulations, TEST_STEPS-begin_cot])
act_leg_pos_vector = np.zeros((number_of_simulations, 3, TEST_STEPS-begin))
act_joint_angles_vector = np.zeros((number_of_simulations, 3, TEST_STEPS-begin))
torque_plot = np.empty([12, TEST_STEPS]) # 12 dim: each joint of each leg,
compare_torque = np.empty([12, 3, TEST_STEPS])
labels_positions = np.array(["x", "y", "z"])
labels_joint = np.array(["hip", "thigh", "calf"])
colors = np.array(["b", "g", "r", "c"])
# legs = ["Front right", "Front left", "Rear Right", "Rear Left"]
legs = ["Front left", "Front right", "Rear Left", "Rear Right"]


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
        cpg = HopfNetwork(time_step=TIME_STEP, gait="TROT", omega_swing=2.4 * 2 * np.pi, omega_stance=0.9 * 2 * np.pi,
                          ground_penetration=0.008, alpha =41.9, ground_clearance=0.1, robot_height=0.315,  des_step_len = 0.049)
        # cpg = HopfNetwork(time_step=TIME_STEP, gait="TROT", omega_swing=2.4 * 2 * np.pi, omega_stance=0.9 * 2 * np.pi,
        #                   ground_penetration=0.008, alpha =45, ground_clearance=0.1, robot_height=0.32,  des_step_len = 0.058)
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

    des_joint_angles = np.zeros((3, TEST_STEPS))
    act_joint_angles = np.zeros((3, TEST_STEPS))

    r = np.zeros((4, TEST_STEPS))
    rdot = np.zeros((4, TEST_STEPS))
    theta = np.zeros((4, TEST_STEPS))
    theta_dot = np.zeros((4, TEST_STEPS))

    ############## Sample Gains
    # joint PD gains
    kp=np.array([100,100,100])
    kd=np.array([2,2,2])
    # Cartesian PD gains
    kpCartesian = np.diag([275.0]*3)  # 570 ca marche bien
    kdCartesian = np.diag([opt.new_val]*3)  # 19.025

    speeds = np.empty([5, TEST_STEPS]) # contains time, speed, Xpos,Ypos,CoT
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

        # get values for plot
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
            if test_joint:
                # Add joint PD contribution to tau for leg i (Equation 4)
                tau += kp * (leg_q - q[3*i:3*i+3]) + kd * (0 - dq[3*i:3*i+3])
            # print(f'leg q is {leg_q}')

            # get values for plots
            if i == 0:
                des_leg_pos[:, j] = leg_xyz
                des_joint_angles[:, j] = leg_q
                act_joint_angles[:, j] = q[0:3]

            # add Cartesian PD contribution
            if ADD_CARTESIAN_PD:
                # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
                jacob_cart, foot_pos = env.robot.ComputeJacobianAndPosition(i)

                # get values for plots
                if i == 0:
                    act_leg_pos[:, j] = foot_pos

                # Get current foot velocity in leg frame (Equation 2)
                foot_vel = np.matmul(jacob_cart, dq[3*i:3*i+3])
                if test_cartesian:
                    # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
                    tau += np.matmul(np.transpose(jacob_cart), np.matmul(kpCartesian, leg_xyz - foot_pos)
                                                             + np.matmul(kdCartesian, 0 - foot_vel))
            torque_plot[3*i:3*i+3, j] = tau

            # Set tau for legi in action vector
            action[3*i:3*i+3] = tau
        # send torques to robot and simulate TIME_STEP seconds
        env.step(action)
    if compare_PD:
        compare_torque[:, sim, :] = torque_plot

    ######################################################
    # PLOTS
    ######################################################

    avg = np.zeros([TEST_STEPS])
    n = 500
    for i in range(int(n/2), TEST_STEPS):
        if i < TEST_STEPS-(int(n/2)-1):
            avg[i] = np.average(speeds[1, int(i-n/2):int(i+n/2)]) # we find the moving average of the n/2 prev and next
        else:
            avg[i] = avg[i-1]  # to avoid array problems we just copy the precedent value


    # Mobile average of the speeds for smoothing
    avg = np.zeros([TEST_STEPS])
    n = 500
    m = 200
    for i in range(int(n/2), TEST_STEPS):
        if i < TEST_STEPS-(int(n/2)-1):
            avg[i] = np.average(speeds[1, int(i-n/2):int(i+n/2)]) # we find the moving average of the n/2 prev and next
        else:
            avg[i] = avg[i-1]  # to avoid array problems we just copy the precedent value
    avg_speed = np.average(avg[6000:9750])
    if optimization:
        opt.opt(avg_speed, sim)
        print("Step n° ", sim%nb_run_opt)
        if sim == number_of_simulations-1:
            print(" the best value is : ", opt.local_best_val, ", with a speed of : ", opt.local_best_speed)
    else:
        print(f"speed after convergence: {avg_speed}[m/s]")

    # plots the speed
    if plots_speed:
        fig_vel_pos, ax = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={'width_ratios': [2, 1]})
        if compare_PD:
            plt.get_current_fig_manager().set_window_title("speed & trajectory " + print_mode[sim])
        else:
            plt.get_current_fig_manager().set_window_title("speed & trajectory " + print_mode[pd_no_compare])

        # plots the speed
        ax1 = ax[0]
        ax1.plot(speeds[0, :], speeds[1, :], label="instant speed")
        ax1.plot(speeds[0, :], avg, label=f"instant speed (averaged over {int(n/2)} next and prev values")
        ax1.axhline(y = avg_speed, color = 'orange', linestyle = 'dashed', linewidth = 1 )
        # ax1.axvline(x = 1.75, color = 'orange', linewidth = 1)
        ax1.set(xlabel=f"time [s]\n speed after convergence: {avg_speed:.4f}[m/s]", ylabel="speed [m/s]")
        ax1.title.set_text("Instant and mobile averaged horizontal speed")
        ax1.legend()

        # plots the position
        x_scale = 15
        y_scale = 15
        ratio = x_scale/y_scale
        ax2 = ax[1]
        ax2.plot(speeds[2, :], speeds[3, :])
        ax2.axvline(x=0, color = 'green', linestyle = 'dashdot', linewidth = 0.5)
        ax2.axhline(y=0, color = 'green', linestyle = 'dashdot', linewidth = 0.5)
        ax2.set_xlim([-x_scale, x_scale])
        ax2.set_ylim([-y_scale, y_scale])
        ax2.set(xlabel="x displacement [m]", ylabel="y displacement [m]", label ="test")
        ax2.title.set_text("X-Y Trajectory")
        ax2.set_aspect('equal', 'box')
        fig_vel_pos.tight_layout()
        if save_plots:
            plt.savefig("Pictures/" + plt.get_current_fig_manager().get_window_title() + ".png")
    if compare_PD:
        avg_vector[sim,:] = avg
        x_pos_vector[sim, :] = speeds[2, :]
        y_pos_vector[sim, :] = speeds[3, :]


    # plots instant CoT

    if plot_CoT:
        if sim == 0:
            plt.figure()
            plt.get_current_fig_manager().set_window_title("CoT")
            plt.plot(speeds[0, :], speeds[4, :])
            plt.xlabel("time [s]")
            plt.ylabel("CoT")
            plt.title("Instant CoT of robot")

        if save_plots:
            plt.savefig("Pictures/" + plt.get_current_fig_manager().get_window_title() + ".png")
        plt.figure()
        if compare_PD:
            mode = sim
        else:
            mode = pd_no_compare

        plt.get_current_fig_manager().set_window_title("CoT " + print_mode[mode])

        plt.plot(speeds[0, begin_cot:], speeds[4, begin_cot:])
        avg_cot = np.average(speeds[4, begin_cot:])
        plt.axhline(y=avg_cot, color = 'orange', linestyle = 'dashed', linewidth = 1)
        print("average Cot is : ",  avg_cot, " ", print_mode[mode])
        plt.xlabel("time [s]")
        plt.ylabel("CoT")
        plt.title(f"Instant CoT of robot over the last {(10000-begin_cot)/1000} [s]")
        if save_plots:
            plt.savefig("Pictures/" + plt.get_current_fig_manager().get_window_title() + ".png")
    if compare_PD:
        cot_compare[sim, :] = speeds[4, begin_cot:]


    if plot_cpg:

        if sim == 0:
            fig = plt.figure(figsize =(8.5, 6))
            plt.get_current_fig_manager().set_window_title("r and theta ")
            subfigs = fig.subfigures(2, 2, wspace=0.07)

            labels = np.array(["time [s]", "amplitudes"])
            ax1 = subfigs[0, 0].subplots(4, sharex=True)

            subfigs[0, 0].suptitle("amplitude of oscillators (r)")
            for i, ax in enumerate(ax1):
                ax.plot(t[:], r[i, :], label = 'leg' + str(i), color = colors[i])
                ax.grid(True)
                if i == 1:
                    ax.set_ylabel(labels[1], loc="bottom")
                ax.legend()
            plt.xlabel(labels[0])
            subfigs[0, 0].subplots_adjust(left = 0.15, hspace = 0.4, bottom = 0.15, right = 0.95)

            labels = np.array(["time [s]", "angles [rad]"])
            ax2 = subfigs[0, 1].subplots(4, sharex=True)
            subfigs[0, 1].suptitle(r"angles of oscillators ($\theta$)")
            for i, ax in enumerate(ax2):
                ax.plot(t[begin:], theta[i, begin:], label = 'leg' + str(i), color = colors[i])
                ax.grid(True)
                if i == 1:
                    ax.set_ylabel(labels[1], loc="bottom")
                ax.legend()
            plt.xlabel(labels[0])

            labels = np.array(["time [s]", "derivate of amplitude"])
            ax3 = subfigs[1, 0].subplots(4, sharex=True)
            subfigs[1, 0].suptitle(r"derivative of amplitude ($\dot r$)")
            for i, ax in enumerate(ax3):
                ax.plot(t[:], rdot[i, :], label = 'leg' + str(i), color = colors[i])
                ax.grid(True)
                if i == 1:
                    ax.set_ylabel(labels[1], loc="top")
                ax.legend()
            plt.xlabel(labels[0])

            labels = np.array(["time [s]", "angular velocity [rad/s]"])
            ax4 = subfigs[1, 1].subplots(4, sharex=True)
            subfigs[1, 1].suptitle(r"Angular velocity ($\dot \theta$)")
            for i, ax in enumerate(ax4):
                ax.plot(t[begin:], theta_dot[i, begin:], label = 'leg' + str(i), color = colors[i])
                ax.grid(True)
                if i == 1:
                    ax.set_ylabel(labels[1], loc="top")
                ax.legend()
            plt.xlabel(labels[0])
            if save_plots:
                plt.savefig("Pictures/" + plt.get_current_fig_manager().get_window_title() + ".png")

        ##### Value comparison between desired and real##################################

        fig = plt.figure(figsize =(10, 5))
        if compare_PD:
            plt.get_current_fig_manager().set_window_title("foot & angle position " + print_mode[sim])
        else:
            plt.get_current_fig_manager().set_window_title("foot & angle position " + print_mode[pd_no_compare])
        subfigs = fig.subfigures(1, 2, wspace=0.07)

        labels = np.array(["time [s]", "Position [m]"])
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

        labels = np.array(["time [s]", "angle [rad]"])
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
        if save_plots:
            plt.savefig("Pictures/" + plt.get_current_fig_manager().get_window_title() + ".png")
    if compare_PD:
        act_leg_pos_vector[sim, :, :] = act_leg_pos[:, begin:]
        act_joint_angles_vector[sim, :, :] = act_joint_angles[:, begin:]

    if plot_torque:
        fig_torque, ax = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [1, 1]})
        fig_torque.suptitle("Torque for each joints of each legs", fontsize=16)
        if compare_PD:
            plt.get_current_fig_manager().set_window_title("Torque  " + print_mode[sim])
        else:
            plt.get_current_fig_manager().set_window_title("Torque  " + print_mode[pd_no_compare])

        for i in range(2):
            for j in range(2):
                for m in range(len(labels_joint)):
                    ax[i, j-1].plot(t[begin:], torque_plot[3*(2*i+j)+m, begin:], label = "Torque for the joint on the " + labels_joint[m],
                                  color=colors[m]) #the j-1 is here in order to place the graph of the left leg on the left

                ax[i, j].grid(True)
                ax[i, j].set_title(legs[2*i+j])
        lines, labels_torque = ax[0,0].get_legend_handles_labels()
        fig_torque.legend(lines, labels_torque, loc='lower right')
        fig_torque.supxlabel("Time [s]")
        fig_torque.supylabel("Torque [Nm]")
        if save_plots:
            plt.savefig("Pictures/" + plt.get_current_fig_manager().get_window_title() + ".png")


# np.savetxt('torque.csv', torque_plot, delimiter=',')
#
# np.savetxt('time.csv', t, delimiter=',')
####################################################################################################################
if compare_PD:

    #### speed comparison
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={'width_ratios': [2, 1]})
    plt.get_current_fig_manager().set_window_title("Speed comparison")
    for sim in range(number_of_simulations):
        ax[0].plot(avg_vector[sim, :], label ="average speed " + print_mode[sim], color = color_mode[sim])
        ax[1].plot(x_pos_vector[sim, :], y_pos_vector[sim, :], label ="trajectory " + print_mode[sim],
                   color = color_mode[sim])
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
    if save_plots:
        plt.savefig("Pictures/" + plt.get_current_fig_manager().get_window_title() + ".png")

    #### cot comparison
    fig_cot = plt.figure()
    plt.get_current_fig_manager().set_window_title("CoT comparison")
    for sim in range(number_of_simulations):
        plt.plot(speeds[0, begin_cot:], cot_compare[sim, :], label="CoT " + print_mode[sim], color = color_mode[sim])
    plt.xlabel("time [s]")
    plt.ylabel("CoT")
    plt.title(f"Instant CoT of robot over the last {(10000-begin_cot)/1000} [s]")
    plt.legend()
    if save_plots:
        plt.savefig("Pictures/" + plt.get_current_fig_manager().get_window_title() + ".png")

    #### foot pos comparison
    handles = []
    plot_labels = []
    fig = plt.figure(figsize=(10, 5))
    plt.get_current_fig_manager().set_window_title("Foot position & Joint angle comparison")
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    labels = np.array(["time [s]", "Position [m]"])
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
    subfigs[0].subplots_adjust(left=0.2, bottom=0.09, right=0.96, top=0.9)
    plt.xlabel(labels[0])

    labels = np.array(["time [s]", "angle [rad]"])
    ax2 = subfigs[1].subplots(3, sharex=True)
    for i, ax in enumerate(ax2):
        ax.set_title("actual joint position for the " + labels_joint[i])
        for sim in range(number_of_simulations):
            ax.plot(t[begin:], act_joint_angles_vector[sim, i, :], label= " " + print_mode[sim],
                    color=color_mode[sim])
        ax.grid(True)
        if i == 1:
            ax.set_ylabel(labels[1])

    subfigs[1].legend(handles, plot_labels, loc='lower right')
    plt.xlabel(labels[0], loc='center')
    if save_plots:
        plt.savefig("Pictures/" + plt.get_current_fig_manager().get_window_title() + ".png")


#### torque plot
    idx = 0
    fig_torque, ax = plt.subplots(3, 1, figsize=(7.5, 6))
    plt.get_current_fig_manager().set_window_title("Torque comparison")
    fig_torque.suptitle("Torque's comparison with different PD's  with the " + legs[idx] + " leg", fontsize=16)
    # this time we only look at one leg, but 3 different PD

    for m in range(len(labels_joint)):
        for sim in range(number_of_simulations):
            ax[m].plot(t[begin:], compare_torque[3*idx + m, sim, begin:], label="Torque " + print_mode[sim],
                       color=colors[sim])

        ax[m].grid(True)
        ax[m].set_title("Torque for the joint on the " + labels_joint[m])

    fig_torque.subplots_adjust(hspace=0.45)
    lines, labels_torque = ax[0].get_legend_handles_labels()
    fig_torque.legend(lines, labels_torque, loc='lower left')
    fig_torque.supxlabel("Time [s]")
    fig_torque.supylabel("Torque [Nm]")
    if save_plots:
        plt.savefig("Pictures/" + plt.get_current_fig_manager().get_window_title() + ".png")

############################################################
# end of plot
############################################################

plt.show()

