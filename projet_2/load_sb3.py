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

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results

LEARNING_ALG = "PPO"
EPISODE_LENGTH = 2

#plot graphs or not
##########################################################################
plot_cpg = False
plot_foot_pos = True
plot_speed_pos = True
plot_training = False
plot_CoT = True
plot_torque = True

###########################################################################
# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config['render'] =  False
env_config['record_video'] = False
env_config['add_noise'] = False 
env_config['competition_env'] = True            #### SET COMPET ENV HERE
# env_config['test_env'] = True

motor_control_mode = "CARTESIAN_PD"                   ##### SET MOTOR CONTROL HERE

############################################### CPG_RL ##########################################################
env_config['motor_control_mode'] = motor_control_mode
env_config['observation_space_mode'] = "LR_COURSE_OBS_V2"
env_config['task_env'] = "CARTESIAN_RWD"
#################################################################################################################

if motor_control_mode == "CPG":
    interm_dir = "./logs/intermediate_models_cpg/"
elif motor_control_mode == "CARTESIAN_PD":
    interm_dir = "./logs/intermediate_models_cartesian_pd/"
elif motor_control_mode == "PD":
    interm_dir = "./logs/intermediate_models_pd/"
else:
    interim_dir = "./logs/intermediate_models/"

#interm_dir = "./logs/comparison-joint-cart-cpg/"


# path to saved models, i.e. interm_dir + '121321105810'
# log_dir = interm_dir + '112622123152'
# log_dir = interm_dir + 'cpg_rl_120122090257'                #supposed to work but already a t 1.5?
# log_dir = interm_dir + 'cpg_rl_112922155242_vel_1.0'                   # this is the last one with 1.0
# log_dir = interm_dir + 'cpg_rl_112822072518'
# log_dir = interm_dir + 'cpg_rl_test_env120622075011'            #test avec obstacles en train
# log_dir = interm_dir + 'CPG_120622193517'                           # 2eme avec nouveaux cpg?
# log_dir = interm_dir + 'CPG_120822101153'                       #moving in y but not ok yet

# log_dir = interm_dir + 'CPG_120822174454'            # new reward fct avec deplacement en y

# log_dir = interm_dir + 'CPG_cart_solve_y_121222100704'      #maybe the right for y

# log_dir = interm_dir + 'CPG_cart_solve_y_121122161834'

# log_dir = interm_dir + 'CPG_y_displacement_121322165520'

# log_dir = interm_dir + 'CPG_y_displacement_121422150803'            #test with in y, might work
##########################################################################################################################################################################

# log_dir = interm_dir + 'CPG_test_best_run_follow_new_rwd_121822234256'

# log_dir = interm_dir + 'CPG_120822174454'            # new reward fct avec deplacement en y

# log_dir = interm_dir + 'CPG_best_run_follow_121022163418'       #funny but bad

# log_dir = interm_dir + 'CPG_test_old_y_121522221641'

# log_dir = interm_dir + 'CPG_test_y_offset_in_hopf121722100211'          #try with y offset in hopf

# log_dir = interm_dir + 'CPGCPG_test_y_offset_only_in_cart_121722174616'    # try with y offset only for cart (not joint)

log_dir = interm_dir + 'CPG_test_with_psi_limited_122122232452'

# log_dir = interm_dir + 'CPG_CPG_y_corrected_try_without_cart121522141406'   #new try without cartesian

# log_dir = interm_dir + 'CPG_cpg_backward_121222204813'          #backward

# log_dir = interm_dir + 'CPG_121022113942'            #test with in y, might work

log_dir = interm_dir + 'CARTESIAN_PD_cartesian_rwd_old_scaling_121922150148 - speedy'            #test with in y, might work

# log_dir = interm_dir + 'CPG_test_added_ztrain_to_actionS_121822171359'          #added z to action space
############################################################################################################################################################33


# # get latest model and normalization stats, and plot
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
if plot_training:
    print(monitor_results)
    plot_results([log_dir], 10e10, 'timesteps', LEARNING_ALG + ' ')
# plt.show()

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation

leg_pos_tab = np.empty((1, 3))
des_leg_pos_tab = np.empty((1, 3))

leg_torque_tab = np.empty((1, 3))
des_leg_torque_tab = np.empty((1, 3))

r_tab = np.empty((1, 4))
rdot_tab = np.empty((1, 4))
theta_tab = np.empty((1, 4))
thetadot_tab = np.empty((1, 4))
# #
robot_pos_tab = np.empty((1, 3))
robot_speed_tab = np.empty((1, 3)) # speed in x y and z

CoT_tab = np.empty([1])

ROBOT_MASS = np.sum(env.envs[0].env.robot.GetTotalMassFromURDF())
G = 9.81


for i in range(100 * EPISODE_LENGTH):
    action, _states = model.predict(obs, deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0
        final_time = info[0]['episode']['t'] / 100
        print(f'Final time before failure is {final_time} [s]')
        break
    # [TODO] save data from current robot states for plots

    # collection of robot states
    dq = np.array(env.envs[0].env.robot.GetMotorVelocities()).reshape(1, -1)
    torques = env.envs[0].env.robot.GetMotorTorques()
    speed = np.array(env.envs[0].env.robot.GetBaseLinearVelocity()).reshape(1, -1)  # [:-1]
    base_pos = np.array(env.envs[0].env.robot.GetBasePosition()).reshape(1, -1)
    _, curr_leg_pos = env.envs[0].env.robot.ComputeJacobianAndPosition(0) # leg 0
    curr_leg_pos = np.array(curr_leg_pos).reshape(1, -1)
    des_leg_torque = np.array(env.envs[0].env.get_des_torques[:3]).reshape(1, -1)

    # CoT computation
    power = np.sum(np.abs(np.multiply(dq, torques)))
    CoT = np.array(power / (np.linalg.norm(speed) * ROBOT_MASS * G)).reshape(1)  # CoT

    torques = np.array(env.envs[0].env.robot.GetMotorTorques()[:3]).reshape(1, -1)


    # Updating arrays
    robot_speed_tab = np.append(robot_speed_tab, speed, axis=0)
    robot_pos_tab = np.append(robot_pos_tab, base_pos, axis=0)
    leg_pos_tab = np.append(leg_pos_tab, curr_leg_pos, axis=0)
    CoT_tab = np.append(CoT_tab, CoT, axis=0)
    leg_torque_tab = np.append(leg_torque_tab, torques, axis=0)
    des_leg_torque_tab = np.append(des_leg_torque_tab, des_leg_torque, axis=0)

    if motor_control_mode == "CPG":
        # collection of CPG states
        xs, ys, zs = env.envs[0].env._cpg.update()
        r = np.array(env.envs[0].env._cpg.get_r()).reshape(1, -1)
        rdot = np.array(env.envs[0].env._cpg.get_dr()).reshape(1, -1)
        theta = np.array(env.envs[0].env._cpg.get_theta()).reshape(1, -1)
        thetadot = np.array(env.envs[0].env._cpg.get_dtheta()).reshape(1, -1)

        # Updating arrays
        des_leg_pos_tab = np.append(des_leg_pos_tab, np.array([[xs[0], ys[0], zs[0]]]), axis=0)
        r_tab = np.append(r_tab, r, axis=0)
        rdot_tab = np.append(rdot_tab, rdot, axis=0)
        theta_tab = np.append(theta_tab, theta, axis=0)
        thetadot_tab = np.append(thetadot_tab, thetadot, axis=0)

    else:
        new_des_leg_pos = np.array(env.envs[0].env.get_des_pos[:3]).reshape(1, -1)
        des_leg_pos_tab = np.append(des_leg_pos_tab, new_des_leg_pos, axis=0)

# [TODO] make plots:

if not dones:
    final_time = EPISODE_LENGTH

t = np.arange(0, final_time, final_time / len(des_leg_pos_tab))

if plot_cpg and motor_control_mode == "CPG":
    colors = np.array(["b", "g", "r", "c"])
    fig = plt.figure()
    subfigs = fig.subfigures(2, 2, wspace=0.07)

    labels = np.array(["time [s]", "amplitudes"])
    labels = np.array(["time [s]", "amplitudes"])
    ax1 = subfigs[0, 0].subplots(4, sharex=True)
    subfigs[0, 0].suptitle("amplitude of oscillators (r)")
    for i, ax in enumerate(ax1):
        ax.plot(t, r_tab[:, i], label='leg' + str(i), color=colors[i])
        ax.grid(True)
        if i == 1:
            ax.set_ylabel(labels[1], loc="bottom")
        ax.legend()
    plt.xlabel(labels[0])

    labels = np.array(["time [s]", "angles [rad]"])
    ax2 = subfigs[0, 1].subplots(4, sharex=True)
    subfigs[0, 1].suptitle("angles of oscillators (theta)")
    for i, ax in enumerate(ax2):
        ax.plot(t, theta_tab[:, i], label='leg' + str(i), color=colors[i])
        ax.grid(True)
        if i == 1:
            ax.set_ylabel(labels[1], loc="bottom")
        ax.legend()
    plt.xlabel(labels[0])

    labels = np.array(["time [s]", "derivate of amplitude"])
    labels = np.array(["time [s]", "derivate of amplitude"])
    ax3 = subfigs[1, 0].subplots(4, sharex=True)
    subfigs[1, 0].suptitle("derivative of amplitude (r dot)")
    for i, ax in enumerate(ax3):
        ax.plot(t, rdot_tab[:, i], label='leg' + str(i), color=colors[i])
        ax.grid(True)
        if i == 1:
            ax.set_ylabel(labels[1], loc="top")
        ax.legend()
    plt.xlabel(labels[0])

    labels = np.array(["time [s]", "angular velocity [rad/s]"])
    ax4 = subfigs[1, 1].subplots(4, sharex=True)
    subfigs[1, 1].suptitle("Angular velocity (theta dot)")
    for i, ax in enumerate(ax4):
        ax.plot(t, thetadot_tab[:, i], label='leg' + str(i), color=colors[i])
        ax.grid(True)
        if i == 1:
            ax.set_ylabel(labels[1], loc="top")
        ax.legend()
    plt.xlabel(labels[0])

######################################################################

# Plot Cost of transport
if plot_CoT:
    fig = plt.figure()
    plt.plot(t, CoT_tab)
    plt.title("Cost of transport")
    plt.xlabel("time [s]")
    plt.ylabel("instant CoT [-]")

# Plot foot positions current and desired and foot torques current and desired
if plot_foot_pos:
    fig = plt.figure(figsize=(10, 8))

    fig.suptitle("Desired positions vs actual position and desired torque vs actual torque")
    subfigs = fig.subfigures(1, 2, wspace=0.07)

    labels = np.array(["time [s]", "position [m]"])
    labels_positions = np.array(["x", "y", "z"])

    ax1 = subfigs[0].subplots(3, 1, sharex=True)
    subfigs[0].suptitle("Position desired vs actual")
    subfigs[0].subplots_adjust(left=0.20, hspace=0.4, bottom=0.15, right=0.95)
    for i, ax in enumerate(ax1):
        ax.plot(t, des_leg_pos_tab[:, i], label="desired leg position for " + labels_positions[i])
        ax.plot(t, leg_pos_tab[:, i], label="actual leg position for " + labels_positions[i], color="r")
        ax.grid(True)
        if i == 1:
            ax.set_ylabel(labels[1])
        ax.legend()
    plt.xlabel(labels[0])

    labels = np.array(["time [s]", "torque [N/m]"])
    labels_joint = np.array(["hip", "thigh", "calf"])

    ax2 = subfigs[1].subplots(3, 1, sharex=True)
    subfigs[1].suptitle("Torque desired vs actual")
    subfigs[1].subplots_adjust(left=0.15, hspace=0.4, bottom=0.15, right=0.95)
    for i, ax in enumerate(ax2):
        ax.plot(t, des_leg_torque_tab[:, i], label="desired leg torque for " + labels_joint[i])
        ax.plot(t, leg_torque_tab[:, i], label="actual leg torque for " + labels_joint[i], color="r")
        ax.grid(True)
        if i == 1:
            ax.set_ylabel(labels[1])
        ax.legend()
    plt.xlabel(labels[0])

# Plot robot speed on x, y and z and its position displacement
if plot_speed_pos:
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Speeds and displacement")
    labels = np.array(["time [s]", "speed [m/s]"])
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    labels_speed = np.array(["Vx", "Vy", "Vz"])
    ax1 = subfigs[0].subplots(3, sharex=True)
    subfigs[0].suptitle("Speeds in x,y and z")
    subfigs[0].subplots_adjust(left=0.20, hspace=0.4, bottom=0.15, right=0.95)
    for i, ax in enumerate(ax1):
        ax.plot(t, robot_speed_tab[:, i], label=labels_speed[i])
        ax.legend()
        if i == 1:
            ax.set_ylabel(labels[1])
    plt.xlabel(labels[0])

    labels = np.array(["x position", "y position"])
    ax2 = subfigs[1].subplots(1)
    subfigs[1].suptitle("Displacement")
    subfigs[1].subplots_adjust(left=0.15, hspace=0.4, bottom=0.15, right=0.95)
    ax2.plot(robot_pos_tab[:, 0], robot_pos_tab[:, 1])
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[1])

speed_xy = robot_speed_tab[:, :2]
lin_speed = np.sum(np.abs(speed_xy)**2, axis=-1)**(1./2)
print(f"max speed is {lin_speed.max()} [m/s]")

n = 100
avg_cot = np.mean(CoT_tab[-100:])

print(f"Cost of transport over the last {n} values of the run: {avg_cot}")

plt.show()





