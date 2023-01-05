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

"""
Run stable baselines 3 on quadruped env 
Check the documentation! https://stable-baselines3.readthedocs.io/en/master/
"""
import os
from datetime import datetime
# stable baselines 3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

import wandb
from wandb.integration.sb3 import WandbCallback

import stable_baselines3.common.noise as sb3_noise

# utils
from utils.utils import CheckpointCallback
from utils.file_utils import get_latest_model
# gym environment
from env.quadruped_gym_env import QuadrupedGymEnv

project_name = "SAC-sweep"

def main():
    NUM_ENVS = 10    # how many pybullet environments to create for data collection

    motor_control_mode = "CARTESIAN_PD"                          ##### SET MOTOR CONTROL HERE

    ########
    env_configs = {"motor_control_mode": motor_control_mode,
                   "observation_space_mode": "LR_COURSE_OBS_V1",
                   "task_env": "LR_COURSE_TASK_V1",
                   "test_env": False,
                   # "render": True
                   }
    ###########################################
    # env_configs = {}

    interim_dir = "./logs/sweeps/"

    # directory to save policies and normalization parameters
    NAME = motor_control_mode + "regular_forward" + datetime.now().strftime("%m%d%y%H%M%S")
    SAVE_PATH = interim_dir + NAME + '/'
    os.makedirs(SAVE_PATH, exist_ok=True)
    # checkpoint to save policy network periodically
    checkpoint_callback = CheckpointCallback(save_freq=20000//NUM_ENVS, save_path=SAVE_PATH, name_prefix='rl_model', verbose=2)
    # create Vectorized gym environment
    env = lambda: QuadrupedGymEnv(**env_configs)
    env = make_vec_env(env, monitor_dir=SAVE_PATH, n_envs=NUM_ENVS)
    # normalize observations to stabilize learning (why?)

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)


    run = wandb.init(
        name=NAME,
        project=project_name,
        config=env_configs,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
    )

    # Multi-layer perceptron (MLP) policy of two layers of size ,
    policy_kwargs = dict(net_arch=[256,256])
    n_steps = 4096

    sac_config={"learning_rate":wandb.config.learning_rate,
                "buffer_size":300000,
                "batch_size":wandb.config.batch_size,
                "ent_coef":'auto',
                "gamma":0.99,
                "tau":0.005,
                "train_freq":wandb.config.train_freq,
                "gradient_steps":1,
                "learning_starts": 10000,
                "verbose":1,
                "tensorboard_log":f"runs",
                "policy_kwargs": policy_kwargs,
                "seed":None,
                "device": "auto"}

    model = SAC('MlpPolicy', env, **sac_config)
    #run.config.update(sac_config)


    # Learn and save (may need to train for longer)
    model.learn(total_timesteps=200000, log_interval=1, callback=[WandbCallback(), checkpoint_callback])
    # Don't forget to save the VecNormalize statistics when saving the agent
    model.save(os.path.join(SAVE_PATH, "rl_model") )
    env.save(os.path.join(SAVE_PATH, "vec_normalize.pkl"))
    model.save_replay_buffer(os.path.join(SAVE_PATH, "off_policy_replay_buffer"))

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'rollout/ep_rew_mean'},
    'parameters':
    {
        'batch_size': {'values': [128, 256, 512]},
        'train_freq': {'values': [1, 2, 3, 4, 5]},
        'learning_rate': {'values': [0.0001, 0.001, 0.01, 0.1]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

wandb.agent(sweep_id, function=lambda: main(), count=50)