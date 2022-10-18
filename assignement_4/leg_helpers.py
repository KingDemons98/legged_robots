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

"""Defines some robot leg functions."""

import numpy as np

def jacobian_rel(q,l1=0.209,l2=0.195):
    """ Jacobian based on relative angles (like URDF)
        Input: motor angles (array), link lengths
        return: jacobian, foot position
    """
    # Jacobian 
    J = np.zeros((2,2))
    J[0,0] = -l1 * np.cos(q[0]) - l2 * np.cos(q[0] + q[1])
    J[1,0] =  l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    J[0,1] = -l2 * np.cos(q[0] + q[1]) 
    J[1,1] =  l2 * np.sin(q[0] + q[1])
    
    # foot pos
    pos = np.zeros(2)
    pos[0] =  -l1 * np.sin(q[0]) - l2 * np.sin(q[0]+q[1])
    pos[1] =  -l1 * np.cos(q[0]) - l2 * np.cos(q[0]+q[1])

    return J, pos

def ik_geometrical(xz,angleMode="<",l1=0.209,l2=0.195):
    """ Inverse kinematics based on geometrical reasoning.
        Input: Desired foot xz position (array) 
               angleMode (whether leg should look like > or <) 
               link lengths
        return: joint angles
    """
    sideSign = -1
    if angleMode == ">":
        sideSign = 1

    q = np.zeros(2)
    q[1] = sideSign * np.arccos( ((xz[0])**2 + (xz[1])**2 - l1**2 - l2**2) / (2*l1*l2) )
    q[0] = np.arctan2(-xz[0],-xz[1]) - np.arctan2( l2 * np.sin(q[1]) ,  (l1 + l2*np.cos(q[1])) )

    return q