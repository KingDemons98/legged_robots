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
CPG in polar coordinates based on: 
CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion 
authors: Bellegarda, Ijspeert
https://ieeexplore.ieee.org/abstract/document/9932888
"""
import numpy as np

# for RL 
MU_LOW = 1
MU_UPP = 2

foot_y = 0.0838
sideSign = np.array([-1, 1, -1, 1])

class HopfNetwork():
  """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.  

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)
  """
  def __init__(self,
                mu=1**2,                 # intrinsic amplitude, converges to sqrt(mu)
                omega_swing=5*2*np.pi,   # frequency in swing phase (can edit)
                omega_stance=2*2*np.pi,  # frequency in stance phase (can edit)
                gait="TROT",             # Gait, can be TROT, WALK, PACE, BOUND, etc.
                alpha=50,                # amplitude convergence factor
                coupling_strength=1,     # coefficient to multiply coupling matrix
                couple=True,             # whether oscillators should be coupled
                time_step=0.001,         # time step
                ground_clearance=0.07,   # foot swing height 
                ground_penetration=0.01, # foot stance penetration into ground 
                robot_height=0.3,        # in nominal case (standing) 
                des_step_len=0.05,       # desired step length
                max_step_len_rl=0.1,     # max step length, for RL scaling 
                use_RL=False,            # whether to learn parameters with RL
               comparison=False

                ):
    
    ###############
    self.comparison = comparison
    # initialize CPG data structures: amplitude is row 0, and phase is row 1
    if use_RL and not self.comparison:
      self.X = np.zeros((3, 4))
      self.X_dot = np.zeros((3, 4))
    else:
      self.X = np.zeros((2, 4))
      self.X_dot = np.zeros((2, 4))

    # save parameters 
    self._mu = mu
    self._omega_swing = omega_swing
    self._omega_stance = omega_stance  
    self._alpha = alpha
    self._couple = couple
    self._coupling_strength = coupling_strength
    self._dt = time_step
    self._set_gait(gait)

    # set oscillator initial conditions  
    self.X[0,:] = np.random.rand(4) * .1
    # self.X[0,:] = np.ones((4)) * 0.1
    self.X[1,:] = self.PHI[0,:]
    if use_RL and not self.comparison:
      self.X[2, :] = np.random.rand(4) * .001#TODO init this

    # save body and foot shaping
    self._ground_clearance = ground_clearance 
    self._ground_penetration = ground_penetration
    self._robot_height = robot_height 
    self._des_step_len = des_step_len

    # for RL
    self.use_RL = use_RL
    self._omega_rl = np.zeros(4)
    self._mu_rl = np.zeros(4)
    self._psi_rl = np.zeros(4)
    self._max_step_len_rl = max_step_len_rl
    if use_RL:
      self.X[0,:] = MU_LOW # mapping MU_LOW=1 to MU_UPP=2



  def _set_gait(self,gait):
    """ For coupling oscillators in phase space. 
    [TODO] update all coupling matrices
    """
    self.PHI_trot = np.zeros((4,4))
    self.PHI_walk = np.zeros((4,4))
    self.PHI_bound = np.zeros((4,4))
    self.PHI_pace = np.zeros((4,4))
    self.PHI_rotary_gallop = np.zeros((4, 4))

    self.PHI_trot = np.array([[0.0, 0.5, 0.5, 0.0], [0.5, 0.0, 0.0, 0.5], [0.5, 0.0, 0.0, 0.5], [0.0, 0.5, 0.5, 0.0]]) * 2 * np.pi
    self.PHI_walk = np.array([[0.0, 0.5, -0.25, 0.25], [0.5, 0.0, 0.25, 0.75], [0.25, -0.25, 0.0, 0.5], [-0.25, -0.75, 0.5, 0.0]]) * 2 * np.pi
    self.PHI_bound = np.array([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]) * 2 * np.pi
    self.PHI_pace = np.array([[0.0, 0.5, 0.0, 0.5], [0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5], [0.5, 0.5, 0.0, 0.0]]) * 2 * np.pi
    self.PHI_rotary_gallop = np.array([[0.0, 0.1, -0.4, -0.5], [-0.1, 0.0, -0.5, -0.6], [0.4, 0.5, 0.0, -0.1], [0.5, 0.6, 0.1, 0.0]]) * 2 * np.pi

    if gait == "TROT":
      self.PHI = self.PHI_trot
    elif gait == "PACE":
      self.PHI = self.PHI_pace
    elif gait == "BOUND":
      self.PHI = self.PHI_bound
    elif gait == "WALK":
      self.PHI = self.PHI_walk
    elif gait == "ROTARY_GALLOP":
      self.PHI = self.PHI_rotary_gallop
    else:
      raise ValueError( gait + ' not implemented.')


  def update(self):
    """ Update oscillator states. """

    # update parameters, integrate
    if not self.use_RL:
      self._integrate_hopf_equations()
    else:
      self._integrate_hopf_equations_rl()
    
    # map CPG variables to Cartesian foot xz positions (Equations 8, 9) 
    x = np.zeros(4)
    if not self.comparison:
      y = np.zeros(4)
    else:
      y = np.zeros(3)
    z = np.zeros(4)
    if self.use_RL:
      r = np.clip(self.X[0, :], MU_LOW, MU_UPP)

    for i in range(len(self.X[0])):
      theta = self.X[1, i]
      if self.use_RL:
        phi = self.X[2, i]
        x[i] = -self._max_step_len_rl * (r[i] - MU_LOW) * np.cos(theta) * np.cos(phi)
        # y[i] = sideSign[i] * foot_y -self._max_step_len_rl * (r[i] - MU_LOW) * np.cos(theta) * np.sin(phi)
        if self.comparison:
          y[i] = sideSign[i] * foot_y
        else:
          y[i]= -self._max_step_len_rl * (r[i] - MU_LOW) * np.cos(theta) * np.sin(phi)

      else:
        r = self.X[0, i]
        x[i] = -self._des_step_len * r * np.cos(theta)
      if np.sin(theta) > 0.0:
        z[i] = -self._robot_height + self._ground_clearance * np.sin(theta)
      else:
        z[i] = -self._robot_height + self._ground_penetration * np.sin(theta)

    # scale x by step length
    if not self.use_RL:
      # use des step len, fixed
      return x, z
    else:
      return x, y, z

      
        
  def _integrate_hopf_equations(self):
    """ Hopf polar equations and integration. Use equations 6 and 7. """
    # bookkeeping - save copies of current CPG states 
    X = self.X.copy()
    X_dot_prev = self.X_dot.copy() 
    X_dot = np.zeros((2,4))

    # loop through each leg's oscillator
    for i in range(4):
      # get r_i, theta_i from X
      r, theta = X[0, i], X[1, i]
      # compute r_dot (Equation 6)
      r_dot = self._alpha * (self._mu - r **2) * r
      # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
      if (theta %(2 * np.pi) < np.pi):
        theta_dot = self._omega_swing
      else:
        theta_dot = self._omega_stance
      # loop through other oscillators to add coupling (Equation 7)
      if self._couple:
        for j in range(len(X[0, :])):
          theta_dot += X[0, j] * self._coupling_strength * np.sin(X[1, j] - theta - self.PHI[i][j])

      # set X_dot[:,i]
      X_dot[:,i] = [r_dot, theta_dot]

    # integrate 
    self.X = X + (X_dot + X_dot_prev) * self._dt / 2
    self.X_dot = X_dot
    # mod phase variables to keep between 0 and 2pi
    self.X[1,:] = self.X[1,:] % (2*np.pi)


  ###################### Helper functions for accessing CPG States
  def get_r(self):
    """ Get CPG amplitudes (r) """
    return self.X[0,:]

  def get_theta(self):
    """ Get CPG phases (theta) """
    return self.X[1,:]

  def get_dr(self):
    """ Get CPG amplitude derivatives (r_dot) """
    return self.X_dot[0,:]

  def get_dtheta(self):
    """ Get CPG phase derivatives (theta_dot) """
    return self.X_dot[1,:]

  def get_phi(self):
    return self.X[2, :]

  def get_dphi(self):
    return self.X_dot[2, :]

  ###################### Functions for setting parameters for RL
  def set_omega_rl(self, omegas):
    """ Set intrinisc frequencies. """
    self._omega_rl = omegas 

  def set_mu_rl(self, mus):
    """ Set intrinsic amplitude setpoints. """
    self._mu_rl = mus

  def set_psi_rl(self, psis):
    self._psi_rl = psis
  def _integrate_hopf_equations_rl(self):
    """ Hopf polar equations and integration, using quantities set by RL """
    # bookkeeping - save copies of current CPG states 
    X = self.X.copy()
    X_dot_prev = self.X_dot.copy()
    if self.comparison:
      X_dot = np.zeros((2, 4))
    else:
      X_dot = np.zeros((3, 4))

    # loop through each leg's oscillator, find current velocities
    for i in range(4):
      # get r_i, theta_i from X
      if self.comparison:
        r, theta = X[:, i]
      else:
        r, theta, phi = X[:, i]
      # amplitude (use mu from RL, i.e. self._mu_rl[i])
      r_dot = self._alpha * (self._mu_rl[i] - r **2) * r
      # phase (use omega from RL, i.e. self._omega_rl[i])
      theta_dot = self._omega_rl[i]
      if not self.comparison:
        phi_dot = self._psi_rl[i]
        X_dot[:, i] = [r_dot, theta_dot, phi_dot]
      else:
        X_dot[:, i] = [r_dot, theta_dot]

    # integrate 
    self.X = X + (X_dot_prev + X_dot) * self._dt / 2
    self.X_dot = X_dot
    self.X[1,:] = self.X[1,:] % (2*np.pi)